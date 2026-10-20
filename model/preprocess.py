from typing import List, Optional, Union
import os

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from scipy.sparse.linalg import eigsh
from threadpoolctl import threadpool_limits


def _canonical_rank(obs_names, n: int) -> np.ndarray:
    """Stable node ranks used to resolve equal-distance/eigenvalue ties."""
    if obs_names is None:
        return np.arange(n, dtype=np.int64)
    names = np.asarray(obs_names, dtype="U")
    if names.shape != (n,):
        raise ValueError(f"obs_names must have shape ({n},), got {names.shape}")
    if np.unique(names).size != n:
        raise ValueError("obs_names must be unique")
    order = np.argsort(names, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n, dtype=np.int64)
    return rank


def _canonical_knn_adjacency(spatial, k: int, obs_names=None) -> sp.csr_matrix:
    """Build a device-independent directed KNN graph with an explicit tie rule."""
    from sklearn.neighbors import NearestNeighbors

    xy = np.asarray(spatial, dtype=np.float64)
    n = xy.shape[0]
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("spatial must be an (N, 2) array")
    if not 1 <= int(k) < n:
        raise ValueError(f"KNN k={k} invalid for {n} spots")
    rank = _canonical_rank(obs_names, n)
    d2, _ = NearestNeighbors(n_neighbors=2).fit(xy).kneighbors(xy)
    spacing = float(np.median(d2[:, 1]))
    if not np.isfinite(spacing) or spacing <= 0:
        spacing = 1.0
    # Query a small margin first; widen only when the kth rounded-distance
    # tie crosses the query boundary.
    m = min(n, int(k) + 31)
    dist, idx = NearestNeighbors(n_neighbors=m).fit(xy).kneighbors(xy)
    rows = np.empty(n * int(k), dtype=np.int64)
    cols = np.empty(n * int(k), dtype=np.int64)
    for i in range(n):
        keep = idx[i] != i
        dd = np.round(dist[i][keep] / spacing, 6)
        jj = idx[i][keep]
        if len(dd) > int(k) and dd[-1] <= dd[int(k) - 1] and m < n:
            full_d, full_j = NearestNeighbors(n_neighbors=n).fit(xy).kneighbors(xy[i:i + 1])
            keep2 = full_j[0] != i
            dd = np.round(full_d[0][keep2] / spacing, 6)
            jj = full_j[0][keep2]
        order = np.lexsort((rank[jj], dd))
        jj = jj[order[:int(k)]]
        rows[i * int(k):(i + 1) * int(k)] = i
        cols[i * int(k):(i + 1) * int(k)] = jj
    A = sp.coo_matrix((np.ones(n * int(k), dtype=np.float32), (rows, cols)), shape=(n, n)).tocsr()
    A.sort_indices()
    return A

def Cal_Spatial_Net(
    adata: AnnData,
    rad_cutoff: Optional[float] = None,
    k_cutoff: Optional[int] = None,
    model: str = "KNN",
    return_data: bool = False,
    graph_dir: Optional[str] = None,
    save_graph_csv: bool = True,
):

    print("🚀 Calculating spatial neighbor graph ...")
    n_cells = adata.shape[0]

    if model == "KNN":
        # CPU float64 construction with a stable distance/rank tie rule avoids
        # CPU/GPU torch_geometric tie-order differences.
        A = _canonical_knn_adjacency(
            np.asarray(adata.obsm["spatial"], dtype=np.float64),
            int(k_cutoff),
            obs_names=np.asarray(adata.obs_names, dtype="U"),
        )
    elif model == "Radius":
        from sklearn.neighbors import NearestNeighbors

        xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
            raise ValueError("spatial must be a finite (N, 2) array")
        if rad_cutoff is None or not np.isfinite(rad_cutoff) or rad_cutoff <= 0:
            raise ValueError("radius must be finite and positive")
        if n_cells < 2:
            raise ValueError("radius graph needs at least two spots")
        neighbors = NearestNeighbors(radius=rad_cutoff).fit(xy).radius_neighbors(xy, return_distance=False)
        rows, cols = [], []
        for i, indices in enumerate(neighbors):
            indices = indices[indices != i]
            if len(indices) == 0:
                raise ValueError(f"radius={rad_cutoff} leaves spot {adata.obs_names[i]} without a spatial neighbor")
            rows.extend([i] * len(indices))
            cols.extend(indices)
        A = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_cells, n_cells))
        A.sort_indices()
    else:
        raise ValueError("❌ Unknown model type. Use 'KNN' or 'Radius'.")

    row, col = A.nonzero()
    G = pd.DataFrame({"Cell1": row, "Cell2": col})

    G["Cell1"] = G["Cell1"].astype(int)
    G["Cell2"] = G["Cell2"].astype(int)
    adata.uns["Spatial_Net"] = G

    if not save_graph_csv:
        print("Skip saving Spatial_Net CSV.")
    else:
        if graph_dir is not None:
            base_dir = graph_dir
            base_name = (
                os.path.basename(adata.filename).split(".h5ad")[0]
                if getattr(adata, "filename", None)
                else "adata"
            )
        else:
            if getattr(adata, "filename", None):
                base_dir = os.path.dirname(adata.filename)
                base_name = os.path.basename(adata.filename).split(".h5ad")[0]
            elif "source" in adata.uns:
                base_dir = os.path.dirname(adata.uns["source"])
                base_name = os.path.basename(adata.uns["source"]).split(".h5ad")[0]
            else:
                base_dir = "."
                base_name = "adata"

            base_dir = os.path.join(base_dir, "SpatialGraphs")

        os.makedirs(base_dir, exist_ok=True)
        csv_path = os.path.join(base_dir, f"{base_name}_spatial_net.csv")
        G.to_csv(csv_path, index=False)
        print(f"💾 Spatial graph saved to: {csv_path}")

    print(f"✅ The graph contains {A.nnz} edges, {n_cells} cells.")
    print(f"Average neighbors per cell: {A.nnz / n_cells:.2f}")

    if return_data:
        return adata

def compute_pseudo_layer_ids_from_spatial(
    spatial,
    n_bins: int,
    knn_k: int = 18,
    mode: str = "fiedler",
    binning: str = "quantile",
    obs_names=None,
):
    """
    Build pseudo-layer ids from spatial coordinates using graph Laplacian only.
    Only `mode='fiedler'` is supported to enforce spectral pseudo-depth.
    """

    import numpy as np

    spatial = np.asarray(spatial)
    assert spatial.ndim == 2 and spatial.shape[1] == 2, "spatial must be (N,2)"
    N = spatial.shape[0]
    if n_bins <= 1:
        return np.zeros(N, dtype=np.int64), np.zeros(N, dtype=np.float32)

    if mode.lower() != "fiedler":
        raise ValueError(f"Unsupported mode={mode}. Use mode='fiedler'.")

    dense_bytes = 8 * N * N

    # ---- 1) build spatial kNN graph (undirected) ----
    directed = _canonical_knn_adjacency(spatial, int(knn_k), obs_names=obs_names)
    row, col = directed.nonzero()

    # Symmetrize adjacency
    row2 = np.concatenate([row, col])
    col2 = np.concatenate([col, row])
    data = np.ones_like(row2, dtype=np.float32)
    A = sp.csr_matrix((data, (row2, col2)), shape=(N, N))
    A.data[:] = 1.0

    # ---- 2) graph Laplacian L = D - A ----
    deg = np.asarray(A.sum(axis=1)).reshape(-1)
    D = sp.diags(deg, format="csr")
    L = (D - A).astype(np.float64)

    rank = _canonical_rank(obs_names, N)
    from scipy.sparse.csgraph import connected_components
    n_components, _ = connected_components(A, directed=False, return_labels=True)
    if n_components != 1:
        raise ValueError(f"Fiedler graph must be connected; found {n_components} components")
    if dense_bytes <= 256 * 1024**2:
        from scipy.linalg import eigh
        with threadpool_limits(limits=1):
            vals, vecs = eigh(L.toarray(), subset_by_index=[0, min(2, N - 1)], driver="evr")
        if vals.shape[0] < 3 or vals[1] - vals[0] <= 1e-9 or vals[2] - vals[1] <= 1e-9:
            raise ValueError("Fiedler vector is not unique for this spatial graph")
        fiedler = vecs[:, 1].astype(np.float64)
    else:
        v0 = np.cos((rank + 0.5) * np.pi / N)
        with threadpool_limits(limits=1):
            vals, vecs = eigsh(L, k=3, which="SM", v0=v0,
                               tol=1e-8, maxiter=max(5000, N * 5))
        order = np.argsort(vals)
        if vals[order[1]] - vals[order[0]] <= 1e-9 or vals[order[2]] - vals[order[1]] <= 1e-9:
            raise ValueError("Fiedler vector is not unique for this spatial graph")
        fiedler = vecs[:, order[1]].astype(np.float64)
        residual = np.linalg.norm(L @ fiedler - vals[order[1]] * fiedler)
        if residual > 1e-5:
            raise RuntimeError(f"Fiedler residual too large: {residual}")
    rx = float(np.corrcoef(fiedler, spatial[:, 0])[0, 1])
    ry = float(np.corrcoef(fiedler, spatial[:, 1])[0, 1])
    if abs(rx) >= 0.1:
        sign = np.sign(rx)
    elif abs(ry) >= 0.1:
        sign = np.sign(ry)
    else:
        tied = np.flatnonzero(np.abs(fiedler) == np.max(np.abs(fiedler)))
        sign = np.sign(fiedler[tied[np.argmin(rank[tied])]])
    fiedler *= sign if sign != 0 else 1.0

    # Normalize to a stable 0~1 pseudo-depth (rank-based is more robust)
    # This also removes sign ambiguity of eigenvectors.
    order = np.lexsort((rank, fiedler))
    ranks = np.empty(N, dtype=np.int64)
    ranks[order] = np.arange(N, dtype=np.int64)
    pseudo_depth = ranks.astype(np.float32) / max(1, N - 1)

    # ---- 4) binning into pseudo layers ----
    if binning.lower() == "quantile":
        # equal number of points per bin (robust to density variation)
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(pseudo_depth, qs)
        # avoid identical edges causing empty bins
        edges[0] -= 1e-6
        edges[-1] += 1e-6
    elif binning.lower() == "uniform":
        edges = np.linspace(pseudo_depth.min() - 1e-6, pseudo_depth.max() + 1e-6, n_bins + 1)
    else:
        raise ValueError(f"Unsupported binning={binning}. Use 'quantile' or 'uniform'.")

    layer_ids = np.digitize(pseudo_depth, edges[1:-1], right=False).astype(np.int64)
    layer_ids = np.clip(layer_ids, 0, n_bins - 1)

    return layer_ids, pseudo_depth


def scanpy_workflow(
    adata: AnnData,
    filter_cell: Optional[bool] = False,
    min_gene: Optional[int] = 200,
    min_cell: Optional[int] = 30,
    call_hvg: Optional[bool] = True,
    n_top_genes: Optional[Union[int, List]] = 3000,
    batch_key: Optional[str] = None,
    n_comps: Optional[int] = 50,
    viz: Optional[bool] = False,
    resolution: Optional[float] = 0.8,
) -> AnnData:

    matrices = {"X": adata.X, **{f"layers/{key}": value for key, value in adata.layers.items()}}
    if adata.raw is not None:
        matrices["raw.X"] = adata.raw.X
    for name, matrix in matrices.items():
        if matrix is not None:
            values = matrix.data if sp.issparse(matrix) else np.asarray(matrix)
            if not np.isfinite(values).all():
                raise ValueError(f"{name} must be finite")

    source = next((key for key in ("counts", "count") if key in adata.layers), None)
    if source is not None:
        if not looks_like_counts(adata.layers[source]):
            raise ValueError(f"layers['{source}'] must contain nonnegative integer counts")
        adata.X = adata.layers[source].copy()
        source = f"layers/{source}"
    elif getattr(adata, "raw", None) is not None:
        raw_x = adata.raw.X
        if looks_like_counts(raw_x):
            adata = adata.raw.to_adata()
            source = "raw/X"
        elif looks_like_counts(adata.X):
            source = "X"
        elif call_hvg and isinstance(n_top_genes, int):
            raise ValueError("raw.X is not count-like and X does not contain valid counts")
        else:
            source = "X/transformed"
    elif looks_like_counts(adata.X):
        source = "X"
    elif call_hvg and isinstance(n_top_genes, int):
        raise ValueError("seurat_v3 HVG requires raw counts in layers['counts'], layers['count'], raw.X, or X")
    else:
        source = "X/transformed"
    start_from_counts = source != "X/transformed"
    adata.uns["sasg_input_source"] = source
    print(f"Expression source: {source}")

    # --- Step 2: optional filtering ---
    if filter_cell:
        sc.pp.filter_cells(adata, min_genes=min_gene)
        sc.pp.filter_genes(adata, min_cells=min_cell)

    # --- Step 3: highly variable genes selection ---
    if call_hvg:
        if isinstance(n_top_genes, int):
            if adata.n_vars > n_top_genes:
                sc.pp.highly_variable_genes(
                    adata,
                    n_top_genes=n_top_genes,
                    flavor="seurat_v3",
                    batch_key=batch_key,
                    layer=None,
                )
            else:
                adata.var["highly_variable"] = True
                print("All genes are marked as highly variable.")
        elif isinstance(n_top_genes, list):
            adata.var["highly_variable"] = False
            n_top_genes = list(set(adata.var.index).intersection(set(n_top_genes)))
            adata.var.loc[n_top_genes, "highly_variable"] = True
    else:
        print("Skipping highly variable gene selection.")

    # --- Step 3.5: subset to HVGs ---
    if call_hvg and "highly_variable" in adata.var:
        hv = adata.var["highly_variable"].values
        if hv.sum() > 0 and hv.sum() < adata.n_vars:
            adata = adata[:, hv].copy()
            print(f"✅ Subset to HVGs: {hv.sum()} genes")

    # --- Step 4: detect if negative ---
    try:
        has_neg = (adata.X.min() < 0) if sp.issparse(adata.X) else (np.min(adata.X) < 0)
    except Exception:
        X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        has_neg = (np.min(X) < 0)

    # --- Step 5: normalize/log1p ONCE for count-like data ---
    is_counts_like = False
    if not has_neg:
        try:
            is_counts_like = bool(start_from_counts or looks_like_counts(adata.X))
        except Exception:
            is_counts_like = bool(start_from_counts)

    if (not has_neg) and is_counts_like:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        print("⚠️ Skip normalize/log1p. has_neg=", has_neg, " start_from_counts=", start_from_counts,
              " looks_like_counts=", is_counts_like)

    # --- Step 6: convert sparse to dense ---
    if sp.issparse(adata.X):
        adata.X = adata.X.toarray()
    if not np.isfinite(adata.X).all():
        raise FloatingPointError("Non-finite expression after normalization")

    # --- Step 7: scale once only ---
    sc.pp.scale(adata, max_value=10)
    if not np.isfinite(adata.X).all():
        raise FloatingPointError("Non-finite expression after scaling")

    # --- Step 8: PCA ---
    if n_comps > 0:
        sc.tl.pca(adata, n_comps=n_comps, svd_solver="arpack")
        if not np.isfinite(adata.obsm["X_pca"]).all():
            raise FloatingPointError("Non-finite PCA features")

    # --- Step 9: optional visualization and clustering ---
    if viz:
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=n_comps)
        sc.tl.umap(adata)
        sc.tl.leiden(
            adata,
            resolution=resolution,
            flavor="igraph",
            directed=False,
            n_iterations=2,
        )

    return adata

def looks_like_counts(X, max_check=200000):
    if X is None:
        return False
    values = X.data if sp.issparse(X) else np.asarray(X)
    values = np.asarray(values)
    if values.size == 0 or not np.issubdtype(values.dtype, np.number):
        return False
    flat = values.reshape(-1)
    for start in range(0, flat.size, int(max_check)):
        block = flat[start:start + int(max_check)]
        if (not np.isfinite(block).all() or (block < 0).any() or
                not np.allclose(block, np.round(block), rtol=0.0, atol=1e-3)):
            return False
    return True

def compute_local_normals_from_spatial(spatial: np.ndarray, knn_k: int = 10, obs_names=None):

    spatial = np.asarray(spatial, dtype=np.float32)
    assert spatial.ndim == 2 and spatial.shape[1] == 2

    N = spatial.shape[0]
    xy = np.asarray(spatial, dtype=np.float64)
    rank = _canonical_rank(obs_names, N)
    adjacency = _canonical_knn_adjacency(xy, int(knn_k), obs_names=obs_names)

    normals = np.zeros((N, 2), dtype=np.float32)
    scales  = np.zeros((N,), dtype=np.float32)

    for i in range(N):
        neighbors = adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i + 1]]
        distance = ((xy[neighbors] - xy[i]) ** 2).sum(axis=1)
        neighbors = neighbors[np.lexsort((rank[neighbors], distance))]
        nb = spatial[neighbors]  # (k,2)
        X = nb - nb.mean(axis=0, keepdims=True)  # center
        C = (X.T @ X) / max(1, X.shape[0])       # 2x2 covariance

        # eigen-decomposition (2D small)
        vals, vecs = np.linalg.eigh(C)
        n = vecs[:, np.argmin(vals)]             # smallest eigenvector = normal
        n = n / (np.linalg.norm(n) + 1e-12)

        # robust scale along normal direction
        proj = np.abs((nb - spatial[i]) @ n)     # (k,)
        s = np.median(proj) + 1e-6

        normals[i] = n.astype(np.float32)
        scales[i]  = float(s)

    return normals, scales
