import argparse
import hashlib
import json
import os
import uuid
import numpy as np
import scipy.sparse as sp
import anndata as ad
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from threadpoolctl import threadpool_limits

from model.augment import build_augmented_graph
from model.preprocess import (
    Cal_Spatial_Net,
    scanpy_workflow,
    compute_pseudo_layer_ids_from_spatial,
    compute_local_normals_from_spatial
)
from model.encoder import (
    scipy_to_torch_coo,
    normalize_rows_absolute_torch,
    masked_spot_component_operator,
    RelationPairs,
    CollaborativeEncoder,
    CrossLayerScorer,
    sample_negatives,
    node_embedding_from_layers,
    FeatureDecoder,
    InsufficientNegativeSupportError,
)


def _validate_args(a, n_spot: int, n_pca: int | None = None):
    """Reject impossible or ambiguous requests before graph/training work starts."""
    errors = []
    def finite(value):
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            return False

    if int(a.epochs) <= 0:
        errors.append(f"epochs={a.epochs} must be a positive integer")
    for name in ("alpha", "lr", "lambda_recon", "weight_decay", "layer_gamma",
                 "normal_gamma", "normal_margin", "neg_adj_ratio", "feat_noise_std"):
        if not finite(getattr(a, name, 0.0)):
            errors.append(f"{name} must be finite")
    if finite(getattr(a, "lambda_recon", 0.0)) and float(a.lambda_recon) < 0:
        errors.append("lambda_recon must be >= 0")
    if finite(getattr(a, "weight_decay", 0.0)) and float(a.weight_decay) < 0:
        errors.append("weight_decay must be >= 0")
    if finite(getattr(a, "layer_gamma", 0.0)) and float(a.layer_gamma) < 0:
        errors.append("layer_gamma must be >= 0")
    if finite(getattr(a, "layer_gamma", 0.0)) and float(a.layer_gamma) > 20:
        errors.append("layer_gamma must be <= 20 for stable sampling weights")
    if finite(getattr(a, "normal_gamma", 0.0)) and float(a.normal_gamma) < 0:
        errors.append("normal_gamma must be >= 0")
    if finite(getattr(a, "normal_gamma", 0.0)) and float(a.normal_gamma) > 20:
        errors.append("normal_gamma must be <= 20 for stable sampling weights")
    if finite(getattr(a, "normal_margin", 0.0)) and float(a.normal_margin) < 0:
        errors.append("normal_margin must be >= 0")
    if float(getattr(a, "neg_layer_margin", 0)) < 0:
        errors.append("neg_layer_margin must be >= 0")
    neg_adj_ratio = getattr(a, "neg_adj_ratio", 0.3)
    if finite(neg_adj_ratio) and not 0.0 <= float(neg_adj_ratio) <= 1.0:
        errors.append("neg_adj_ratio must lie in [0, 1]")
    if int(a.K) < 1:
        errors.append(f"K={a.K} must be >= 1")
    if not 1 <= int(a.k) < int(n_spot):
        errors.append(f"k={a.k} must be in [1, {int(n_spot) - 1}]")
    if not 0.0 < float(a.alpha) < 1.0:
        errors.append(f"alpha={a.alpha} must lie in (0, 1)")
    if not 0.0 < float(a.mask_ratio_feat) < 1.0:
        errors.append(f"mask_ratio_feat={a.mask_ratio_feat} must lie in (0, 1)")
    if not 0.0 <= float(a.neg_hard_ratio) <= 1.0:
        errors.append(f"neg_hard_ratio={a.neg_hard_ratio} must lie in [0, 1]")
    if int(a.neg_oversample) < 1:
        errors.append(f"neg_oversample={a.neg_oversample} must be >= 1")
    if int(getattr(a, "dim", 64)) < 1 or int(getattr(a, "hidden", 512)) < 4:
        errors.append("dim must be positive and hidden must be >= 4")
    if getattr(a, "embed_agg", "concat") not in {"concat", "mean", "last"}:
        errors.append("embed_agg must be concat, mean, or last")
    if str(getattr(a, "activation", "prelu")).lower() != "prelu":
        errors.append("the training entry point supports only prelu")
    if int(getattr(a, "pca_comps", 0)) < 1:
        errors.append("pca_comps must be >= 1")
    for name in ("pseudo_layer_knn", "normal_knn"):
        value = int(getattr(a, name, {"pseudo_layer_knn": 20, "normal_knn": 15}[name]))
        active = (name == "pseudo_layer_knn" and bool(getattr(a, "layer_aware", False))
                  and int(getattr(a, "pseudo_layer_bins", 0)) > 0) or (
                      name == "normal_knn" and bool(getattr(a, "normal_aware", False)))
        if value < 1 or (active and value >= int(n_spot)):
            errors.append(f"{name} must be in [1, {int(n_spot) - 1}]")
    if str(getattr(a, "pseudo_layer_binning", "quantile")).lower() not in {"quantile", "uniform"}:
        errors.append("pseudo_layer_binning must be 'quantile' or 'uniform'")
    bins = int(getattr(a, "pseudo_layer_bins", 0))
    if bins < 0 or bins > int(n_spot):
        errors.append(f"pseudo_layer_bins must be 0 or in [2, {int(n_spot)}]")
    if bool(getattr(a, "layer_aware", False)) and 0 < bins < 2 and int(getattr(a, "neg_layer_margin", 0)) > 0:
        errors.append("pseudo_layer_bins=1 is incompatible with a positive neg_layer_margin")
    if not finite(a.lr) or float(a.lr) <= 0.0:
        errors.append(f"lr={a.lr} must be finite and positive")
    if a.graph_model == "Radius" and not 0.0 < float(a.radius) < float("inf"):
        errors.append(f"radius={a.radius} must be finite and positive")
    if n_pca is not None:
        if int(a.pca_comps) > int(n_pca):
            errors.append(f"pca_comps={a.pca_comps} exceeds available PCA dimension {n_pca}")
        if int(a.topN) < 1 or int(a.topN) > int(n_pca):
            errors.append(f"topN={a.topN} must be between 1 and available PCA dimension {n_pca}")
    if errors:
        raise ValueError("Invalid SASG-ST settings: " + "; ".join(errors))


def set_seed(seed: int = 0, strict: bool = True):
    import random

    if strict:
        workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace not in (":4096:8", ":16:8"):
            raise ValueError(f"Invalid CUBLAS_WORKSPACE_CONFIG={workspace!r}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(bool(strict), warn_only=False)

def unique_trainable_params(*modules):
    seen = set()
    params = []
    for module in modules:
        if module is None:
            continue
        for p in module.parameters():
            if p is None or not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            params.append(p)
    return params


def _require_finite(value, name: str):
    if value is None:
        return
    if torch.is_tensor(value):
        ok = bool(torch.isfinite(value).all().item())
    else:
        try:
            ok = bool(np.isfinite(np.asarray(value)).all())
        except (TypeError, ValueError) as exc:
            raise FloatingPointError(f"Cannot validate {name} as a numeric array") from exc
    if not ok:
        raise FloatingPointError(f"Non-finite values detected in {name}")

def get_A_from_spatial_net(adata):
    G = adata.uns["Spatial_Net"].copy()

    if np.issubdtype(G["Cell1"].dtype, np.integer) and np.issubdtype(
        G["Cell2"].dtype, np.integer
    ):
        row, col = G["Cell1"].to_numpy(), G["Cell2"].to_numpy()
    else:
        cells = np.array(adata.obs_names)
        idx = dict(zip(cells, range(cells.shape[0])))
        G["Cell1"] = G["Cell1"].map(idx)
        G["Cell2"] = G["Cell2"].map(idx)
        row, col = G["Cell1"].to_numpy(), G["Cell2"].to_numpy()

    A = sp.coo_matrix(
        (np.ones(G.shape[0], dtype=np.float32), (row, col)),
        shape=(adata.n_obs, adata.n_obs),
        dtype=np.float32,
    )
    return A.tocsr()


def _cached_spatial_graphs(adata):
    cached = {}
    if "A_spatial" in adata.obsp:
        cached["obsp['A_spatial']"] = sp.csr_matrix(adata.obsp["A_spatial"]) != 0
    if "Spatial_Net" in adata.uns:
        cached["uns['Spatial_Net']"] = get_A_from_spatial_net(adata) != 0
    return cached


def _check_cached_graphs(cached, requested, a):
    requested = sp.csr_matrix(requested) != 0
    for name, graph in cached.items():
        difference = (graph != requested).nnz if graph.shape == requested.shape else -1
        if difference:
            raise ValueError(
                f"Cached spatial graph {name} differs from requested "
                f"graph_model={a.graph_model}, k={a.k}, radius={a.radius} "
                f"({difference} differing entries). Remove the cache or use matching settings."
            )


def _torch_float32_input(values):
    result = np.asarray(values, dtype=np.float32)
    return result.copy(order="C") if any(stride < 0 for stride in result.strides) else result


def _apply_spot_feature_mask(
    base_input: torch.Tensor,
    num_spot: int,
    mask_idx: torch.Tensor,
    mode: str = "zero",
    noise_std: float = 0.02,
) -> torch.Tensor:
    if mask_idx is None or mask_idx.numel() == 0:
        return base_input

    mode = str(mode).lower()
    masked = base_input.clone()
    spot_feat = masked[:num_spot]

    if mode == "zero":
        spot_feat[mask_idx] = 0.0
    elif mode == "noise":
        spot_feat[mask_idx] = torch.randn_like(spot_feat[mask_idx]) * float(noise_std)
    elif mode == "mean":
        mean_feat = spot_feat.mean(dim=0, keepdim=True)
        spot_feat[mask_idx] = mean_feat.expand(mask_idx.numel(), -1)
    else:
        raise ValueError(f"Unsupported feat_mask_mode={mode}. Use one of: zero/noise/mean.")

    masked[:num_spot] = spot_feat
    return masked


def _reserve_output_prefix(prefix, K, dim):
    """Keep an atomic reservation so concurrent runs never share artifacts."""
    candidate = prefix
    while True:
        stem = f"{candidate}.augK{K}_d{dim}"
        try:
            os.mkdir(stem + ".reserved")
        except FileExistsError:
            candidate = prefix + "_" + uuid.uuid4().hex
            continue
        if any(os.path.exists(stem + suffix) for suffix in (
                "_for_cluster.npz", "_none.h5ad", "_checkpoint.pt", "_training.json", ".SUCCESS")):
            candidate = prefix + "_" + uuid.uuid4().hex
            continue
        return candidate


def main(a):
    set_seed(a.seed, strict=not getattr(a, "no_strict_determinism", False))
    adata = ad.read_h5ad(a.h5)
    if not adata.obs_names.is_unique:
        raise ValueError("Input obs_names must be unique")
    _validate_args(a, adata.n_obs)
    if adata.X is not None:
        raw_values = adata.X.data if sp.issparse(adata.X) else np.asarray(adata.X)
        _require_finite(raw_values, "input expression")
    for key, matrix in adata.layers.items():
        values = matrix.data if sp.issparse(matrix) else np.asarray(matrix)
        _require_finite(values, f"input layer {key}")
    if adata.raw is not None:
        raw_values = adata.raw.X.data if sp.issparse(adata.raw.X) else np.asarray(adata.raw.X)
        _require_finite(raw_values, "raw expression")

    # ---------- output prefix (avoid overwrite) ----------
    if a.out_prefix and a.out_prefix.strip():
        prefix = a.out_prefix.strip()
    else:
        prefix = os.path.splitext(a.h5)[0]

    # ensure output dir exists (if prefix includes a directory)
    out_dir = os.path.dirname(prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    prefix = _reserve_output_prefix(prefix, a.K, a.dim)


    # ---------- spatial graph ----------
    cached_graphs = _cached_spatial_graphs(adata)
    adata = Cal_Spatial_Net(
        adata,
        rad_cutoff=a.radius,
        k_cutoff=a.k,
        model=a.graph_model,
        return_data=True,
        save_graph_csv=False,
    )
    requested_A = get_A_from_spatial_net(adata)
    if a.graph_model == "Radius":
        off_diagonal = requested_A.tocsr(copy=True)
        off_diagonal.setdiag(0)
        off_diagonal.eliminate_zeros()
        if off_diagonal.nnz == 0:
            raise ValueError("Radius graph has no off-diagonal spatial edges; check radius and coordinate units")
    _check_cached_graphs(cached_graphs, requested_A, a)

    # ---------- gene expression features (optional Scanpy workflow) ----------
    if a.use_scanpy_workflow:
        # Feature-only h5ad inputs are the immutable, prevalidated PCA cache.
        # Count-bearing inputs are always rebuilt under the requested protocol.
        if adata.n_vars == 0:
            if "X_pca" not in adata.obsm or adata.obsm["X_pca"].shape[1] < a.pca_comps:
                raise ValueError("Feature-only input must contain the requested X_pca cache")
            X = np.asarray(adata.obsm["X_pca"][:, :a.pca_comps], dtype=np.float32)
            input_source = "X_pca/cache"
            print(f"Use immutable X_pca feature cache with shape {X.shape}.")
        else:
            print("Running scanpy_workflow...")
            with threadpool_limits(limits=int(a.blas_threads)):
                adata_p = scanpy_workflow(adata.copy(), n_comps=a.pca_comps, call_hvg=True, viz=False)
            if "X_pca" not in adata_p.obsm:
                raise ValueError("X_pca not found after scanpy_workflow.")
            X = adata_p.obsm["X_pca"][:, :a.pca_comps]
            input_source = adata_p.uns["sasg_input_source"]
            adata.uns["sasg_input_source"] = input_source
            adata.obsm["X_pca"] = np.asarray(X, dtype=np.float32)
    else:
        print(f"Checking raw X data: {adata.X}")
        X = adata.X
        input_source = "X"

    print(f"X shape: {X.shape}")
    _require_finite(X, "PCA features")
    _validate_args(a, adata.n_obs, int(X.shape[1]))

    # ---------- features to reconstruct (for MAE) ----------
    X_recon = None
    if a.lambda_recon > 0:
        if str(getattr(a, "recon_target", "pca")).lower() != "pca":
            raise ValueError(f"Unsupported recon_target={a.recon_target}. Only 'pca' is supported.")
        if a.use_scanpy_workflow:
            X_recon = torch.tensor(
                _torch_float32_input(X),
                dtype=torch.float32,
                device=a.device,
            )
            _require_finite(X_recon, "reconstruction target")
            print(f"MAE: use PCA features dim={X_recon.shape[1]} as reconstruction target.")
        else:
            print("Warning: recon_target='pca' requires --use_scanpy_workflow. Disable MAE reconstruction.")
            a.lambda_recon = 0.0

    # ---------- adjacency matrix ----------
    A = requested_A

    A = A.tocsr().astype(np.float32)
    if sp.issparse(X):
        X_graph = X.tocsr().astype(np.float32)
    else:
        X_graph = sp.csr_matrix(np.asarray(X, dtype=np.float32))
    spatial_np = np.asarray(adata.obsm["spatial"], dtype=np.float32)

    # -------- coordinate-derived pseudo layers --------
    layer_ids_np = None
    pseudo_depth_cont = None

    if a.layer_aware and (getattr(a, "pseudo_layer_bins", 0) > 0):
        layer_ids_np, pseudo_depth_cont = compute_pseudo_layer_ids_from_spatial(
            spatial_np,
            n_bins=int(a.pseudo_layer_bins),
            knn_k=int(getattr(a, "pseudo_layer_knn", 18)),
            mode="fiedler",
            binning=str(getattr(a, "pseudo_layer_binning", "quantile")),
            obs_names=np.asarray(adata.obs_names, dtype="U"),
        )

    if pseudo_depth_cont is not None:
        spatial_for_layer = np.asarray(pseudo_depth_cont, dtype=np.float32).reshape(-1, 1)
        layer_axis_used = 0
    else:
        spatial_for_layer = spatial_np
        layer_axis_used = int(a.layer_axis)

    # ---------- augmented graph ----------
    Pe = build_augmented_graph(
        A, X_graph,
        alpha=a.alpha,
        normalize_A=True,
        normalize_X=True,
        topN=a.topN,
        spatial=spatial_for_layer,
        layer_aware=a.layer_aware,
        layer_gamma=a.layer_gamma,
        layer_axis=layer_axis_used,
        layer_scale_mode=a.layer_scale_mode,
        layer_cutoff=a.layer_cutoff,
    )


    P = scipy_to_torch_coo(Pe).to(a.device)
    # Preserve signed edges by normalising with absolute row mass.
    Ptilde = normalize_rows_absolute_torch(P)

    init_embed = None
    N_spot = A.shape[0]
    n = P.size(0)

    if a.use_scanpy_workflow:
        X_np = _torch_float32_input(X)  # PCA features when scanpy_workflow
        feat_dim = X_np.shape[1]

        init_embed = torch.randn(n, feat_dim, device=a.device) * 0.02
        init_embed[:N_spot, :] = torch.tensor(
            X_np, dtype=torch.float32, device=a.device
        )
        if n - N_spot != feat_dim:
            raise ValueError("The component-node count must equal the PCA dimension.")
        init_embed[N_spot:, :] = torch.diag(torch.tensor(
            np.asarray(X_np, dtype=np.float64).std(axis=0), dtype=torch.float32, device=a.device))

        if feat_dim == a.dim:
            print(
                f"Use X_pca to initialize first {N_spot} nodes; initialize {n - N_spot} component nodes on PCA axes."
            )
        else:
            print(
                f"Use X_pca(dim={feat_dim}) to initialize first {N_spot} nodes and PCA axes to initialize {n - N_spot} component nodes; "
                f"encoder will project {feat_dim}->{a.dim}."
            )
    else:
        print("Scanpy workflow is not used, node embeddings are randomly initialized.")

    # ---------- model and optimizer ----------
    activation_module = nn.PReLU()

    enc = CollaborativeEncoder(
        num_nodes=n,
        dim=a.dim,
        K=a.K,
        init_embed=init_embed,
        activation=activation_module,
    ).to(a.device)
    with torch.no_grad():
        for lin in [enc.lin0, *enc.lins]:
            if lin.weight.shape[0] != lin.weight.shape[1]:
                raise ValueError("Fixed-propagation initialisation requires square encoder layers.")
            lin.weight.copy_(torch.eye(lin.weight.shape[0], device=a.device))
            lin.bias.zero_()
        activation_module.weight.fill_(1.0)
    scorer = CrossLayerScorer(dim=a.dim, K=a.K, hidden=a.hidden, activation=nn.PReLU()).to(a.device)

    feat_decoder = None
    recon_crit = None
    if (X_recon is not None) and (a.lambda_recon > 0):
        if a.embed_agg == "concat":
            dec_in_dim = a.dim * (a.K + 1)
        else:
            dec_in_dim = a.dim
        feat_out_dim = X_recon.shape[1]
        feat_decoder = FeatureDecoder(
            in_dim=dec_in_dim,
            out_dim=feat_out_dim,
            hidden=256,
            activation=nn.PReLU(),
        ).to(a.device)
        recon_crit = torch.nn.MSELoss(reduction="mean")
        print(
            f"Enable MAE feature decoder: in_dim={dec_in_dim}, out_dim={feat_out_dim}, "
            f"lambda_recon={a.lambda_recon}, mask_ratio_feat={a.mask_ratio_feat}, "
            f"feat_mask_mode={a.feat_mask_mode}, feat_noise_std={a.feat_noise_std}, "
            f"recon_target={a.recon_target}."
        )

    params = unique_trainable_params(enc, scorer, feat_decoder)

    opt = torch.optim.Adam(
        params,
        lr=a.lr,
        weight_decay=a.weight_decay,
        betas=(0.9, 0.999),
    )
    bce = torch.nn.BCEWithLogitsLoss(reduction="mean")

    scheduler = None
    if a.scheduler:
        scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
        print("Using Cosine Annealing LR scheduler.")

    # -------- optional layer ids / normal-aware priors --------
    layer_ids = None
    if layer_ids_np is not None:
        layer_ids = torch.tensor(layer_ids_np, dtype=torch.long, device=a.device)

    normals = None
    normal_scales = None
    if getattr(a, "normal_aware", False):
        normals, normal_scales = compute_local_normals_from_spatial(
            spatial_np,
            knn_k=int(getattr(a, "normal_knn", 30)),
            obs_names=np.asarray(adata.obs_names, dtype="U"),
        )
        print(f"Normal-aware enabled. normal_knn={int(getattr(a, 'normal_knn', 30))}, "
              f"normal_margin={float(getattr(a, 'normal_margin', 1.0))}, "
              f"normal_gamma={float(getattr(a, 'normal_gamma', 2.0))}")

    if pseudo_depth_cont is not None:
        print(
            "Layer-aware negatives: "
            f"source=pseudo_fiedler, bins={int(a.pseudo_layer_bins)}, "
            f"knn={int(getattr(a, 'pseudo_layer_knn', 18))}, "
            f"binning={str(getattr(a, 'pseudo_layer_binning', 'quantile'))}, "
            f"gamma={float(a.layer_gamma):.3f}, "
            f"margin={int(a.neg_layer_margin) if hasattr(a, 'neg_layer_margin') else 2}, "
            f"neg_adj_ratio={float(getattr(a, 'neg_adj_ratio', 0.3)):.2f}, "
            f"hard_ratio={float(getattr(a, 'neg_hard_ratio', 0.7)):.2f}, "
            f"oversample={int(getattr(a, 'neg_oversample', 6))}, "
            "layer_axis=0"
        )

    # ---------- training loop ----------
    relations = RelationPairs(P, N_spot)
    ss = relations.ss
    ss_keys = torch.minimum(ss[:, 0], ss[:, 1]) * N_spot + torch.maximum(ss[:, 0], ss[:, 1])
    ss_neg_support = N_spot * (N_spot - 1) // 2 - int(torch.unique(ss_keys).numel())
    pairs_per_channel = min(a.pos_per_epoch // 2, relations.ss.size(0), relations.sf.size(0),
                            relations.non_association.size(0) // 2, ss_neg_support // 2)
    if pairs_per_channel < 1:
        raise ValueError("No feasible paired SS/SC positive and negative quota for this tissue")
    effective_pos_per_epoch = 2 * int(pairs_per_channel)
    if effective_pos_per_epoch != a.pos_per_epoch:
        print(f"Training relation quota: {effective_pos_per_epoch} positives per epoch "
              f"(requested {a.pos_per_epoch}; limited by legal pair support).")
    history = []
    for ep in tqdm(range(a.epochs), desc="Training Epochs", ncols=100):
        enc.train()
        scorer.train()

        # MAE-style masking on encoder input (spot nodes only), then encode once
        mask_idx = None
        base_input_override = None
        if (feat_decoder is not None) and (recon_crit is not None):
            num_mask = max(1, int(a.mask_ratio_feat * N_spot))
            mask_idx = torch.randperm(N_spot, device=a.device)[:num_mask]
            base_input = enc.get_base_input()
            base_input_override = _apply_spot_feature_mask(
                base_input=base_input,
                num_spot=N_spot,
                mask_idx=mask_idx,
                mode=str(getattr(a, "feat_mask_mode", "zero")),
                noise_std=float(getattr(a, "feat_noise_std", 0.02)),
            )

        step_operator = (masked_spot_component_operator(P, mask_idx, N_spot)
                         if mask_idx is not None else Ptilde)
        Hs = enc(step_operator, base_input_override=base_input_override)
        h_all = torch.cat(Hs, dim=1)  # used for negative mining scores

        use_layer_neg = bool(a.layer_aware) and (layer_ids is not None)

        oversample = max(int(a.neg_oversample), 128) if N_spot <= 512 else int(a.neg_oversample)
        while True:
            ss_pos, sf_pos = relations.positives(effective_pos_per_epoch)
            try:
                ni, nj = sample_negatives(
                    P, num_neg=2 * ss_pos.size(0), h=h_all,
                    layer_ids=(layer_ids if use_layer_neg else None),
                    layer_margin=(int(a.neg_layer_margin) if (use_layer_neg and hasattr(a, "neg_layer_margin")) else 2),
                    layer_gamma=(float(a.layer_gamma) if use_layer_neg else 0.0),
                    neg_adj_ratio=(float(getattr(a, "neg_adj_ratio", 0.3)) if use_layer_neg else 0.0),
                    hard_ratio=float(a.neg_hard_ratio) if hasattr(a, "neg_hard_ratio") else 0.7,
                    oversample_factor=oversample, spatial=spatial_np,
                    normals=normals, normal_scales=normal_scales,
                    normal_margin=float(getattr(a, "normal_margin", 1.0)),
                    normal_gamma=float(getattr(a, "normal_gamma", 2.0)),
                    sampling_seed=int(a.seed), sampling_epoch=int(ep),
                )
                ni, nj = ni.to(a.device), nj.to(a.device)
                ss_neg = torch.stack([ni, nj], 1)
                if ss_neg.size(0) != 2 * ss_pos.size(0):
                    raise RuntimeError("Negative sampler failed to emit the exact SS quota.")
                sf_neg = relations.feature_negatives(
                    2 * sf_pos.size(0), scorer, Hs,
                    a.neg_hard_ratio, oversample,
                    sampling_seed=int(a.seed), sampling_epoch=int(ep))
                break
            except InsufficientNegativeSupportError as exc:
                if effective_pos_per_epoch <= 2:
                    raise
                effective_pos_per_epoch = max(2, 2 * (effective_pos_per_epoch // 4))
                print(f"Reducing relation quota to {effective_pos_per_epoch}: {exc}")
        relation_losses = []
        relation_scores = []
        for positives, negatives in ((ss_pos, ss_neg), (sf_pos, sf_neg)):
            pos_logits = scorer.edge_score(Hs, positives[:, 0], positives[:, 1])
            neg_logits = scorer.edge_score(Hs, negatives[:, 0], negatives[:, 1])
            _require_finite(pos_logits, "positive relation logits")
            _require_finite(neg_logits, "negative relation logits")
            logits = torch.cat([pos_logits, neg_logits]).clamp(-30.0, 30.0)
            targets = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
            relation_losses.append(bce(logits, targets))
            relation_scores.append((pos_logits, neg_logits, logits))
        loss = 0.5 * (relation_losses[0] + relation_losses[1])
        s_pos = torch.cat([relation_scores[0][0], relation_scores[1][0]])
        s_neg = torch.cat([relation_scores[0][1], relation_scores[1][1]])
        s = torch.cat([relation_scores[0][2], relation_scores[1][2]])

        loss_recon = None
        if (feat_decoder is not None) and (recon_crit is not None) and (mask_idx is not None) and (mask_idx.numel() > 0):
            h_all = node_embedding_from_layers(Hs, mode=a.embed_agg)  # [N_all, dec_in_dim]
            h_spot = h_all[:N_spot]

            h_mask = h_spot[mask_idx]  # [num_mask, dec_in_dim]
            x_target = X_recon[mask_idx]  # [num_mask, feat_dim]

            x_pred = feat_decoder(h_mask)
            loss_recon = recon_crit(x_pred, x_target)

            loss = loss + a.lambda_recon * loss_recon

        _require_finite(loss, "training loss")

        # backward and update
        opt.zero_grad()
        loss.backward()
        for parameter in params:
            if parameter.grad is not None:
                _require_finite(parameter.grad, "training gradient")
        grad_norms = [torch.linalg.vector_norm(parameter.grad.detach())
                      for parameter in params if parameter.grad is not None]
        if grad_norms:
            grad_norm = torch.linalg.vector_norm(torch.stack(grad_norms))
            _require_finite(grad_norm, "gradient norm")
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0, error_if_nonfinite=True)
        opt.step()

        if a.scheduler:
            scheduler.step()

        history.append({
            "epoch": ep + 1,
            "relation_positives": effective_pos_per_epoch,
            "loss": float(loss.detach().item()),
            "loss_edge": float((0.5 * (relation_losses[0] + relation_losses[1])).detach().item()),
            "loss_recon": None if loss_recon is None else float(loss_recon.detach().item()),
            "lr": float(opt.param_groups[0]["lr"]),
        })

        if (ep + 1) % max(1, a.epochs // 10) == 0:
            with torch.inference_mode():
                logit_abs = s.abs().mean().item()
                p_pos = s_pos.sigmoid().mean().item()
                p_neg = s_neg.sigmoid().mean().item()
                if loss_recon is not None:
                    recon_val = loss_recon.item()
                else:
                    recon_val = None
            msg = (
                f"[epoch {ep + 1}/{a.epochs}] loss={loss.item():.4f} "
                f"| mean|logit|={logit_abs:.2f} | p_pos={p_pos:.3f} p_neg={p_neg:.3f} "
                f"| n_pos={int(s_pos.numel())} n_neg={int(s_neg.numel())}"
            )
            if recon_val is not None:
                msg += f" | loss_recon={recon_val:.4f}"
            print(msg)

    # Recompute BatchNorm statistics on the complete, unmasked graph so the
    # training masks do not leak into the exported embedding.
    for module in enc.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.reset_running_stats()
            module.momentum = None
    enc.train()
    for module in enc.modules():
        if isinstance(module, nn.Dropout):
            module.eval()
    with torch.no_grad():
        enc(Ptilde)

    # ---------- extract embedding ----------
    enc.eval()
    with torch.inference_mode():
        Hs = enc(Ptilde)
        Z_all = node_embedding_from_layers(Hs, mode=a.embed_agg).cpu().numpy()
    _require_finite(Z_all, "exported embedding")

    N_spot = A.shape[0]
    Z_spot = np.asarray(Z_all[:N_spot], dtype=np.float32)

    adata.obsp["A_spatial"] = A
    rep_key = a.use_rep.strip()

    if not rep_key:
        for k in ["X_dom", "X_emb", "X_pca", "X_umap", "X"]:
            if k in adata.obsm:
                rep_key = k
                break

    if Z_spot.ndim != 2:
        raise ValueError(
            f"Z_spot dimension error: expected 2 dimensions, but got {Z_spot.ndim} dimensions"
        )

    if not rep_key:
        rep_key = "X_emb"

    adata.obsm[rep_key] = Z_spot
    adata.obsm["emb"] = Z_spot
    print(f"Saved embedding to adata.obsm['{rep_key}'] and adata.obsm['emb'].")

    checkpoint_path = f"{prefix}.augK{a.K}_d{a.dim}_checkpoint.pt"
    for module in (enc, scorer, feat_decoder):
        if module is not None:
            for name, value in module.state_dict().items():
                _require_finite(value, f"checkpoint tensor {name}")
    def check_state(value, name):
        if torch.is_tensor(value):
            _require_finite(value, name)
        elif isinstance(value, dict):
            for key, item in value.items():
                check_state(item, f"{name}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                check_state(item, f"{name}[{index}]")
    check_state(opt.state_dict(), "optimizer state")
    if scheduler is not None:
        check_state(scheduler.state_dict(), "scheduler state")
    torch.save({
        "format": "SASG-ST checkpoint v1",
        "encoder": enc.state_dict(),
        "scorer": scorer.state_dict(),
        "feature_decoder": None if feat_decoder is None else feat_decoder.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "arguments": vars(a),
        "history": history,
    }, checkpoint_path)
    with open(checkpoint_path, "rb") as handle:
        checkpoint_sha256 = hashlib.sha256(handle.read()).hexdigest()
    metadata_path = f"{prefix}.augK{a.K}_d{a.dim}_training.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump({"seed": int(a.seed), "strict_determinism": not a.no_strict_determinism,
                   "input_source": input_source,
                   "arguments": vars(a), "history": history, "checkpoint": checkpoint_path,
                   "checkpoint_sha256": checkpoint_sha256}, handle, indent=2)
    print(f"Training checkpoint saved to: {checkpoint_path}")

    # Write the clustering artifact only after all model states are validated.
    extra_path = f"{prefix}.augK{a.K}_d{a.dim}_for_cluster.npz"
    extra_tmp = extra_path + ".tmp.npz"
    A_coo = A.tocoo()
    np.savez_compressed(
        extra_tmp,
        embedding=Z_spot.astype(np.float32),
        obs_names=np.asarray(adata.obs_names, dtype="U"),
        spatial=spatial_np,
        A_row=A_coo.row.astype(np.int64),
        A_col=A_coo.col.astype(np.int64),
        A_weight=A_coo.data.astype(np.float32),
        rep_key=np.array([rep_key]),
        dim=np.array([a.dim], dtype=np.int64),
        K=np.array([a.K], dtype=np.int64),
    )
    os.replace(extra_tmp, extra_path)
    print(f"Extra clustering file saved to: {extra_path}")

    out_h5 = f"{prefix}.augK{a.K}_d{a.dim}_none.h5ad"
    print(f"[output] prefix = {prefix}")

    adata.write_h5ad(out_h5)
    success_path = f"{prefix}.augK{a.K}_d{a.dim}.SUCCESS"
    artifacts = {}
    for path in (extra_path, checkpoint_path, metadata_path, out_h5):
        digest = hashlib.sha256()
        with open(path, "rb") as artifact:
            for block in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(block)
        artifacts[os.path.basename(path)] = digest.hexdigest()
    with open(success_path + ".tmp", "w", encoding="utf-8") as handle:
        json.dump({"prefix": prefix, "artifacts_sha256": artifacts}, handle, indent=2)
    os.replace(success_path + ".tmp", success_path)
    print(f"Main output saved to: {out_h5}. (Only embeddings, no clustering labels)")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train SASG-ST with fixed common settings.")
    p.add_argument("--h5", required=True)
    p.add_argument("--graph_model", choices=["Radius", "KNN"], default="KNN")
    p.add_argument("--radius", type=float, default=80.0)
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--dim", type=int, default=64,
                   help="Encoder/scorer representation width; default preserves the historical 64-wide model.")
    p.add_argument("--hidden", type=int, default=512,
                   help="Cross-layer scorer hidden width; default preserves the historical 512-wide model.")
    p.add_argument("--epochs", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no_strict_determinism",
        action="store_true",
        help="Disable strict deterministic algorithms for faster training.",
    )
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--blas_threads", type=int, default=1,
                   help="BLAS threads used for preprocessing PCA; fixed for reproducibility.")
    p.add_argument("--lambda_recon", type=float, default=1.0,
                   help="Weight of masked PCA reconstruction loss.")
    p.add_argument("--mask_ratio_feat", type=float, default=0.2,
                   help="Fraction of spots masked for reconstruction (0, 1).")
    p.add_argument("--out_prefix", type=str, default="", help="Output prefix for result files")
    p.add_argument("--layer_aware", dest="layer_aware", action="store_true")
    p.add_argument("--no_layer_aware", dest="layer_aware", action="store_false")
    p.add_argument("--layer_gamma", type=float, default=2.0)
    p.add_argument("--neg_layer_margin", type=int, default=2,
                   help="Minimum pseudo-layer distance for negatives; 0 disables the margin.")
    p.add_argument("--neg_hard_ratio", type=float, default=0.6,
                   help="Fraction of negatives selected by embedding-similarity ranking.")
    p.add_argument("--pseudo_layer_bins", type=int, default=7,
                   help="Fiedler bins; 0 disables spectral bins but layer-aware graph weighting still uses the coordinate axis. Use --no_layer_aware to disable both.")
    p.add_argument("--normal_aware", dest="normal_aware", action="store_true",
                   help="Use coordinate-derived local-normal constraints for negatives.")
    p.add_argument("--no_normal_aware", dest="normal_aware", action="store_false")
    p.add_argument("--normal_margin", type=float, default=1.0, help="Min normalized normal-distance for negatives.")

    p.set_defaults(
        topN=50,
        use_scanpy_workflow=True,
        pca_comps=64,
        pos_per_epoch=20000,
        embed_agg="concat",
        use_rep="emb",
        feat_mask_mode="zero",
        feat_noise_std=0.02,
        recon_target="pca",
        activation="prelu",
        scheduler=True,
        weight_decay=1e-4,
        layer_axis=1,
        layer_scale_mode="edge_median",
        layer_cutoff=None,
        neg_oversample=8,
        neg_adj_ratio=0.3,
        pseudo_layer_knn=20,
        pseudo_layer_binning="quantile",
        normal_knn=15,
        normal_gamma=2.0,
        layer_aware=True,
        normal_aware=True,
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
