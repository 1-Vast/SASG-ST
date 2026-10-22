"""Command-line entry point for SASG-ST training."""
from model.train import main, parse_args


if __name__ == "__main__":
    main(parse_args())
