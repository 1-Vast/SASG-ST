"""Forward the command line to the supported training entry point."""
from model.main import main, parse_args


if __name__ == "__main__":
    main(parse_args())
