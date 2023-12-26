import sys
from argparse import ArgumentParser, Namespace

actions = {}


def add_action(f):
    actions[f.__name__] = f
    return f


@add_action
def pretrain(args):
    parser = ArgumentParser()
    help = "path to model configuration file"
    parser.add_argument("--config", "-c", required=True, help=help)
    help = "character repetition when rendering training data"
    parser.add_argument("--rep", "-r", type=int, default=32, help=help)
    help = "learning rate"
    parser.add_argument("--lr", type=float, default=7e-4, help=help)
    help = "directory of fonts to be used when rendering training data"
    parser.add_argument("--fonts", help=help)
    args = parser.parse_args(args)
    raise RuntimeError("Not implemented")


@add_action
def download_fonts(args):
    parser = ArgumentParser()
    help = "output directory that contains font files"
    parser.add_argument("--output-folder", default="fonts", help=help)
    args = parser.parse_args(args)
    raise RuntimeError("Not implemented")


@add_action
def train(args):
    parser = ArgumentParser()
    help = "path to model configuration file"
    parser.add_argument("--config", "-c", required=True, help=help)
    help = "Number of sub-processes to use (torch's num_workers)"
    parser.add_argument("--num-workers", "-p", type=int, default=0, help=help)
    args = parser.parse_args(args)
    raise RuntimeError("Not implemented")


def main():
    parser = ArgumentParser(
        description="Run `--help` with each intent for details.",
    )
    parser.add_argument("intent", choices=list(actions))

    # No-intent
    intent = sys.argv[1]
    args = sys.argv[2:]
    if intent not in actions:
        parser.parse_args()

    # Has intent
    actions[intent](args)


if __name__ == "__main__":
    main()
