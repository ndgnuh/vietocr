#!/bin/env python3
import sys
from argparse import ArgumentParser

try:
    import icecream
    icecream.install()
except ImportError:
    pass

actions = {}


def add_action(f):
    if isinstance(f, str):

        def wrap(g):
            actions[f] = g
            return g

        return wrap
    actions[f.__name__] = f
    return f


@add_action("list")
def show(args):
    pass


@add_action
def proto(args):
    import yaml

    with open("./conf.yaml") as f:
        print(yaml.load(f, Loader=yaml.FullLoader))


@add_action
def mkconfig(args):
    parser = ArgumentParser()
    help = "Output config file"
    parser.add_argument("--output", "-o", required=True, help=help)
    args = parser.parse_args(args)

    # Import packages
    import yaml

    from vietocr.configs import OcrConfig

    config = OcrConfig(
        vocab="vietnamese",
        type="ctc",
        backbone="fvtr",
        head="fc",
    ).to_dict()
    with open(args.output, "w") as f:
        yaml.dump(config, f, sort_keys=False)


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

    from vietocr.configs import OcrConfig

    config = OcrConfig.from_yaml(args.config)
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
    """Train a model."""
    parser = ArgumentParser()
    help = "path to model configuration file"
    parser.add_argument("--config", "-c", required=True, help=help)
    help = "Number of sub-processes to use (torch's num_workers)"
    parser.add_argument("--num-workers", "-p", type=int, default=0, help=help)
    args = parser.parse_args(args)

    from vietocr.configs import OcrConfig
    from vietocr.trainer import OcrTrainer

    # Load and modify configs if needed
    config = OcrConfig.from_yaml(args.config)
    config.num_workers = args.num_workers

    # Training
    trainer = OcrTrainer(config)
    trainer.fit()


@add_action
def test(args):
    """Test a model for accuracy."""
    parser = ArgumentParser()
    help = "path to model configuration file"
    parser.add_argument("--config", "-c", required=True, help=help)
    help = "if output is set, the metric statistics are written to this file"
    parser.add_argument("--output", "-o", required=False, help=help)
    args = parser.parse_args(args)

    # Load config
    from vietocr.configs import OcrConfig

    config = OcrConfig.from_yaml(args.config)

    # Load test
    from vietocr.trainer import run_test

    run_test(config, output=args.output)


def main():
    """Entrypoint."""
    parser = ArgumentParser(description="Run `--help` with each intent for details.")
    parser.add_argument("intent", choices=list(actions))

    # No-args
    if len(sys.argv) < 2:
        parser.parse_args()

    # No-intent
    intent = sys.argv[1]
    args = sys.argv[2:]
    if intent not in actions:
        parser.parse_args()

    # Has intent
    actions[intent](args)


if __name__ == "__main__":
    main()
