#!/bin/env python3
import argparse
import json
from argparse import ArgumentParser
from pprint import pprint

actions = {}


def read_json(fpath: str):
    with open(fpath, encoding="utf-8") as f:
        return json.load(f)


def read_yaml(fpath: str):
    import yaml

    with open(fpath, encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def add_action(fn):
    actions[fn.__name__] = fn
    return fn


@add_action
def test(args):
    # Parse options
    parser = ArgumentParser()
    parser.add_argument("--config", "-c", help="Model config", required=True)
    parser.add_argument("--data", "-d", help="Test data path", required=True)
    parser.add_argument(
        "--batch-size", "-b", default=1, type=int, help="Test loader batch size"
    )
    args = parser.parse_args(args)

    # Import test module
    from vietocr.scripts.test import main as run_test
    from vietocr.tool.config import get_config

    # Run tests
    config = get_config(args.config)
    run_test(config, args.batch_size, args.data)


@add_action
def export(args):
    # Parse options
    parser = ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args(args)

    # Import
    import torch
    from torch import nn

    from vietocr.model.transformerocr import OnnxWrapper
    from vietocr.tool.config import get_config
    from vietocr.tool.translate import build_model

    # Action
    config = get_config(args.config)
    model, _ = build_model(config, move_to_device=False)
    model = OnnxWrapper(model, config["image_height"])
    model.export(args.output)


@add_action
def train(args):
    # Check args
    parser = ArgumentParser()
    parser.add_argument("--config", "-c", help="model config", required=True)
    parser.add_argument("--experiment", "-e", help="training config", required=True)
    parser.add_argument("--name", help="Custom experiment name")
    args = parser.parse_args(args)

    # Import deps
    from vietocr.model.trainer import Trainer

    # Training
    if args.name is not None:
        config["name"] = args.name
    trainer = Trainer(config)
    trainer.train()


def main():
    parser = ArgumentParser()
    parser.add_argument("action", choices=list(actions))
    args, other_args = parser.parse_known_args()
    dispatch = actions[args.action]
    dispatch(other_args)


if __name__ == "__main__":
    main()
