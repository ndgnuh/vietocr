import argparse
import yaml
import json
from icecream import install

from pprint import pprint
from vietocr.tool.config import get_config

install()


def read_json(fpath: str):
    with open(fpath, encoding='utf-8') as f:
        return json.load(f)


def read_yaml(fpath: str):
    with open(fpath, encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def train(config, args):
    # Keep the script fast for other purpose
    from vietocr.model.trainer import Trainer
    if args.name is not None:
        config['name'] = args.name
    trainer = Trainer(config)
    trainer.train()


def run_test(config):
    from vietocr.tool.translate import build_model
    model, vocab = build_model(config)


def export(config, output):
    import torch
    from vietocr.tool.predictor import Predictor

    # Get sample data
    sample_input = torch.rand(2, 3, config['image_height'], 100)

    # Load the model
    model = Predictor(config).model

    # Export
    print(sample_input.shape)
    torch.onnx.export(
        model.eval(),
        sample_input,
        output,
        opset_version=15,
        do_constant_folding=True,
        input_names=["images"],
        dynamic_axes=dict(
            images=[0, 3]  # Dynamic width and height
        )
    )


def main():
    parser = argparse.ArgumentParser()
    # keeping the option as --config
    parser.add_argument(
        '-c', '--config',
        dest="config",
        required=True,
        help="config files, latter ones will merge/override the formers"
    )
    sp = parser.add_subparsers(dest='action', required=True)

    # Train
    train_parser = sp.add_parser('train')
    train_parser.add_argument(
        '-e',
        '--experiment',
        dest="training",
        required=True)
    train_parser.add_argument(
        '--name',
        dest="name",
        help="Custom experiment name",
        required=False)

    # Testing
    test_parser = sp.add_parser("test")
    test_parser.add_argument(
        "--test-annotation",
        "-i",
        dest="test_annotation",
        required=True
    )
    test_parser.add_argument(
        "--batch_size",
        "-b",
        dest="batch_size",
        default=4,
        type=int,
    )

    # Export
    export_parser = sp.add_parser('export', help="Export to ONNX")
    export_parser.add_argument('output', help="Output file path")

    # Dump config file
    sp.add_parser('dump')
    args = parser.parse_args()
    print(args)
    config = get_config(args.config)

    # Dispatch
    action = args.action
    if action == "train":
        config['training'] = read_yaml(args.training)
        train(config, args)
    elif action == "test":
        from vietocr.scripts import test
        test.main(config, args)
    elif action == "export":
        export(config, args.output)
    elif action == "dump":
        pprint(config)


if __name__ == '__main__':
    main()
