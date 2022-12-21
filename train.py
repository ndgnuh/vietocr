import argparse
import yaml
import json
from icecream import install

from os import path
from pprint import pprint
from vietocr.tool.dict_utils import merge_dict

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

    trainer = Trainer(config)

    # Weight
    if args.continue_weight is not None:
        trainer.load_weights(args.continue_weight)

    if args.checkpoint is not None:
        trainer.load_checkpoint(checkpoint)

    trainer.train()


def export(config, weight_path, output):
    import torch
    from PIL import Image
    from vietocr.tool.translate import process_input
    from vietocr.tool.predictor_wrapper import Predictor

    # Get sample data
    data = config['dataset']
    sample_input = Image.open("image/sample.png")
    sample_input = process_input(
        sample_input,
        image_height=data['image_height'],
        image_min_width=data['image_min_width'],
        image_max_width=data['image_max_width'],
    )

    # Load the model
    config['device'] = 'cpu'
    config['weights'] = weight_path
    config['predictor'] = dict(beamsearch=False)
    model = Predictor(config)

    # Export
    print(sample_input.shape)
    torch.onnx.export(
        model.eval(),
        sample_input,
        output,
        opset_version=12,
        do_constant_folding=True,
        input_names=["images"],
        dynamic_axes=dict(
            images=[3]
        )
    )


def main():
    parser = argparse.ArgumentParser()
    # keeping the option as --config
    parser.add_argument(
        '-c', '--config',
        action="append",
        dest="configs",
        default=[path.join("config", "base.yml")],
        help="config files, latter ones will merge/override the formers"
    )
    sp = parser.add_subparsers(dest='action', required=True)

    # Train
    train_parser = sp.add_parser('train')
    train_parser.add_argument(
        '--continue',
        dest="continue_weight",
        required=False,
        help='Continue training from this weight')
    train_parser.add_argument(
        '--checkpoint',
        dest="checkpoint",
        required=False,
        help='your checkpoint')

    # Export
    export_parser = sp.add_parser('export')
    export_parser.add_argument('-w', '--weight',
                               dest="weight",
                               required=True,
                               help='model weight path')

    # Dump config file
    sp.add_parser('dump')
    args = parser.parse_args()
    action = args.action
    configs = [read_yaml(config_file) for config_file in args.configs]
    config = merge_dict(*configs)

    config['experiment_name'] = path.splitext(
        path.basename(args.configs[-1])
    )[0]
    if action == "train":
        train(config, args)
    elif action == "export":
        export(config, args.weight, "test.onnx")
    elif action == "dump":
        pprint(config)


if __name__ == '__main__':
    main()
