import argparse
import yaml
import json

from os import path
from pprint import pprint
from vietocr.model.trainer import Trainer
from vietocr.tool.config import Cfg
from vietocr.tool.dict_utils import munchify, merge_dict


def read_json(fpath: str):
    with open(fpath, encoding='utf-8') as f:
        return json.load(f)


def read_yaml(fpath: str):
    with open(fpath, encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def main():
    parser = argparse.ArgumentParser()
    # keeping the option as --config
    parser.add_argument('-c', '--config',
                        action="append",
                        dest="configs",
                        required=True,
                        default=[path.join("config", "base.yml")],
                        help="config files, latter ones will merge/override the formers")
    parser.add_argument('--checkpoint',
                        dest="checkpoint",
                        required=False,
                        help='your checkpoint')

    args = parser.parse_args()
    configs = [read_yaml(config_file) for config_file in args.configs]
    config = merge_dict(*configs)
    pprint(config)
    trainer = Trainer(config)

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    trainer.train()


if __name__ == '__main__':
    main()
