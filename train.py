import argparse
import yaml
import json

from vietocr.model.trainer import Trainer
from vietocr.tool.config import Cfg


def read_json(fpath: str):
    with open(fpath, encoding='utf-8') as f:
        return json.load(f)


def read_yaml(fpath: str):
    with open(fpath, encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='see example at ')
    parser.add_argument('--checkpoint', required=False, help='your checkpoint')
    parser.add_argument('--data-config', required=False,
                        help='override dataset configuration')

    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)

    if args.data_config is not None:
        data_config = read_yaml(args.data_config)
        print(data_config)
        config['dataset'] = data_config['dataset']

    trainer = Trainer(config)

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    trainer.train()


if __name__ == '__main__':
    main()
