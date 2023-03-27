import yaml
from os import path
from vietocr.tool.utils import download_config

config_dir = path.dirname(__file__)
config_dir = path.join(config_dir, "..", "data", "config")
config_dir = path.normpath(config_dir)


url_config = {
    'fvtr_t': 'fvtr_t.yml',
    'fvtr_v2_t@vn': 'fvtr_v2_t@vn.yml',
    'fvtr_v2_t@kr': 'fvtr_v2_t@kr.yml',
    'inception_v3_seq2seq': 'inception_v3_s2s.yml',
    'inception_v3_seq2seq_large': 'inception_v3_s2s_large.yml',
    'mobilenet_v3l_seq2seq': 'mv3l_s2s.yml',
    'resnet50_seq2seq': 'resnet50_s2s.yml',
    'vgg_transformer': 'vgg-transformer.yml',
    'vgg_seq2seq': 'vgg-seq2seq.yml',
}


def list_configs():
    return list(url_config.keys())


def read_yaml(fpath: str):
    with open(fpath, encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_config(name_or_path: str):
    if name_or_path in url_config:
        config_path = url_config[name_or_path]
        config_path = path.join(config_dir, config_path)
    else:
        config_path = name_or_path

    config_path = path.normpath(config_path)
    config = read_yaml(config_path)
    config['name'] = path.splitext(path.basename(config_path))[0]
    return config


class Cfg(dict):
    def __init__(self, config_dict):
        super(Cfg, self).__init__(**config_dict)
        self.__dict__ = self

    @staticmethod
    def load_config_from_file(fname):
        #base_config = download_config(url_config['base'])
        base_fname = path.join(path.dirname(fname), url_config['base'])
        with open(base_fname, encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
        with open(fname, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        base_config.update(config)

        return Cfg(base_config)

    @staticmethod
    def load_config_from_name(name):
        base_config = download_config(url_config['base'])
        config = download_config(url_config[name])

        base_config.update(config)
        return Cfg(base_config)

    def save(self, fname):
        with open(fname, 'w') as outfile:
            yaml.dump(dict(self), outfile,
                      default_flow_style=False, allow_unicode=True)
