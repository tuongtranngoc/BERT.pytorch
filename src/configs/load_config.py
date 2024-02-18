from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import yaml
import os

__cfg_types__ = {
    'pretrain_wikitext': 'src/configs/pretrain/wikitext.yaml',
    'fintune_snli': 'src/configs/finetune/snli.yaml'
}


def configuration(cfg_type):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: config
    """
    file_path = __cfg_types__[cfg_type]
    ext = os.path.splitext(file_path)[1]
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    with open(file_path, 'rb') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    return config