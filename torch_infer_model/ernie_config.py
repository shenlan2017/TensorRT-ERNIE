# -*- coding: utf-8 -*

import logging
import json
import paddle
import torch
import numpy as np
import collections
import os
import pdb
import argparse

from ernie_model import Model

class ErnieConfig(object):
    """parse ernie config"""

    def __init__(self):
        self._config_dict = self._parse()

    def _parse(self):
        config_dict = {
            'hidden_size' : 768,
            'num_hidden_layers' : 12,
            'hidden_dropout_prob' : 0.9,
            'vocab_size' : 50000,
            'max_position_embeddings' : 513,
            'sent_type_vocab_size' : 4,
            'num_attention_heads' : 12,
            'query_hidden_size_per_head' : 64,
            'attention_probs_dropout_prob' : 0.9,
            'intermediate_size' : 3072,
            'hidden_act' : 'relu',
            'aside_emb_0_i' : 1432,
            'aside_emb_0_o' : 20,
            'aside_emb_124567_i' : 11,
            'aside_emb_1234567_o' : 20,
            'aside_emb_3_i' : 13,
            'aside_cls_out' : 384,
            'aside_embed_fc' : 160,
        }
        for i in range(1, 8, 1):
            config_dict[f'aside_emb_{i}_i'] = config_dict['aside_emb_124567_i']
            config_dict[f'aside_emb_{i}_o'] = config_dict['aside_emb_1234567_o']
        config_dict['aside_emb_3_i'] = 13
        return config_dict

    def __getitem__(self, key):
        """
        :param key:
        :return:
        """
        return self._config_dict.get(key, None)

    def __setitem__(self, key, value):
        """
        :param key, value:
        """
        self._config_dict[key] = value

    def has(self, key):
        """
        :param key:
        :return:
        """
        if key in self._config_dict:
            return True
        return False

    def get(self, key, default_value):
        """
        :param key,default_value:
        :retrun:
        """
        if key in self._config_dict:
            return self._config_dict[key]
        else:
            return default_value

    def print_config(self):
        """
        :return:
        """
        for arg, value in self._config_dict:
            logging.info('%s: %s' % (arg, value))
        logging.info('------------------------------------------------')

