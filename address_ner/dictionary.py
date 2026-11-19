# -*- coding: utf-8 -*-
"""从标注数据构建、加载地址分词词典。"""
import os
import sys

import jieba

from .config import DICT_FREQ_SCALE, DICT_MIN_FREQ


def load_address_dict(dict_file):
    """把用户词典加载进 jieba。"""
    if os.path.exists(dict_file):
        jieba.load_userdict(dict_file)
    else:
        print(
            f"Dictionary file {dict_file} not found, using default jieba dictionary",
            file=sys.stderr,
        )


def _label_type(label):
    return label.split('-')[-1] if '-' in (label or '') else 'O'


def _same_entity(current_label, label):
    current_type = _label_type(current_label) if current_label is not None else ''
    return current_type == _label_type(label)


def _flush_element(address_elements, current_element, current_label):
    if current_element and current_label is not None and current_label != 'O':
        element_text = ''.join(current_element)
        address_elements[element_text] = address_elements.get(element_text, 0) + 1


def build_address_dict(train_file, output_dict_file, min_freq=DICT_MIN_FREQ):
    """从训练数据中提取地址要素，生成分词词典。"""
    address_elements = {}
    current_element = []
    current_label = None

    with open(train_file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.rstrip()
            if len(line) == 0:
                _flush_element(address_elements, current_element, current_label)
                current_element = []
                current_label = None
                continue
            try:
                char, label = line.split(' ')
                if label == 'O':
                    _flush_element(address_elements, current_element, current_label)
                    current_element = []
                    current_label = None
                elif label.startswith('B-'):
                    _flush_element(address_elements, current_element, current_label)
                    current_element = [char]
                    current_label = label
                elif label.startswith('I-') or label.startswith('E-'):
                    if current_label is not None and _same_entity(current_label, label):
                        current_element.append(char)
                    else:
                        _flush_element(address_elements, current_element, current_label)
                        current_element = [char]
                        current_label = label
            except ValueError:
                print(f"Skipping invalid line: {line}", file=sys.stderr)

    _flush_element(address_elements, current_element, current_label)

    with open(output_dict_file, 'w', encoding='utf-8') as f:
        for word, freq in address_elements.items():
            if freq >= min_freq:
                print(f"{word} {freq * DICT_FREQ_SCALE}", file=f)

    return address_elements
