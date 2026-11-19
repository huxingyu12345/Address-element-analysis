# -*- coding: utf-8 -*-
"""CoNLL 读写与测试集格式转换。"""
import sys


def read_conll_sents(file_path):
    """读取 CoNLL 文件，返回 [(char, tag), ...] 列表。"""
    sents = []
    c_tags = []
    for line in open(file_path, encoding='utf8'):
        line = line.rstrip()
        if len(line) == 0:
            if c_tags:
                sents.append(c_tags)
            c_tags = []
            continue
        try:
            c, t = line.split(' ')
            c_tags.append((c, t))
        except ValueError:
            print(f"Skipping invalid line: {line}", file=sys.stderr)
    if c_tags:
        sents.append(c_tags)
    return sents


def process_test_file(input_file, output_file):
    """将 GUID\\u0001文本 的测试文件转为 CoNLL。"""
    with open(input_file, 'r', encoding='utf8') as fin, \
         open(output_file, 'w', encoding='utf8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\u0001')
            if len(parts) >= 2:
                _guid, text = parts[0], parts[1]
                for token in text:
                    print(token, 'O', sep=' ', file=fout)
                print('', file=fout)


def write_predictions(output_file, features, labels):
    """写出 ID\\u0001原文\\u0001标签序列。"""
    with open(output_file, 'w', encoding='utf-8') as fout:
        for i, seq_labels in enumerate(labels):
            text = ''.join([c['w'] for c in features[i]])
            print(i + 1, text, ' '.join(seq_labels), sep='\u0001', file=fout)


def write_feature_importance(output_file, top_features):
    with open(output_file, 'w', encoding='utf-8') as f:
        for label, features in top_features.items():
            f.write(f"\n标签 {label} 的重要特征:\n")
            for feature, weight in features:
                f.write(f"  {feature}: {weight}\n")
