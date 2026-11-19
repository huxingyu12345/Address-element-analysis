# -*- coding: utf-8 -*-
"""统一入口。

用法:
    python main.py
    python main.py all
    python main.py train
    python main.py predict
    python main.py dict
"""
import argparse

from address_ner.pipeline import main as run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description='中文地址要素解析：词典构建 / CRF 训练 / 预测',
    )
    parser.add_argument(
        'mode',
        nargs='?',
        default='all',
        choices=['all', 'train', 'predict', 'dict'],
        help='all=训练+预测（默认），train=只训练，predict=只预测，dict=只构建词典',
    )
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='关闭 BERT 句向量特征，仅使用字符/词级特征',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='dict 模式下强制重建地址词典',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    use_llm = False if args.no_llm else None
    run_pipeline(mode=args.mode, use_llm=use_llm, force_dict=args.force)


if __name__ == '__main__':
    main()
