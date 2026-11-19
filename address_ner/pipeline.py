# -*- coding: utf-8 -*-
"""训练与预测主流程。"""
import os

import torch

from .config import (
    CONLL_TEST_FILE,
    DEV_CONLL_FILE,
    DEV_FILE,
    DEV_PRED_FILE,
    DICT_FILE,
    EXP_DIR,
    FEATURE_TEMPLATE,
    FINAL_TEST_FILE,
    MODEL_PATH,
    OUTPUT_DIR,
    PRED_FILE,
    TRAIN_FILE,
    USE_LLM,
)
from .dataset import process_test_file, write_feature_importance, write_predictions
from .dictionary import build_address_dict
from .model import EnhancedCRFTrainer
from .processor import EnhancedDataProcessor


def resolve_paths():
    if os.path.exists(FINAL_TEST_FILE):
        return {
            'test_file': FINAL_TEST_FILE,
            'conll_test_file': CONLL_TEST_FILE,
            'output_file': PRED_FILE,
            'using_dev_as_test': False,
        }
    return {
        'test_file': DEV_FILE,
        'conll_test_file': DEV_CONLL_FILE,
        'output_file': DEV_PRED_FILE,
        'using_dev_as_test': True,
    }


def prepare_resources(paths):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(EXP_DIR, exist_ok=True)

    if not os.path.exists(DICT_FILE):
        print("正在构建地址词典...")
        build_address_dict(TRAIN_FILE, DICT_FILE)
    else:
        print(f"使用现有词典: {DICT_FILE}")

    if (not paths['using_dev_as_test']) or (not os.path.exists(paths['conll_test_file'])):
        print("正在处理测试文件...")
        process_test_file(paths['test_file'], paths['conll_test_file'])


def create_processor(use_llm=None):
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("未检测到GPU，将使用CPU")

    if use_llm is None:
        use_llm = USE_LLM

    print("初始化增强数据处理器...")
    return EnhancedDataProcessor(
        feature_template=FEATURE_TEMPLATE,
        dict_file=DICT_FILE,
        use_llm=use_llm,
        use_gpu=use_gpu,
    )


def train_and_save(processor):
    print("加载训练数据...")
    train_features, train_labels = processor.load_conll(TRAIN_FILE)
    print(f'训练样本数: {len(train_features)}')

    print("开始训练增强CRF模型...")
    trainer = EnhancedCRFTrainer(MODEL_PATH)
    trainer.train(train_features, train_labels)
    trainer.save_model()
    print(f"模型已保存至: {MODEL_PATH}.pkl")

    print("分析模型特征重要性...")
    top_features = trainer.get_top_features()
    if top_features:
        feature_importance_file = os.path.join(EXP_DIR, 'feature_importance.txt')
        write_feature_importance(feature_importance_file, top_features)
        print(f"特征重要性分析已保存至: {feature_importance_file}")
    return trainer


def predict_and_save(processor, conll_test_file, output_file):
    print("加载测试数据并进行预测...")
    test_features, _ = processor.load_conll(conll_test_file)

    new_model = EnhancedCRFTrainer(MODEL_PATH)
    new_model.load_model()
    labels = new_model.predict(test_features)

    print(f"正在保存预测结果至: {output_file}")
    write_predictions(output_file, test_features, labels)


def run_build_dict(force=False):
    if os.path.exists(DICT_FILE) and not force:
        print(f"词典已存在，跳过构建: {DICT_FILE}")
        print("如需重建，请使用: python main.py dict --force")
        return
    print("正在构建地址词典...")
    build_address_dict(TRAIN_FILE, DICT_FILE)
    print(f"词典已保存至: {DICT_FILE}")


def run_train(use_llm=None):
    paths = resolve_paths()
    prepare_resources(paths)
    processor = create_processor(use_llm=use_llm)
    train_and_save(processor)
    print("训练完成！")


def run_predict(use_llm=None):
    model_file = MODEL_PATH + '.pkl'
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"未找到模型文件: {model_file}\n请先运行: python main.py train"
        )
    paths = resolve_paths()
    prepare_resources(paths)
    processor = create_processor(use_llm=use_llm)
    predict_and_save(processor, paths['conll_test_file'], paths['output_file'])
    print("预测完成！")


def run_all(use_llm=None):
    paths = resolve_paths()
    prepare_resources(paths)
    processor = create_processor(use_llm=use_llm)
    train_and_save(processor)
    predict_and_save(processor, paths['conll_test_file'], paths['output_file'])
    print("所有任务已完成！")


def main(mode='all', use_llm=None, force_dict=False):
    if mode == 'dict':
        run_build_dict(force=force_dict)
    elif mode == 'train':
        run_train(use_llm=use_llm)
    elif mode == 'predict':
        run_predict(use_llm=use_llm)
    elif mode == 'all':
        run_all(use_llm=use_llm)
    else:
        raise ValueError(f"未知运行模式: {mode}")


if __name__ == '__main__':
    main()
