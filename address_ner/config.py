# -*- coding: utf-8 -*-
"""路径、超参数和标签定义。"""
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

TRAIN_FILE = os.path.join(ROOT_DIR, 'data', 'train_dev.txt')
DEV_FILE = os.path.join(ROOT_DIR, 'data', 'dev.txt')
FINAL_TEST_FILE = os.path.join(ROOT_DIR, 'data', 'final_test.txt')
CONLL_TEST_FILE = os.path.join(ROOT_DIR, 'data', 'test.conll')
DEV_CONLL_FILE = os.path.join(ROOT_DIR, 'data', 'dev.conll')
DICT_FILE = os.path.join(ROOT_DIR, 'address_dict.txt')

EXP_DIR = os.path.join(ROOT_DIR, 'experiments', 'enhanced_model')
MODEL_PATH = os.path.join(EXP_DIR, 'enhanced_crf')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')
PRED_FILE = os.path.join(OUTPUT_DIR, 'enhanced_pred.txt')
DEV_PRED_FILE = os.path.join(OUTPUT_DIR, 'enhanced_dev_pred.txt')

FEATURE_TEMPLATE = 10
USE_LLM = True
LLM_MODEL_NAME = 'tiansz/bert-base-chinese'
CUDA_DEVICE = 'cuda:7'
EMBEDDING_SIZE = 768
WORD_EMB_DIM = 20
SENT_EMB_DIM = 10
SENT_EMB_STRIDE = 5
LLM_KEYWORDS = ('省', '市', '区', '县', '镇', '乡', '街道', '路', '号')

DICT_MIN_FREQ = 2
DICT_FREQ_SCALE = 1000

CRF_ALGORITHM = 'lbfgs'
CRF_C1 = 0.2
CRF_C2 = 0.1
CRF_MAX_ITERATIONS = 500
CRF_ALL_POSSIBLE_TRANSITIONS = True
TOP_FEATURE_N = 10

LABEL_TYPES = {
    'prov': '省份',
    'city': '城市',
    'district': '区县',
    'town': '乡镇街道',
    'road': '道路',
    'roadno': '道路号码',
    'poi': '兴趣点',
    'subpoi': '子兴趣点',
    'houseno': '门牌号',
    'floorno': '楼层',
    'devzone': '开发区',
    'community': '社区',
    'cellno': '单元号',
    'assist': '辅助定位',
    'village_group': '村组',
    'intersection': '路口',
    'distance': '距离',
}
