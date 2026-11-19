# -*- coding: utf-8 -*-
"""数据处理器：词典 + 特征提取 + CoNLL 加载。"""
import numpy as np

from .config import DICT_FILE, EMBEDDING_SIZE, FEATURE_TEMPLATE, LABEL_TYPES, LLM_MODEL_NAME
from .dataset import read_conll_sents
from .dictionary import load_address_dict
from .embeddings import EmbeddingClient
from .features import FeatureExtractor


class EnhancedDataProcessor:
    def __init__(
        self,
        feature_template=FEATURE_TEMPLATE,
        dict_file=DICT_FILE,
        use_llm=True,
        model_name=LLM_MODEL_NAME,
        use_gpu=True,
    ):
        self.feature_template = feature_template
        self.dict_file = dict_file
        self.use_llm = use_llm
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.label_types = dict(LABEL_TYPES)

        load_address_dict(self.dict_file)

        self.embedder = None
        if self.use_llm:
            self.embedder = EmbeddingClient(model_name=self.model_name, use_gpu=self.use_gpu)
            self.use_llm = self.embedder.enabled

        self.extractor = FeatureExtractor(
            feature_template=self.feature_template,
            embedder=self.embedder if self.use_llm else None,
        )

    def get_llm_embeddings(self, text):
        if not self.use_llm or self.embedder is None:
            return np.zeros(EMBEDDING_SIZE)
        return self.embedder.encode(text)

    def word2features(self, sent, i):
        return self.extractor.word2features(sent, i)

    def sent2features(self, sent):
        return self.extractor.sent2features(sent)

    def sent2labels(self, sent):
        return self.extractor.sent2labels(sent)

    def load_conll(self, file_path):
        sents = read_conll_sents(file_path)
        features = [self.sent2features(s) for s in sents]
        labels = [self.sent2labels(s) for s in sents]
        return features, labels
