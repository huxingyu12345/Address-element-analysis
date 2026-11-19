# -*- coding: utf-8 -*-
"""CRF 训练、预测与持久化。"""
import os
import pickle

from sklearn_crfsuite import CRF

from .config import (
    CRF_ALGORITHM,
    CRF_ALL_POSSIBLE_TRANSITIONS,
    CRF_C1,
    CRF_C2,
    CRF_MAX_ITERATIONS,
    TOP_FEATURE_N,
)


class EnhancedCRFTrainer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.crf = CRF(
            algorithm=CRF_ALGORITHM,
            c1=CRF_C1,
            c2=CRF_C2,
            max_iterations=CRF_MAX_ITERATIONS,
            all_possible_transitions=CRF_ALL_POSSIBLE_TRANSITIONS,
            verbose=True,
        )

    def train(self, train_features, train_labels):
        self.crf.fit(train_features, train_labels)

    def predict(self, test_features):
        return self.crf.predict(test_features)

    def save_model(self):
        model_dir = os.path.dirname(self.model_name)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        pickle.dump(self.crf, open(self.model_name + '.pkl', 'wb'))

    def load_model(self):
        self.crf = pickle.load(open(self.model_name + '.pkl', 'rb'))

    def get_top_features(self, top_n=TOP_FEATURE_N):
        """获取 CRF 模型中最重要的特征。"""
        if not hasattr(self.crf, 'state_features_'):
            return {}

        top_features = {}
        for label in self.crf.classes_:
            if label in self.crf.state_features_:
                sorted_features = sorted(
                    self.crf.state_features_[label].items(),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )
                top_features[label] = sorted_features[:top_n]
        return top_features
