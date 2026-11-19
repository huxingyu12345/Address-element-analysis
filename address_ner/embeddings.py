# -*- coding: utf-8 -*-
"""预训练语言模型句向量。"""
import numpy as np
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from .config import CUDA_DEVICE, EMBEDDING_SIZE, LLM_MODEL_NAME

_EMBEDDING_FIELDS = ('text_embedding', 'embedding', 'embeddings', 'sentence_embedding')


class EmbeddingClient:
    def __init__(self, model_name=LLM_MODEL_NAME, use_gpu=True):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.enabled = True
        self.pipeline = None
        self._init_pipeline()

    def _init_pipeline(self):
        try:
            if self.use_gpu and torch.cuda.is_available():
                device = CUDA_DEVICE
                print("使用GPU加速语言模型推理")
            else:
                device = 'cpu'
                if self.use_gpu:
                    print("GPU不可用，使用CPU进行语言模型推理")
                else:
                    print("使用CPU进行语言模型推理")

            self.pipeline = pipeline(
                Tasks.sentence_embedding,
                model=self.model_name,
                device=device,
            )
            print(f"成功加载语言模型: {self.model_name} (设备: {device})")
        except Exception as e:
            print(f"加载语言模型失败，将不使用语言模型特征: {e}")
            self.enabled = False

    def encode(self, text):
        """返回文本嵌入；失败时回退为零向量。"""
        if not self.enabled:
            return np.zeros(EMBEDDING_SIZE)

        try:
            result = self.pipeline(text)
            embedding = self._extract_embedding(result)
            if embedding is None:
                return np.zeros(EMBEDDING_SIZE)
            return embedding
        except Exception as e:
            print(f"获取语言模型嵌入失败: {e}")
            return np.zeros(EMBEDDING_SIZE)

    def _extract_embedding(self, result):
        if isinstance(result, np.ndarray):
            return result
        if isinstance(result, torch.Tensor):
            return result.detach().cpu().numpy()
        if isinstance(result, list):
            return np.array(result)
        if not hasattr(result, '__getitem__'):
            print(f"语言模型返回的结果格式不支持: {type(result)}")
            return None

        embedding = None
        for field in _EMBEDDING_FIELDS:
            if field in result:
                embedding = result[field]
                break

        if embedding is None:
            available = list(result.keys()) if hasattr(result, 'keys') else '无法获取字段信息'
            print(f"语言模型返回结果中不包含已知的嵌入字段，可用字段: {available}")
            return None
        if isinstance(embedding, np.ndarray):
            return embedding
        if isinstance(embedding, (list, torch.Tensor)):
            return np.array(embedding)

        print(f"语言模型返回的嵌入格式不支持: {type(embedding)}")
        return None
