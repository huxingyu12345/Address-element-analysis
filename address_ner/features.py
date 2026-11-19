# -*- coding: utf-8 -*-
"""字符级 / 词级 / 语言模型特征。"""
import re
import string

import jieba

from .config import (
    FEATURE_TEMPLATE,
    LLM_KEYWORDS,
    SENT_EMB_DIM,
    SENT_EMB_STRIDE,
    WORD_EMB_DIM,
)


def _is_chinese(char):
    return bool(re.match(r'[\u4e00-\u9fff]', char))


def _locate_word(words, char_index):
    """定位字符所属词及其在分词结果中的下标。"""
    char_pos = 0
    current_word = None
    word_idx = 0
    for w in words:
        if char_pos <= char_index < char_pos + len(w):
            current_word = w
            word_idx = words.index(w)
            break
        char_pos += len(w)
    return current_word, word_idx, char_pos


def _add_embedding_slice(features, prefix, vector, dim, error_msg):
    try:
        embedding_dim = min(dim, len(vector))
        for j in range(embedding_dim):
            if isinstance(j, int) and 0 <= j < len(vector):
                features[f'{prefix}_{j}'] = float(vector[j])
    except Exception as e:
        print(f"{error_msg}: {e}")


class FeatureExtractor:
    def __init__(self, feature_template=FEATURE_TEMPLATE, embedder=None):
        self.feature_template = feature_template
        self.embedder = embedder

    @property
    def use_llm(self):
        return self.embedder is not None and self.embedder.enabled

    def sent2labels(self, sent):
        return [x[-1] for x in sent]

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def word2features(self, sent, i):
        """为第 i 个字符提取特征。"""
        word = sent[i][0]
        text = ''.join([x[0] for x in sent])
        words = list(jieba.cut(text))
        current_word, word_idx, char_pos = _locate_word(words, i)

        features = {
            'w': word,
            'is_digit': word.isdigit(),
            'is_punctuation': word in string.punctuation,
            'is_chinese': _is_chinese(word),
            'is_alpha': word.isalpha() and not _is_chinese(word),
            'is_alnum': word.isalnum(),
            'is_space': word.isspace(),
        }

        if current_word:
            char_position = i - (char_pos - len(current_word))
            features['char_position'] = char_position
            features['char_position_ratio'] = char_position / len(current_word)
            features['is_first_char'] = (char_position == 0)
            features['is_last_char'] = (char_position == len(current_word) - 1)
            features['word'] = current_word
            if word_idx > 0:
                features['-1:word'] = words[word_idx - 1]
                features['-1-0:word'] = words[word_idx - 1] + current_word
            if word_idx < len(words) - 1:
                features['+1:word'] = words[word_idx + 1]
                features['0+1:word'] = current_word + words[word_idx + 1]

        if len(word) >= 1:
            features['prefix1'] = word[:1]
            features['suffix1'] = word[-1:]
        if len(word) >= 2:
            features['prefix2'] = word[:2]
            features['suffix2'] = word[-2:]
        if len(word) >= 3:
            features['prefix3'] = word[:3]
            features['suffix3'] = word[-3:]

        self._add_char_context(features, sent, i, word)
        self._add_llm_features(features, sent, i, word, text, current_word)
        return features

    def _add_char_context(self, features, sent, i, word):
        if i > 0:
            word1 = sent[i - 1][0]
            features.update({
                '-1:w': word1,
                '-1-0:w': word1 + word,
                '-1:is_chinese': _is_chinese(word1),
                '-1:is_digit': word1.isdigit(),
            })
        else:
            features['BOS'] = True

        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            features.update({
                '+1:w': word1,
                '0+1:w': word + word1,
                '+1:is_chinese': _is_chinese(word1),
                '+1:is_digit': word1.isdigit(),
            })
        else:
            features['EOS'] = True

        if self.feature_template > 2 and i > 1:
            word2 = sent[i - 2][0]
            features.update({
                '-2:w': word2,
                '-2-1:w': word2 + sent[i - 1][0],
                '-2-1-0:w': word2 + sent[i - 1][0] + word,
                '-2:is_chinese': _is_chinese(word2),
                '-2:is_digit': word2.isdigit(),
            })
        elif self.feature_template > 2:
            features['BOS2'] = True

        if self.feature_template > 2 and i < len(sent) - 2:
            word2 = sent[i + 2][0]
            features.update({
                '+2:w': word2,
                '+1+2:w': sent[i + 1][0] + word2,
                '0+1+2:w': word + sent[i + 1][0] + word2,
                '+2:is_chinese': _is_chinese(word2),
                '+2:is_digit': word2.isdigit(),
            })
        elif self.feature_template > 2:
            features['EOS2'] = True

    def _add_llm_features(self, features, sent, i, word, text, current_word):
        if not self.use_llm:
            return
        if not (i == 0 or i == len(sent) - 1 or word in LLM_KEYWORDS):
            return

        if current_word:
            word_embedding = self.embedder.encode(current_word)
            _add_embedding_slice(features, 'word_emb', word_embedding, WORD_EMB_DIM, '处理词嵌入时出错')

        if i % SENT_EMB_STRIDE == 0:
            sentence_embedding = self.embedder.encode(text)
            _add_embedding_slice(features, 'sent_emb', sentence_embedding, SENT_EMB_DIM, '处理句子嵌入时出错')
