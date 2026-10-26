"""Official LoCoMo QA scorer, extracted verbatim from the upstream harness.

Source: snap-research/locomo, task_eval/evaluation.py. Only the F1 path is
kept (the paper's metric); the bert_score / rouge paths and the retrieval /
dialogue drivers are omitted, which removes the bert_score import the upstream
module makes at top level. Category semantics are unchanged: categories 2/3/4
score by token F1, category 1 by multi-answer comma-split partial F1, category
3 references are truncated at the first semicolon, and category 5 (adversarial)
is excluded upstream of this scorer.
"""
from __future__ import annotations

import string
from collections import Counter

import numpy as np
import regex
from nltk.stem import PorterStemmer

ps = PorterStemmer()


def normalize_answer(s):
    s = s.replace(',', "")

    def remove_articles(text):
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def f1(prediction, ground_truth):
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]
    return np.mean([max([f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths])


def eval_question_answering(qas, eval_key='prediction'):
    """Per-question F1 list, official category routing, adversarial excluded."""
    scores = []
    for line in qas:
        answer = line['answer'] if type(line[eval_key]) == list else str(line['answer'])
        if line['category'] == 3:
            answer = answer.split(';')[0].strip()
        output = line[eval_key]
        if line['category'] in [2, 3, 4]:
            scores.append(f1_score(output, answer))
        elif line['category'] in [1]:
            scores.append(f1(output, answer))
        elif line['category'] in [5]:
            raise ValueError('Adversarial category 5 must be excluded before scoring')
        else:
            raise ValueError(f"Unknown LoCoMo category {line['category']}")
    return scores
