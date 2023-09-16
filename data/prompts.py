import numpy as np
import pandas as pd
import random
from promptsource.templates import DatasetTemplates
from copy import deepcopy

from data.registry import DATASET_LABEL_REGISTRY, MODEL_TYPE_REGISTRY

def negAndPosLabels(label, dataset_name):
    '''
            When len(candicate) is larger than 2, randomly select the correctness
            Then randomly select the candidate, and return the true label
    '''
    num_labels = len(DATASET_LABEL_REGISTRY[dataset_name])
    if num_labels == 2:
        neg_label, pos_label = 0, 1
        if dataset_name == "story-cloze":
            neg_label, pos_label = neg_label + 1, pos_label + 1
        return neg_label, pos_label
    else:
        candidates = list(range(num_labels))
        candidates.pop(label)
        neg_label = random.sample(candidates, 1)[0]
        pos_label = label
        if np.random.uniform() > 0.5:
            neg_label, pos_label = pos_label, neg_label
        return neg_label, pos_label

def concatAnswer(question, ans, mdl_name):
    '''
    alternate implementation for ContrastDataset.encode() method in contrast.py
    we can switch these in and out if we want; hopefully ContrastDataset.encode() default just works well
    '''
    # Add a ? at the end of the question if not;
    # Add an A: before the answer.

    if 'gpt' not in mdl_name and "roberta" not in mdl_name:  # Do not have `\n` token.
        # TODO: check whether this is valid
        question = question.replace("\n", " ")
    if ans == "":  # null one, don't do anything
        return question

    # for bert model, should add [SEP]
    if 'deberta' in mdl_name:
        return question + " [SEP] " + ans
    elif "roberta" in mdl_name:
        return question + "</s></s>" + ans
    elif "gpt" in mdl_name:
        if question[-1] != '\n' and question[-1] != " ":
            return question + '\n' + ans
        return question + ans
    else:  # T5 based moel
        if question[-1] == "\n" or question[-1] == " ":
            return question + ans
        return question + " " + ans