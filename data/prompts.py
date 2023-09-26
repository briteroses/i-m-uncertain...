import numpy as np
import random
from promptsource.templates import DatasetTemplates

from data.registry import DATASET_LABEL_REGISTRY, MODEL_TYPE_REGISTRY, get_label_name_for_dataset

class CustomPrompt(DatasetTemplates):
    def __init__(self, dataset_name, formatted_prompt, format_list):
        super().__init__(dataset_name)
        self.dataset_name = dataset_name
        self.label_name = get_label_name_for_dataset(self.dataset_name)
        self.formatted_prompt = formatted_prompt
        self.format_list = format_list
    
    def apply(self, example):
        formatter = []
        for example_feature in self.format_list:
            if example_feature in ("neg_label", "pos_label"):
                write_label = DATASET_LABEL_REGISTRY[self.dataset_name][example[example_feature]]
                formatter.append(write_label)
            else:
                formatter.append(example[example_feature])
        question = self.formatted_prompt.format(*formatter)
        answer = DATASET_LABEL_REGISTRY[self.dataset_name][example[self.label_name]]
        return question, answer


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