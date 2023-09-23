import sys
from copy import deepcopy

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from promptsource.templates import DatasetTemplates
from datasets import load_dataset

from data.prompts import CustomPrompt, negAndPosLabels
from data.registry import DATASET_LABEL_REGISTRY, MODEL_TYPE_REGISTRY, get_label_name_for_dataset, PROMPT_DICT

IDK_DUMMY_LABEL = -1 # hopefully this dummy value is safe...
IDK_ANSWER_TEXT = ["Uncertain", "I don't know"]

class ContrastDataset(Dataset):
    """
    Given a dataset and tokenizer (from huggingface), along with a collection of prompts for that dataset from promptsource and a corresponding prompt index, 
    returns a dataset that creates contrast pairs using that prompt
    
    Truncates examples larger than max_len, which can mess up contrast pairs, so make sure to only give it examples that won't be truncated.
    """
    def __init__(self, raw_dataset, tokenizer, prompt,
                 model_name="deberta", dataset_name="imdb", use_decoder=False, device="cuda"):

        # data and tokenizer
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device
        
        # for formatting the answers
        assert model_name in MODEL_TYPE_REGISTRY.keys(), "invalid model name given"
        self.model_name = model_name
        self.use_decoder = use_decoder
        if self.use_decoder:
            assert MODEL_TYPE_REGISTRY[self.model_name] != "encoder"

        self.dataset_name = dataset_name

        # prompt
        self.prompt = prompt

    def __len__(self):
        return len(self.raw_dataset)

    def encode(self, nl_prompt, truncation=True):
        """
        Tokenize a given natural language prompt (from after applying self.prompt to an example)
        
        For encoder-decoder models, we can either:
        (1) feed both the question and answer to the encoder, creating contrast pairs using the encoder hidden states
            (which uses the standard tokenization, but also passes the empty string to the decoder), or
        (2) feed the question the encoder and the answer to the decoder, creating contrast pairs using the decoder hidden states
        
        If self.decoder is True we do (2), otherwise we do (1).
        """
        # get question and answer from prompt
        question, answer = nl_prompt
        
        # tokenize the question and answer (depending upon the model type and whether self.use_decoder is True)
        if MODEL_TYPE_REGISTRY[self.model_name] == "encoder_decoder":
            input_ids = self.get_encoder_decoder_input_ids(question, answer, truncation=truncation)
        elif MODEL_TYPE_REGISTRY[self.model_name] == "encoder":
            input_ids = self.get_encoder_input_ids(question, answer, truncation=truncation)
        else:
            input_ids = self.get_decoder_input_ids(question, answer, truncation=truncation)
        
        # get rid of the batch dimension since this will be added by the Dataloader
        if input_ids["input_ids"].shape[0] == 1:
            for k in input_ids:
                input_ids[k] = input_ids[k].squeeze(0)

        return input_ids


    def get_encoder_input_ids(self, question, answer, truncation=True):
        """
        Format the input ids for encoder-only models; standard formatting.
        """
        combined_input = question + " " + answer
        input_ids = self.tokenizer(combined_input, truncation=truncation, padding="max_length", return_tensors="pt")

        return input_ids


    def get_decoder_input_ids(self, question, answer, truncation=True):
        """
        Format the input ids for encoder-only models.
        This is the same as get_encoder_input_ids except that we add the EOS token at the end of the input (which apparently can matter)
        """
        combined_input = question + " " + answer + self.tokenizer.eos_token
        input_ids = self.tokenizer(combined_input, truncation=truncation, padding="max_length", return_tensors="pt")

        return input_ids


    def get_encoder_decoder_input_ids(self, question, answer, truncation=True):
        """
        Format the input ids for encoder-decoder models.
        There are two cases for this, depending upon whether we want to use the encoder hidden states or the decoder hidden states.
        """
        if self.use_decoder:
            # feed the same question to the encoder but different answers to the decoder to construct contrast pairs
            input_ids = self.tokenizer(question, truncation=truncation, padding="max_length", return_tensors="pt")
            decoder_input_ids = self.tokenizer(answer, truncation=truncation, padding="max_length", return_tensors="pt")
        else:
            # include both the question and the answer in the input for the encoder
            # feed the empty string to the decoder (i.e. just ignore it -- but it needs an input or it'll throw an error)
            input_ids = self.tokenizer(question, answer, truncation=truncation, padding="max_length", return_tensors="pt")
            decoder_input_ids = self.tokenizer("", return_tensors="pt")
        
        # move everything into input_ids so that it's easier to pass to the model
        input_ids["decoder_input_ids"] = decoder_input_ids["input_ids"]
        input_ids["decoder_attention_mask"] = decoder_input_ids["attention_mask"]

        return input_ids


    def __getitem__(self, index):
        neg_prompt, pos_prompt = self.get_prompts_at_index(index)

        # tokenize
        neg_ids, pos_ids = self.encode(neg_prompt), self.encode(pos_prompt)

        # verify these are different (e.g. tokenization didn't cut off the difference between them)
        if self.use_decoder and MODEL_TYPE_REGISTRY[self.model_name] == "encoder_decoder":
            assert (neg_ids["decoder_input_ids"] - pos_ids["decoder_input_ids"]).sum() != 0, print("The decoder_input_ids for the contrast pairs are the same!", neg_ids, pos_ids)
        else:
            assert (neg_ids["input_ids"] - pos_ids["input_ids"]).sum() != 0, print("The input_ids for the contrast pairs are the same!", neg_ids, pos_ids)

        label_name = get_label_name_for_dataset(self.dataset_name)
        ground_truth_label = self.raw_dataset[int(index)][label_name]

        return neg_ids, pos_ids, neg_prompt, pos_prompt, ground_truth_label

    def get_prompts_at_index(self, index):
        # get the original example
        data = self.raw_dataset[int(index)]
        label_name = get_label_name_for_dataset(self.dataset_name)
        ground_truth_label = data[label_name]

        # get the possible labels
        # (for simplicity assume the binary case for contrast pairs)
        # label_list = self.prompt.get_answer_choices_list(data)
        # assert len(label_list) == 2, print("Make sure there are only two possible answers! Actual number of answers:", label_list)

        # reconvert to dataset format but with fake/candidate labels to create the contrast pair
        neg_label, pos_label = negAndPosLabels(ground_truth_label, self.dataset_name)

        neg_example, pos_example = deepcopy(data), deepcopy(data)
        neg_example[label_name] = neg_label
        neg_example["neg_label"] = neg_label
        neg_example["pos_label"] = pos_label
        pos_example[label_name] = pos_label
        pos_example["neg_label"] = neg_label
        pos_example["pos_label"] = pos_label

        # construct contrast pairs by answering the prompt with the two different possible labels
        # (for example, label 0 might be mapped to "no" and label 1 might be mapped to "yes")
        neg_prompt, pos_prompt = self.prompt.apply(neg_example), self.prompt.apply(pos_example)
        
        return neg_prompt, pos_prompt


class UncertaintyContrastDataset(ContrastDataset):
    def __init__(self, *args, **kwargs):
        # can just use exact same init as contrast dataset
        super().__init__(*args, **kwargs)
        idk_text_selection = 0
        assert idk_text_selection < len(IDK_ANSWER_TEXT), "provided index out of range of \'idk\' answer options"
        self.idk_text_selection = idk_text_selection

    def __getitem__(self, index):
        neg_prompt, pos_prompt, idk_prompt = self.get_prompts_at_index(index)

        # tokenize
        neg_ids, pos_ids, idk_ids = self.encode(neg_prompt), self.encode(pos_prompt), self.encode(idk_prompt)

        # verify these are different (e.g. tokenization didn't cut off the difference between them)
        if self.use_decoder and MODEL_TYPE_REGISTRY[self.model_name] == "encoder_decoder":
            assert (neg_ids["decoder_input_ids"] - pos_ids["decoder_input_ids"]).sum() != 0, "The decoder_input_ids for the contrast pairs are the same!"
            assert (pos_ids["decoder_input_ids"] - idk_ids["decoder_input_ids"]).sum() != 0, "The decoder_input_ids for the contrast pairs are the same!"
            assert (idk_ids["decoder_input_ids"] - neg_ids["decoder_input_ids"]).sum() != 0, "The decoder_input_ids for the contrast pairs are the same!"
        else:
            assert (neg_ids["input_ids"] - pos_ids["input_ids"]).sum() != 0, "The input_ids for the contrast pairs are the same!"
            assert (pos_ids["input_ids"] - idk_ids["input_ids"]).sum() != 0, "The input_ids for the contrast pairs are the same!"
            assert (idk_ids["input_ids"] - neg_ids["input_ids"]).sum() != 0, "The input_ids for the contrast pairs are the same!"

        label_name = get_label_name_for_dataset(self.dataset_name)
        ground_truth_label = self.raw_dataset[int(index)][label_name]

        return neg_ids, pos_ids, idk_ids, neg_prompt, pos_prompt, idk_prompt, ground_truth_label

    def get_prompts_at_index(self, index):
        data = self.raw_dataset[int(index)]

        label_name = get_label_name_for_dataset(self.dataset_name)
        ground_truth_label = data[label_name]

        neg_label, pos_label = negAndPosLabels(ground_truth_label, self.dataset_name)
        neg_example, pos_example, idk_example = deepcopy(data), deepcopy(data), deepcopy(data)
        neg_example[label_name] = neg_label
        neg_example["neg_label"] = neg_label
        neg_example["pos_label"] = pos_label
        pos_example[label_name] = pos_label
        pos_example["neg_label"] = neg_label
        pos_example["pos_label"] = pos_label
        idk_example[label_name] = IDK_DUMMY_LABEL
        idk_example["neg_label"] = neg_label
        idk_example["pos_label"] = pos_label

        neg_prompt, pos_prompt = self.prompt.apply(neg_example), self.prompt.apply(pos_example)
        idk_question, _ = self.prompt.apply(idk_example)
        idk_answer = IDK_ANSWER_TEXT[self.idk_text_selection]
        idk_prompt = (idk_question, idk_answer)

        return neg_prompt, pos_prompt, idk_prompt


def getLoadName(set_name):
    if set_name in ["imdb", "amazon_polarity", "ag_news", "dbpedia_14", "piqa"]:
        return [set_name]
    elif set_name in ["copa", "rte", "boolq"]:
        return ["super_glue", set_name]
    elif set_name in ["qnli"]:
        return ["glue", set_name]
    elif set_name == "story_cloze":
        return ["story_cloze", "2016"]


def get_contrast_dataloader(dataset_name, split, tokenizer, prompt_idx, use_custom_prompt=False,
                            batch_size=16, num_examples=1000,
                            model_name="deberta", use_decoder=False,
                            device="cuda", pin_memory=True, num_workers=1,
                            use_uncertainty=False):
    """
    Creates a dataloader for a given dataset (and its split), tokenizer, and prompt index

    Takes a random subset of (at most) num_examples samples from the dataset that are not truncated by the tokenizer.
    """
    # load the raw dataset
    if dataset_name != "story-cloze":
        raw_dataset = load_dataset(*getLoadName(dataset_name))[split]
    else:
        raw_dataset = load_dataset(*getLoadName(dataset_name), data_dir="./datasets/rawdata")[split]

    # load all the prompts for that dataset
    if use_custom_prompt and dataset_name in PROMPT_DICT and prompt_idx < len(PROMPT_DICT[dataset_name]):
        formatted_prompt, format_list = PROMPT_DICT[dataset_name][prompt_idx]
        source_prompt = CustomPrompt(dataset_name, formatted_prompt, format_list)
    else:
        if use_custom_prompt:
            print("prompt index was outside of provided custom prompts; defaulting to promptsource...", file=sys.stderr)
        prompt_idx = 0
        all_prompts = DatasetTemplates(*getLoadName(dataset_name))
        prompt_name_list = list(all_prompts.name_to_id_mapping.keys())
        source_prompt = all_prompts[prompt_name_list[prompt_idx]]

    # create the ConstrastDataset
    dataset_object = UncertaintyContrastDataset if use_uncertainty else ContrastDataset
    contrast_dataset = dataset_object(raw_dataset, tokenizer, source_prompt, 
                                       model_name=model_name, dataset_name=dataset_name, use_decoder=use_decoder, 
                                       device=device)

    # get a random permutation of the indices; we'll take the first num_examples of these that do not get truncated
    # then, we ignore examples that would be truncated (since this messes up contrast pairs),
    # up until we reach the desired number of examples to keep
    random_idxs = np.random.permutation(len(contrast_dataset))
    keep_idxs = []
    for idx in random_idxs:
        if use_uncertainty:
            neg_prompt, pos_prompt, _ = contrast_dataset.get_prompts_at_index(int(idx))
        else:
            neg_prompt, pos_prompt = contrast_dataset.get_prompts_at_index(int(idx))
        neg_text = neg_prompt[0] + " " + neg_prompt[1]
        pos_text = pos_prompt[0] + " " + pos_prompt[1]
        if len(tokenizer.encode(neg_text, truncation=False)) < tokenizer.model_max_length - 12 \
                and len(tokenizer.encode(pos_text, truncation=False)) < tokenizer.model_max_length - 12:  # include small margin to be conservative
            keep_idxs.append(idx)
            if len(keep_idxs) >= num_examples:
                break

    # create and return the corresponding dataloader
    subset_dataset = torch.utils.data.Subset(contrast_dataset, keep_idxs)
    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)

    return dataloader