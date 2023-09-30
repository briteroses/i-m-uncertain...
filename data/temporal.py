import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from promptsource.templates import DatasetTemplates
from datasets import Dataset

from data.contrast import ContrastDataset, IDK_DUMMY_LABEL
from data.registry import SAVE_PREFIX, MODEL_TYPE_REGISTRY

ESTIMATED_TRAINING_CUTOFFS = {
    'deberta': '2/10/2021',
    'gpt-j': '8/31/2021',
    'gpt2-large': '8/20/2019',
    'T0pp': '10/12/2021',
    'unifiedqa': '11/13/2020',
}

PROMPTS_FOR_TEMPORAL = [
    (
        "Consider the following statement: ''' {} '''\nBetween true and false, this statement is ",
        lambda text: text.capitalize(),
    ),
    (
        "Consider the following statement: ''' {} '''\nIs this statement true or false?",
        lambda text: text.capitalize(),
    ),
    (
        "Is it true or false that {}?",
        lambda text: text.rstrip('.'),
    ),
]

class TemporalDataset(ContrastDataset):
    def __init__(self, *args, **kwargs):
        # can just use exact same init as contrast dataset
        super().__init__(*args, **kwargs)
        self.idk_word = "uncertain"

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

        ground_truth_label = self.raw_dataset[int(index)]['label']

        return neg_ids, pos_ids, idk_ids, neg_prompt, pos_prompt, idk_prompt, ground_truth_label

    # inherited method is a misnomer here: the prompting happens in the main method using this dataset class
    def get_prompts_at_index(self, index):
        question = self.raw_dataset[int(index)]['question']

        neg_prompt = (question, "true")
        pos_prompt = (question, "false")
        idk_prompt = (question, self.idk_word)

        return neg_prompt, pos_prompt, idk_prompt


def temporal_dataloader(dataset_name, temporal_split, tokenizer, temporal_prompt_idx,
                            model_name="deberta", use_decoder=False,
                            device="cuda", pin_memory=True, num_workers=1):
    assert temporal_split in ("raw", "masked"), "need to specify between raw or masked dataset variant"

    path_to_temporal_dataset = SAVE_PREFIX + 'temporal_dataset.csv'
    df = pd.read_csv(path_to_temporal_dataset)
    texts = df['text'].tolist()
    labels = df['label'].astype(int).values
    timestamps = pd.to_datetime(df['timestamp'], format='%m/%d/%Y').values

    template, text_preprocessor = PROMPTS_FOR_TEMPORAL[temporal_prompt_idx]
    questions = list(map(lambda text: template.format(text_preprocessor(text)), texts))

    if temporal_split == 'raw':
        raw_dataset = Dataset.from_dict({
            'question': questions,
            'label': labels
        })

        ds = TemporalDataset(raw_dataset, tokenizer, None,
                                model_name=model_name, dataset_name=dataset_name, use_decoder=use_decoder, 
                                device=device)
        
        loader = DataLoader(ds, batch_size=1, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)

        return loader

    elif temporal_split == 'masked':
        training_cutoff = pd.to_datetime(ESTIMATED_TRAINING_CUTOFFS[model_name], format='%m/%d/%Y')
        temporal_mask = timestamps > training_cutoff
        temporal_masked_labels = np.where(temporal_mask, IDK_DUMMY_LABEL, labels)

        raw_temporal_masked_dataset = Dataset.from_dict({
            'question': questions,
            'label': temporal_masked_labels
        })
        temporal_masked_ds = TemporalDataset(raw_temporal_masked_dataset, tokenizer, None,
                                            model_name=model_name, dataset_name=dataset_name, use_decoder=use_decoder, 
                                            device=device)
        temporal_masked_loader = DataLoader(temporal_masked_ds, batch_size=1, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)

        return temporal_masked_loader