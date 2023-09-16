DATASET_LABEL_REGISTRY = {
    "imdb": ["negative", "positive"], # This is for normal IMDB
    "amazon-polarity": ["negative", "positive"],
    "ag-news": ["politics", "sports", "business", "technology"],
    "dbpedia-14": ["company", "educational institution", "artist", "athlete", "office holder", "mean of transportation", "building", "natural place", "village", "animal",  "plant",  "album",  "film",  "written work"],
    "copa": ["choice 1", "choice 2"],
    "rte": ["yes", "no"],   # whether entail
    "boolq": ["false", "true"],
    "qnli": ["yes", "no"],  # represent whether entail
    "piqa": ["solution 1", "solution 2"],
    "story-cloze": ["choice 1", "choice 2"],
}

MODEL_TYPE_REGISTRY = {
    "gpt2": "decoder",
    "gpt2-medium": "decoder",
    "gpt2-large": "decoder",
    "gpt2-xl": "decoder",
    "gpt-neo": "decoder",
    "gpt-j": "decoder",
    "T0pp": "encoder_decoder",
    "unifiedqa": "encoder_decoder",
    "T5": "encoder_decoder",
    "deberta-mnli": "encoder",
    "deberta": "encoder",
    "roberta-mnli": "encoder",
}

get_label_name_for_dataset = lambda dataset_name: "label" if dataset_name != "story-cloze" else "answer_right_ending"
