from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM

model_nicknames = {
    "gpt-j": "EleutherAI/gpt-j-6B",
    "T0pp": "bigscience/T0pp",
    "unifiedqa": "allenai/unifiedqa-t5-11b",
    "T5": "t5-11b",
    "deberta-mnli": "microsoft/deberta-xxlarge-v2-mnli",
    "deberta": "microsoft/deberta-xxlarge-v2",
    "roberta-mnli": "roberta-large-mnli",
}

def load_model(model_name, cache_dir=None, parallelize=False, device="cuda"):
    """
    Loads a model and its corresponding tokenizer, either parallelized across GPUs (if the model permits that; usually just use this for T5-based models) or on a single GPU
    """
    full_model_name = model_nicknames.get(model_name, model_name)

    # use the right automodel, and get the corresponding model type
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(full_model_name, cache_dir=cache_dir)
        model_type = "encoder_decoder"
    except:
        try:
            model = AutoModelForMaskedLM.from_pretrained(full_model_name, cache_dir=cache_dir)
            model_type = "encoder"
        except:
            model = AutoModelForCausalLM.from_pretrained(full_model_name, cache_dir=cache_dir)
            model_type = "decoder"
        
    # specify model_max_length (the max token length) to be 512 to ensure that padding works 
    # (it's not set by default for e.g. DeBERTa, but it's necessary for padding to work properly)
    tokenizer = AutoTokenizer.from_pretrained(full_model_name, cache_dir=cache_dir, model_max_length=512)
    model.eval()

    # put on the correct device
    if parallelize:
        model.parallelize()
    else:
        model = model.to(device)

    return model, tokenizer, model_type