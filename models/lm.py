from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    GPTNeoForCausalLM,
    AutoConfig,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
    AutoModelWithLMHead,
    AutoModelForSequenceClassification,
    T5Tokenizer
)

MODEL_REGISTRY = {
    "gpt2": ("gpt2", "decoder", GPT2LMHeadModel, GPT2Tokenizer),
    "gpt2-medium": ("gpt2-medium", "decoder", GPT2LMHeadModel, GPT2Tokenizer),
    "gpt2-large": ("gpt2-large", "decoder", GPT2LMHeadModel, GPT2Tokenizer),
    "gpt2-xl": ("gpt2-xl", "decoder", GPT2LMHeadModel, GPT2Tokenizer),
    "gpt-neo": ("EleutherAI/gpt-neo-2.7B", "decoder", AutoModelForCausalLM, AutoTokenizer),
    "gpt-j": ("EleutherAI/gpt-j-6B", "decoder", AutoModelForCausalLM, AutoTokenizer),
    "T0pp": ("bigscience/T0pp", "encoder_decoder", AutoModelForSeq2SeqLM, AutoTokenizer),
    "unifiedqa": ("allenai/unifiedqa-t5-11b", "encoder_decoder", T5ForConditionalGeneration, AutoTokenizer),
    "T5": ("t5-11b", "encoder_decoder", AutoModelWithLMHead, AutoTokenizer),
    "deberta-mnli": ("microsoft/deberta-xxlarge-v2-mnli", "encoder", AutoModelForSequenceClassification, AutoTokenizer),
    "deberta": ("microsoft/deberta-xxlarge-v2", "encoder", AutoModelForSequenceClassification, AutoTokenizer),
    "roberta-mnli": ("roberta-large-mnli", "encoder", AutoModelForSequenceClassification, AutoTokenizer),
    "t5-3b": ("t5-3b", "encoder_decoder", AutoModelWithLMHead, AutoTokenizer),
    "T0-3b": ("bigscience/T0_3B", "encoder_decoder", AutoModelForSeq2SeqLM, AutoTokenizer),
    "unifiedqa-3b": ("allenai/unifiedqa-t5-3b", "encoder_decoder", T5ForConditionalGeneration, T5Tokenizer),
    "unifiedqa-v2-3b": ("allenai/unifiedqa-v2-t5-3b-1251000", "encoder_decoder", T5ForConditionalGeneration, T5Tokenizer)
}

MODEL_REGISTRY_NAMES = list(MODEL_REGISTRY.keys())

def load_model(model_name, cache_dir=None, parallelize=False, device="cuda"):
    """
    Loads a model and its corresponding tokenizer, either parallelized across GPUs (if the model permits that; usually just use this for T5-based models) or on a single GPU
    """
    assert model_name in MODEL_REGISTRY, f"invalid model name. current implementation only supports: {MODEL_REGISTRY_NAMES}"
    full_model_name, model_type, automodeler, autotoken = MODEL_REGISTRY[model_name]

    model = automodeler.from_pretrained(full_model_name, cache_dir=cache_dir)
    tokenizer = autotoken.from_pretrained(full_model_name, cache_dir=cache_dir, model_max_length=512)
    model.eval()

    # put on the correct device
    if parallelize:
        model.parallelize()
    else:
        model = model.to(device)

    return model, tokenizer, model_type