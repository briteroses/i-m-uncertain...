import argparse

def get_parser():
    """
    Returns the parser we will use for generate.py and evaluate.py
    (We include it here so that we can use the same parser for both scripts)
    """
    parser = argparse.ArgumentParser()
    # setting up model
    parser.add_argument("--model_name", type=str, default="T5", help="Name of the model to use")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for the model and tokenizer")
    parser.add_argument("--parallelize", action="store_true", help="Whether to parallelize the model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for the model")
    # setting up data
    parser.add_argument("--dataset_name", type=str, default="imdb", help="Name of the dataset to use")
    parser.add_argument("--split", type=str, default="test", help="Which split of the dataset to use")
    parser.add_argument("--prompt_idx", type=int, default=0, help="Which prompt to use")
    parser.add_argument("--use_custom_prompt", type=bool, default=True, help="Whether to use a custom prompt")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to use")
    parser.add_argument("--num_examples", type=int, default=10, help="Number of examples to generate")
    # which hidden states we extract
    parser.add_argument("--use_decoder", action="store_true", help="Whether to use the decoder; only relevant if model_type is encoder-decoder. Uses encoder by default (which usually -- but not always -- works better)")
    parser.add_argument("--layer", type=int, default=-1, help="Which layer to use (if not all layers)")
    parser.add_argument("--all_layers", action="store_true", help="Whether to use all layers or not")
    parser.add_argument("--token_idx", type=int, default=-1, help="Which token to use (by default the last token)")
    # saving the hidden states
    parser.add_argument("--save_dir", type=str, default="generated_hidden_states", help="Directory to save the hidden states")
    # are we in uncertainty mode?
    parser.add_argument("--uncertainty", type=bool, default=True, help="Whether to do uncertainty experiments")
    return parser