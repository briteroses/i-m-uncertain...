import argparse

from utils.parser import get_parser
from models.lm import MODEL_REGISTRY_NAMES
from data.contrast import DATASET_REGISTRY
import _evaluate
import _generate

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
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to use")
    parser.add_argument("--num_examples", type=int, default=1000, help="Number of examples to generate")
    # which hidden states we extract
    parser.add_argument("--use_decoder", action="store_true", help="Whether to use the decoder; only relevant if model_type is encoder-decoder. Uses encoder by default (which usually -- but not always -- works better)")
    parser.add_argument("--layer", type=int, default=-1, help="Which layer to use (if not all layers)")
    parser.add_argument("--all_layers", action="store_true", help="Whether to use all layers or not")
    parser.add_argument("--token_idx", type=int, default=-1, help="Which token to use (by default the last token)")
    # saving the hidden states
    parser.add_argument("--save_dir", type=str, default="generated_hidden_states", help="Directory to save the hidden states")

    return parser

if __name__ == '__main__':
    # set up base args
    parser = get_parser()
    generation_args = parser.parse_args()
    parser.add_argument("--nepochs", type=int, default=1000)
    parser.add_argument("--ntries", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ccs_batch_size", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ccs_device", type=str, default="cuda")
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--var_normalize", action="store_true")
    args = parser.parse_args()

    # iterate through all models and datasets via argparse namespace
    for model_name in MODEL_REGISTRY_NAMES:
        for dataset_name in DATASET_REGISTRY:
            setattr(generation_args, 'model_name', model_name)
            setattr(args, 'model_name', model_name)
            setattr(generation_args, 'dataset_name', dataset_name)
            setattr(args, 'dataset_name', dataset_name)
            # parser.parse_args(args=['--model_name', model_name], namespace=generation_args)
            # parser.parse_args(args=['--model_name', model_name], namespace=args)
            # parser.parse_args(args=['--dataset_name', dataset_name], namespace=generation_args)
            # parser.parse_args(args=['--dataset_name', dataset_name], namespace=args)
            print(f"Evaluating baseline CCS on {model_name} model and {dataset_name} dataset...")
            _generate.main(generation_args)
            _evaluate.main(args, generation_args)
            print("\n\n")