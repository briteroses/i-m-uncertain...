import argparse

from utils.parser import get_parser
from data.registry import DATASET_LABEL_REGISTRY, MODEL_TYPE_REGISTRY
import _evaluate
import _generate

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
    for use_custom_prompt in [False, True]:
        for model_name in MODEL_TYPE_REGISTRY.keys():
            for dataset_name in DATASET_LABEL_REGISTRY.keys():
                setattr(generation_args, 'model_name', model_name)
                setattr(args, 'model_name', model_name)
                setattr(generation_args, 'dataset_name', dataset_name)
                setattr(args, 'dataset_name', dataset_name)
                setattr(generation_args, 'use_custom_prompt', use_custom_prompt)
                setattr(args, 'use_custom_prompt', use_custom_prompt)
                print(f"Evaluating baseline CCS on {model_name} model and {dataset_name} dataset {'with a custom prompt' if use_custom_prompt else ''}...")
                _generate.main(generation_args)
                _evaluate.main(args, generation_args)
                print("\n\n")