import argparse

from utils.parser import get_parser
from data.registry import DATASET_LABEL_REGISTRY, MODEL_TYPE_REGISTRY, use_train_or_test
import _evaluate
import _generate
import torch

if __name__ == '__main__':
    # set up parser for the rest of the args
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
    parser.add_argument("--roc", action="store_true", help="Generate ROC curve", default=False)
    parser.add_argument("--save_confidence_scores", action="store_true", help="Save confidence scores and true labels for ROC curve experiments", default=False)
    #parser.add_argument("--uncertainty", action="store_true", help="Run with uncertainty set to True", default=True)
    args = parser.parse_args()
    assert not (args.uncertainty and args.roc), "cannot run uncertainty and roc experiments at the same time"

    # iterate through all models and datasets via argparse namespace
    for use_custom_prompt in [True]:#[False, True]:
        for model_name in ['deberta', 'gpt-j', 'gpt2-large']: #MODEL_TYPE_REGISTRY.keys():
            if args.temporal_experiment:
                setattr(generation_args, 'model_name', model_name)
                setattr(args, 'model_name', model_name)
                setattr(generation_args, 'use_custom_prompt', use_custom_prompt)
                setattr(args, 'use_custom_prompt', use_custom_prompt)
                if generation_args.uncertainty:
                    print(f"Evaluating Uncertainty CCS on {model_name} model and temporal dataset {'with a custom prompt' if use_custom_prompt else ''}...")
                else:
                    print(f"Evaluating baseline CCS on {model_name} model and temporal dataset {'with a custom prompt' if use_custom_prompt else ''}...")
                _generate.main(generation_args)
                _evaluate.temporal_experiment(args, generation_args)
                print("\n\n")
            else:
                for dataset_name in DATASET_LABEL_REGISTRY.keys():
                    setattr(generation_args, 'model_name', model_name)
                    setattr(args, 'model_name', model_name)
                    setattr(generation_args, 'dataset_name', dataset_name)
                    setattr(args, 'dataset_name', dataset_name)
                    setattr(generation_args, 'use_custom_prompt', use_custom_prompt)
                    setattr(args, 'use_custom_prompt', use_custom_prompt)
                    setattr(generation_args, 'split', use_train_or_test(dataset_name))
                    setattr(args, 'split', use_train_or_test(dataset_name))
                    if generation_args.uncertainty:
                        print(f"Evaluating Uncertainty CCS on {model_name} model and {dataset_name} dataset {'with a custom prompt' if use_custom_prompt else ''}...")
                    else:
                        print(f"Evaluating baseline CCS on {model_name} model and {dataset_name} dataset {'with a custom prompt' if use_custom_prompt else ''}...")
                    _generate.main(generation_args)
                    _evaluate.main(args, generation_args)
                    print("\n\n")
            torch.cuda.empty_cache()  # Clear GPU memory
    print(" ~~~~~~~~~~ SUCCESS ~~~~~~~~~~ ")