import argparse
import sys

from models.hidden_states import get_all_hidden_states
from models.lm import load_model
from data.contrast import get_contrast_dataloader
from data.registry import DATASET_LABEL_REGISTRY, MODEL_TYPE_REGISTRY, use_train_or_test
from utils.parser import get_parser
from utils.save_and_load import save_generations, load_all_generations

from probes.uncertainty import UncertaintyDetectingCCS


def Uevaluate(args, generation_args):
    # load hidden states and labels
    generations = load_all_generations(generation_args)
    neg_hs, pos_hs, idk_hs, y = tuple(generations.values())

    # Make sure the shape is correct
    assert neg_hs.shape == pos_hs.shape == idk_hs.shape
    neg_hs, pos_hs, idk_hs = neg_hs[..., -1], pos_hs[..., -1], idk_hs[..., -1]  # take the last layer
    if neg_hs.shape[1] == 1:  # T5 may have an extra dimension; if so, get rid of it
        neg_hs = neg_hs.squeeze(1)
        pos_hs = pos_hs.squeeze(1)
        idk_hs = idk_hs.squeeze(1)

    # Very simple train/test split (using the fact that the data is already shuffled)
    neg_hs_train, neg_hs_test = neg_hs[:len(neg_hs) // 2], neg_hs[len(neg_hs) // 2:]
    pos_hs_train, pos_hs_test = pos_hs[:len(pos_hs) // 2], pos_hs[len(pos_hs) // 2:]
    idk_hs_train, idk_hs_test = idk_hs[:len(idk_hs) // 2], idk_hs[len(idk_hs) // 2:]
    y_train, y_test = y[:len(y) // 2], y[len(y) // 2:]

    # Set up CCS. Note that you can usually just use the default args by simply doing ccs = CCS(neg_hs, pos_hs, y)
    uccs = UncertaintyDetectingCCS(neg_hs_train, pos_hs_train, idk_hs_train, nepochs=args.nepochs, ntries=args.ntries, lr=args.lr, batch_size=args.ccs_batch_size, 
                    verbose=args.verbose, device=args.ccs_device, linear=args.linear, weight_decay=args.weight_decay, 
                    var_normalize=args.var_normalize)
    
    # train and evaluate CCS
    uccs.repeated_train()
    uccs_acc, uccs_coverage = uccs.get_acc(neg_hs_test, pos_hs_test, idk_hs_test, y_test)
    print(f"UCCS accuracy: {uccs_acc} | UCCS coverage: {(100.0*uccs_coverage):1f}%")

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
    for use_custom_prompt in [True]:#[False, True]:
        for model_name in ['deberta']:#MODEL_TYPE_REGISTRY.keys():
            for dataset_name in ['imdb']:#DATASET_LABEL_REGISTRY.keys():
                setattr(generation_args, 'model_name', model_name)
                setattr(args, 'model_name', model_name)
                setattr(generation_args, 'dataset_name', dataset_name)
                setattr(args, 'dataset_name', dataset_name)
                setattr(generation_args, 'use_custom_prompt', use_custom_prompt)
                setattr(args, 'use_custom_prompt', use_custom_prompt)
                setattr(generation_args, 'split', use_train_or_test(dataset_name))
                setattr(args, 'split', use_train_or_test(dataset_name))
                print(f"Evaluating baseline CCS on {model_name} model and {dataset_name} dataset {'with a custom prompt' if use_custom_prompt else ''}...")
                Ugenerate(generation_args)
                Uevaluate(args, generation_args)
                print("\n\n")
    print(" ~~~~~~~~~~ SUCCESS ~~~~~~~~~~ ")