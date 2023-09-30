import sys

from sklearn.linear_model import LogisticRegression

from utils.parser import get_parser
from utils.save_and_load import load_all_generations
from probes.CCS import CCS, MLPProbe, LinearProbe
from probes.uncertainty import UncertaintyDetectingCCS

from data.registry import DATASET_LABEL_REGISTRY, use_train_or_test, SAVE_PREFIX
from data.temporal import temporal_dataloader

from sklearn.metrics import auc
import matplotlib.pyplot as plt
import os 
import numpy as np
import torch

import json

def main(args, generation_args):
    if args.uncertainty:
        save_path = SAVE_PREFIX + "results/"+ ('linear' if args.linear else 'MLP') + "/ccs.json"
        if not os.path.exists(SAVE_PREFIX + "results/" + ('linear' if args.linear else 'MLP')):
            os.makedirs(SAVE_PREFIX + "results/" + ('linear' if args.linear else 'MLP'))
        try:
            with open(save_path, "r") as fin:
                ccs_json =  json.load(fin)
        except FileNotFoundError:
            ccs_json = {}
        if args.model_name in ccs_json and args.dataset_name in ccs_json[args.model_name]:
            print(f"CCS and UCCS results for {args.model_name} and {args.dataset_name} already generated, skipping this...")
  
    # load hidden states and labels
    generations = load_all_generations(generation_args)
    if args.uncertainty:
        neg_hs = generations['neg_hs']
        pos_hs = generations['pos_hs']
        idk_hs = generations['idk_hs']
        y = generations['labels']
    else:
        neg_hs, pos_hs, y = tuple(generations.values())

    # Make sure the shape is correct
    assert neg_hs.shape == pos_hs.shape
    neg_hs, pos_hs = neg_hs[..., -1], pos_hs[..., -1]  # take the last layer
    if neg_hs.shape[1] == 1:  # T5 may have an extra dimension; if so, get rid of it
        neg_hs = neg_hs.squeeze(1)
        pos_hs = pos_hs.squeeze(1)

    # Very simple train/test split (using the fact that the data is already shuffled)
    neg_hs_train, neg_hs_test = neg_hs[:len(neg_hs) // 2], neg_hs[len(neg_hs) // 2:]
    pos_hs_train, pos_hs_test = pos_hs[:len(pos_hs) // 2], pos_hs[len(pos_hs) // 2:]
    y_train, y_test = y[:len(y) // 2], y[len(y) // 2:]

    if args.uncertainty:
        idk_hs = idk_hs[..., -1]
        if idk_hs.shape[1] == 1:
            idk_hs = idk_hs.squeeze(1)
        idk_hs_train, idk_hs_test = idk_hs[:len(idk_hs) // 2], idk_hs[len(idk_hs) // 2:]

    # Make sure logistic regression accuracy is reasonable; otherwise our method won't have much of a chance of working
    # you can also concatenate, but this works fine and is more comparable to CCS inputs
    if not args.uncertainty:
        x_train = neg_hs_train - pos_hs_train
        x_test = neg_hs_test - pos_hs_test
        lr = LogisticRegression(class_weight="balanced")
        lr.fit(x_train, y_train)
        print("Logistic regression accuracy: {}".format(lr.score(x_test, y_test)))

    ccs = CCS(neg_hs_train, pos_hs_train, nepochs=args.nepochs, ntries=args.ntries, lr=args.lr, batch_size=args.ccs_batch_size, 
                verbose=args.verbose, device=args.ccs_device, linear=args.linear, weight_decay=args.weight_decay, 
                var_normalize=args.var_normalize)
    ccs.repeated_train()
    ccs_acc = ccs.get_acc(neg_hs_test, pos_hs_test, y_test)
    print("CCS accuracy: {}".format(ccs_acc))
    if args.uncertainty:
        uccs = UncertaintyDetectingCCS(neg_hs_train, pos_hs_train, idk_hs_train, nepochs=args.nepochs, ntries=args.ntries, lr=args.lr, batch_size=args.ccs_batch_size, 
                    verbose=args.verbose, device=args.ccs_device, linear=args.linear, weight_decay=args.weight_decay, 
                    var_normalize=args.var_normalize)
        uccs.repeated_train()
        uccs_acc, uccs_coverage = uccs.get_acc(neg_hs_test, pos_hs_test, idk_hs_test, y_test)
        print(f"UCCS accuracy: {uccs_acc:4f} | UCCS coverage: {(100.0*uccs_coverage):1f}%")
    
        save_path = SAVE_PREFIX + "results/"+ ('linear' if args.linear else 'MLP') + "/ccs.json"

        try:
            with open(save_path, "r") as fin:
                ccs_json =  json.load(fin)
        except FileNotFoundError:
            ccs_json = {}
        ccs_json[args.model_name] = ccs_json.get(args.model_name, {})
        ccs_json[args.model_name][args.dataset_name] = {'ccs_acc': ccs_acc, 'uccs_acc': uccs_acc, 'uccs_coverage': 100.0*uccs_coverage}
        print(ccs_json)
        with open(save_path, 'w') as fout:
            json.dump(ccs_json, fout)
    
    if args.roc and args.save_confidence_scores:
        scores = ccs.get_scores(neg_hs_test, pos_hs_test)
        fpr, tpr, roc_auc = ccs.compute_roc(scores, y_test)

        if fpr is not None and tpr is not None and roc_auc is not None:  # If it's not binary, we can't compute the ROC curve
            if args.uncertainty:
                if not os.path.exists(SAVE_PREFIX + "results/Uncertainty_ROC_curves/" + ('linear' if args.linear else 'MLP')):
                    os.makedirs(SAVE_PREFIX + "results/Uncertainty_ROC_curves/" + ('linear' if args.linear else 'MLP'))
            else:
                if not os.path.exists(SAVE_PREFIX + "results/ROC_curves/" + ('linear' if args.linear else 'MLP')):
                    os.makedirs(SAVE_PREFIX + "results/ROC_curves/" + ('linear' if args.linear else 'MLP'))
                
            save_path = SAVE_PREFIX + f"results/ROC_curves/" + ('linear' if args.linear else 'MLP') + f"/{args.model_name}_{args.dataset_name}.png"
            
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic Curve')
            plt.legend(loc="lower right")
            plt.savefig(save_path)
            plt.show()

            # Save confidence scores and true labels for ROC sliding window sweep
            save_dir = SAVE_PREFIX + f"results/confidence_scores_and_labels/" + ('linear' if args.linear else 'MLP') + f"/{args.model_name}/{args.dataset_name}"

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.save(os.path.join(save_dir, "confidence_scores.npy"), scores)
            np.save(os.path.join(save_dir, "true_labels.npy"), y_test)

    elif args.save_confidence_scores:
        scores = ccs.get_scores(neg_hs_test, pos_hs_test)
        fpr, tpr, roc_auc = ccs.compute_roc(scores, y_test)
        # Save confidence scores and true labels for ROC sliding window sweep
        save_dir = SAVE_PREFIX + f"results/confidence_scores_and_labels/" + ('linear' if args.linear else 'MLP') + f"/{args.model_name}/{args.dataset_name}"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, "confidence_scores.npy"), scores)
        np.save(os.path.join(save_dir, "true_labels.npy"), y_test)


def temporal_experiment(args, generation_args):
    # apologies for the spaghetti

    save_path = SAVE_PREFIX + f"results/{'linear' if args.linear else 'MLP'}/temporal.json"
    if not os.path.exists(SAVE_PREFIX + f"results/{'linear' if args.linear else 'MLP'}/"):
        os.makedirs(SAVE_PREFIX + f"results/{'linear' if args.linear else 'MLP'}/")
    try:
        with open(save_path, "r") as fin:
            temporal_json =  json.load(fin)
    except FileNotFoundError:
        temporal_json = {}
    if args.model_name in temporal_json:
        print(f"CCS and UCCS results for {args.model_name} and temporal dataset already generated, skipping this...")
    
    for set_uncertainty in [True, False]:
        setattr(generation_args, 'uncertainty', set_uncertainty)
        setattr(args, 'uncertainty', set_uncertainty)

        ccs = None
        # train on the ~7 original datasets
        # dividing epochs by 10 here to maintain good training dynamics; we're doing ~20 times the usual number of datasets
        # 20 <= (2 := size of full dataset / size of 50% train split) * (10 := 10 different datasets)
        for dataset_name in DATASET_LABEL_REGISTRY.keys():
            setattr(generation_args, 'dataset_name', dataset_name)
            setattr(args, 'dataset_name', dataset_name)
            setattr(generation_args, 'split', use_train_or_test(dataset_name))
            setattr(args, 'split', use_train_or_test(dataset_name))
            generations = load_all_generations(generation_args)
            if args.uncertainty:
                neg_hs = generations['neg_hs']
                pos_hs = generations['pos_hs']
                idk_hs = generations['idk_hs']
            else:
                neg_hs, pos_hs, _ = tuple(generations.values())

            # Make sure the shape is correct
            assert neg_hs.shape == pos_hs.shape
            neg_hs, pos_hs = neg_hs[..., -1], pos_hs[..., -1]  # take the last layer
            if neg_hs.shape[1] == 1:  # T5 may have an extra dimension; if so, get rid of it
                neg_hs = neg_hs.squeeze(1)
                pos_hs = pos_hs.squeeze(1)
            if args.uncertainty:
                idk_hs = idk_hs[..., -1]
                if idk_hs.shape[1] == 1:
                    idk_hs = idk_hs.squeeze(1)
            
            data_to_ccs = [neg_hs, pos_hs, idk_hs] if args.uncertainty else [neg_hs, pos_hs]
            if ccs is None:
                ccs = UncertaintyDetectingCCS(neg_hs, pos_hs, idk_hs, nepochs=args.nepochs//20, ntries=args.ntries, lr=args.lr, batch_size=args.ccs_batch_size, 
                        verbose=args.verbose, device=args.ccs_device, linear=args.linear, weight_decay=args.weight_decay, 
                        var_normalize=args.var_normalize) \
                            if args.uncertainty else \
                    CCS(neg_hs, pos_hs, nepochs=args.nepochs//20, ntries=args.ntries, lr=args.lr, batch_size=args.ccs_batch_size, 
                    verbose=args.verbose, device=args.ccs_device, linear=args.linear, weight_decay=args.weight_decay, 
                    var_normalize=args.var_normalize)
            else:
                ccs.load_new_data(*data_to_ccs)
            print(f"continuing training ccs on {dataset_name}...")
            ccs.train()
        
        ccs.save_eval_probe()

        # eval on temporal dataset. ccs only evaluates raw temporal set, while uccs evaluates both raw and masked (only raw has coverage <100%)
        ccs_acc, uccs_acc_raw, uccs_coverage_raw, uccs_acc_masked = None, None, None, None

        if args.uncertainty:
            for set_temporal_split in ['raw', 'masked']:
                setattr(generation_args, 'dataset_name', 'temporal')
                setattr(args, 'dataset_name', 'temporal')
                setattr(generation_args, 'temporal_split', set_temporal_split)
                setattr(args, 'temporal_split', set_temporal_split)

                generations = load_all_generations(generation_args)
                neg_hs = generations['neg_hs']
                pos_hs = generations['pos_hs']
                idk_hs = generations['idk_hs']
                y = generations['labels']

                neg_hs, pos_hs, idk_hs = neg_hs[..., -1], pos_hs[..., -1], idk_hs[..., -1]  # take the last layer
                if neg_hs.shape[1] == 1:  # T5 may have an extra dimension; if so, get rid of it
                    neg_hs = neg_hs.squeeze(1)
                    pos_hs = pos_hs.squeeze(1)
                    idk_hs = idk_hs.squeeze(1)
                
                eval_inputs = [neg_hs, pos_hs, idk_hs, y, args.temporal_split == 'masked']

                if args.temporal_split == "raw":
                    uccs_acc_raw, uccs_coverage_raw = ccs.get_acc(*eval_inputs)
                elif args.temporal_split == "masked":
                    uccs_acc_masked, uccs_coverage_masked = ccs.get_acc(*eval_inputs)
                    assert uccs_coverage_masked == 1

        else:
            setattr(generation_args, 'dataset_name', 'temporal')
            setattr(args, 'dataset_name', 'temporal')
            setattr(generation_args, 'temporal_split', 'raw')
            setattr(args, 'temporal_split', 'raw')

            generations = load_all_generations(generation_args)
            neg_hs, pos_hs, y = tuple(generations.values())

            neg_hs, pos_hs = neg_hs[..., -1], pos_hs[..., -1]  # take the last layer
            if neg_hs.shape[1] == 1:  # T5 may have an extra dimension; if so, get rid of it
                neg_hs = neg_hs.squeeze(1)
                pos_hs = pos_hs.squeeze(1)
            
            eval_inputs = [neg_hs, pos_hs, y]

            ccs_acc = ccs.get_acc(*eval_inputs)
        
    print(f"Finished temporal experiment on {args.model_name}")
    print(f"CCS accuracy: {ccs_acc}")
    print(f"UCCS accuracy on raw temporal set: {uccs_acc_raw:4f} | Corresponding coverage: {(100.0*uccs_coverage_raw):1f}%")
    print(f"UCCS accuracy on masked temporal set: {uccs_acc_masked:4f}")

    save_path = SAVE_PREFIX + f"results/{'linear' if args.linear else 'MLP'}/temporal.json"
    try:
        with open(save_path, "r") as fin:
            temporal_json =  json.load(fin)
    except FileNotFoundError:
        temporal_json = {}
    temporal_json[args.model_name] = {
        'ccs_acc': ccs_acc,
        'uccs_acc_raw': uccs_acc_raw,
        'uccs_coverage_raw': 100.0 * uccs_coverage_raw,
        'uccs_acc_masked': uccs_acc_masked
    }
    print(temporal_json)
    with open(save_path, 'w') as fout:
        json.dump(temporal_json, fout)


if __name__ == "__main__":
    parser = get_parser()
    generation_args = parser.parse_args()  # we'll use this to load the correct hidden states + labels
    # We'll also add some additional args for evaluation
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
    main(args, generation_args)