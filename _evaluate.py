import sys

from sklearn.linear_model import LogisticRegression

from utils.parser import get_parser
from utils.save_and_load import load_all_generations
from probes.CCS import CCS
from probes.uncertainty import UncertaintyDetectingCCS


from sklearn.metrics import auc
import matplotlib.pyplot as plt
import os 
import numpy as np

def main(args, generation_args):
    # load hidden states and labels
    generations = load_all_generations(generation_args)
    if args.uncertainty:
        neg_hs, pos_hs, idk_hs, y = tuple(generations.values())
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
        assert pos_hs.shape == idk_hs.shape
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

    if args.uncertainty:
        uccs = UncertaintyDetectingCCS(neg_hs_train, pos_hs_train, idk_hs_train, nepochs=args.nepochs, ntries=args.ntries, lr=args.lr, batch_size=args.ccs_batch_size, 
                    verbose=args.verbose, device=args.ccs_device, linear=args.linear, weight_decay=args.weight_decay, 
                    var_normalize=args.var_normalize)
        uccs.repeated_train()
        uccs_acc, uccs_coverage = uccs.get_acc(neg_hs_test, pos_hs_test, idk_hs_test, y_test)
        print(f"UCCS accuracy: {uccs_acc} | UCCS coverage: {(100.0*uccs_coverage):1f}%")
    else:
        ccs = current_CCS(neg_hs_train, pos_hs_train, nepochs=args.nepochs, ntries=args.ntries, lr=args.lr, batch_size=args.ccs_batch_size, 
                        verbose=args.verbose, device=args.ccs_device, linear=args.linear, weight_decay=args.weight_decay, 
                        var_normalize=args.var_normalize)
        ccs.repeated_train()
        ccs_acc = ccs.get_acc(neg_hs_test, pos_hs_test, y_test)
        print("CCS accuracy: {}".format(ccs_acc))

    if args.roc:
        scores = ccs.get_scores(neg_hs_test, pos_hs_test)
        fpr, tpr, roc_auc = ccs.compute_roc(scores, y_test)

        if fpr is not None and tpr is not None and roc_auc is not None:  # If it's not binary, we can't compute the ROC curve
            if not os.path.exists("ROC_curves"):
                os.makedirs("ROC_curves")
                
            save_path = f"ROC_curves/{args.model_name}_{args.dataset_name}.png"
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

    if args.save_confidence_scores:
        # Save confidence scores and true labels for ROC sliding window sweep
        save_dir = f"confidence_scores_and_labels/{args.model_name}/{args.dataset_name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, "confidence_scores.npy"), scores)
        np.save(os.path.join(save_dir, "true_labels.npy"), y_test)



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
