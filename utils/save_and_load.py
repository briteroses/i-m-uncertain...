import os
import numpy as np

GENERATION_TYPES = [
    "negative_hidden_states",
    "positive_hidden_states",
    "labels"
]
GENERATION_TYPES_PLUS_IDK = GENERATION_TYPES + ["idk_hidden_states",]


def save_generations(generation, args, generation_type, use_uncertainty):
    """
    Input: 
        generation: numpy array (e.g. hidden_states or labels) to save
        args: arguments used to generate the hidden states. This is used for the filename to save to.
        generation_type: hidden state or label, listed in GENERATION_TYPES

    Saves the generations to an appropriate directory.
    """
    save_dir = args.save_dir
    # Check for uncertainty flag and modify save directory
    if use_uncertainty:
        save_dir = "Uncertainty_" + save_dir
    # construct the filename based on the args
    arg_dict = vars(args)
    exclude_keys = ["save_dir", "cache_dir", "device"]
    filename = generation_type + "__" + "__".join(['{}_{}'.format(k, v) for k, v in arg_dict.items() if k not in exclude_keys]) + ".npy".format(generation_type)

    # create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save
    np.save(os.path.join(save_dir, filename), generation)


def load_single_generation(args, generation_type="labels", use_uncertainty=False):
    save_dir = args.save_dir
    # Check for uncertainty flag and modify load directory
    if use_uncertainty:
        save_dir = "Uncertainty_" + save_dir
    # use the same filename as in save_generations
    arg_dict = vars(args)
    exclude_keys = ["save_dir", "cache_dir", "device"]
    filename = generation_type + "__" + "__".join(['{}_{}'.format(k, v) for k, v in arg_dict.items() if k not in exclude_keys]) + ".npy".format(generation_type)
    
    # loaded_data = np.load(os.path.join(save_dir, filename))
    # if generation_type == "idk_hidden_states":
    #     print("Shape of idk_hs immediately after loading in load_single_generation:", loaded_data.shape)
    
    return np.load(os.path.join(save_dir, filename))


def load_all_generations(args):
    # load all the saved generations: neg_hs, pos_hs, and labels
    generations = []
    for generation_type in (GENERATION_TYPES_PLUS_IDK if args.uncertainty else GENERATION_TYPES):
        generations.append(load_single_generation(args, generation_type=generation_type, use_uncertainty=args.uncertainty))

    # if args.uncertainty:
    #     print(f"Shape of idk_hidden_states immediately after loading: {generations[-1].shape}")
    
    zip_labels = ['neg_hs', 'pos_hs', 'labels']
    if args.uncertainty:
        zip_labels += ['idk_hs']
    generations_dict = dict(zip(zip_labels, generations))
    return generations_dict