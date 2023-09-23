import os
import numpy as np

GENERATION_TYPES = [
    "negative_hidden_states",
    "positive_hidden_states",
    "labels"
]
GENERATION_TYPES_PLUS_IDK = GENERATION_TYPES + ["idk_hidden_states",]


def save_generations(generation, args, generation_type):
    """
    Input: 
        generation: numpy array (e.g. hidden_states or labels) to save
        args: arguments used to generate the hidden states. This is used for the filename to save to.
        generation_type: hidden state or label, listed in GENERATION_TYPES

    Saves the generations to an appropriate directory.
    """
    # construct the filename based on the args
    arg_dict = vars(args)
    exclude_keys = ["save_dir", "cache_dir", "device"]
    filename = generation_type + "__" + "__".join(['{}_{}'.format(k, v) for k, v in arg_dict.items() if k not in exclude_keys]) + ".npy".format(generation_type)

    # create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # save
    np.save(os.path.join(args.save_dir, filename), generation)


def load_single_generation(args, generation_type="labels"):
    # use the same filename as in save_generations
    arg_dict = vars(args)
    exclude_keys = ["save_dir", "cache_dir", "device"]
    filename = generation_type + "__" + "__".join(['{}_{}'.format(k, v) for k, v in arg_dict.items() if k not in exclude_keys]) + ".npy".format(generation_type)
    return np.load(os.path.join(args.save_dir, filename))


def load_all_generations(args):
    # load all the saved generations: neg_hs, pos_hs, and labels
    generations = []
    for generation_type in (GENERATION_TYPES_PLUS_IDK if args.uncertainty else GENERATION_TYPES):
        generations.append(load_single_generation(args, generation_type=generation_type))

    zip_labels = ['neg_hs', 'pos_hs', 'labels']
    if args.uncertainty:
        zip_labels += ['idk_hs']
    generations_dict = dict(zip(zip_labels, generations))
    return generations_dict