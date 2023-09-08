import os
import numpy as np

GENERATION_TYPES = [
    "negative_hidden_states",
    "positive_hidden_states",
    "idk_hidden_states",
    "labels"
]


def save_generations(generation, args, generation_type):
    """
    Input: 
        generation: numpy array (e.g. hidden_states or labels) to save
        args: arguments used to generate the hidden states. This is used for the filename to save to.
        generation_type: hidden state or label, listed in GENERATION_TYPES

    Saves the generations to an appropriate directory.
    """
    assert generation_type in GENERATION_TYPES, "invalid generation_type"
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
    assert generation_type in GENERATION_TYPES, "invalid generation_type"
    arg_dict = vars(args)
    exclude_keys = ["save_dir", "cache_dir", "device"]
    filename = generation_type + "__" + "__".join(['{}_{}'.format(k, v) for k, v in arg_dict.items() if k not in exclude_keys]) + ".npy".format(generation_type)
    return np.load(os.path.join(args.save_dir, filename))


def load_all_generations(args):
    # load all the saved generations: neg_hs, pos_hs, and labels
    generations = []
    for generation_type in GENERATION_TYPES:
        generations.append(load_single_generation(args, generation_type=generation_type))

    generations_dict = dict(zip(['neg_hs', 'pos_hs', 'idk_hs', 'labels'], generations))
    return generations_dict