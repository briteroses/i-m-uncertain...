import os
import numpy as np

GENERATION_TYPES = [
    "negative_hidden_states",
    "positive_hidden_states",
    "labels"
]
GENERATION_TYPES_PLUS_IDK = GENERATION_TYPES + ["idk_hidden_states",]

EXCLUDE_KEYS = ["save_dir", "cache_dir", "device", "parallelize",
                "split", "prompt_idx", "batch_size", "num_examples",
                "use_decoder", "layer", "all_layers", "token_idx",
                "use_custom_prompt", "uncertainty", "temporal_experiment", "temporal_prompt_idx"]

def get_filename(args, generation_type):
    
    arg_dict = vars(args)
    if args.dataset_name == 'temporal':
        exclude_if_not_temporal = EXCLUDE_KEYS
    else:
        exclude_if_not_temporal = EXCLUDE_KEYS + ["temporal_split"]
    filename = generation_type + "__" + "__".join(['{}_{}'.format(k, v) for k, v in arg_dict.items() if k not in exclude_if_not_temporal]) + ".npy".format(generation_type)

    return filename
    

def save_generations(generation, args, generation_type):
    """
    Input: 
        generation: numpy array (e.g. hidden_states or labels) to save
        args: arguments used to generate the hidden states. This is used for the filename to save to.
        generation_type: hidden state or label, listed in GENERATION_TYPES

    Saves the generations to an appropriate directory.
    """
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    filename = get_filename(args, generation_type)

    # save
    np.save(os.path.join(args.save_dir, filename), generation)


def load_single_generation(args, generation_type="labels"):
    
    # loaded_data = np.load(os.path.join(save_dir, filename))
    # if generation_type == "idk_hidden_states":
    #     print("Shape of idk_hs immediately after loading in load_single_generation:", loaded_data.shape)
    filename = get_filename(args, generation_type)

    return np.load(os.path.join(args.save_dir, filename))


def load_all_generations(args):
    # load all the saved generations: neg_hs, pos_hs, and labels
    generations = []
    for generation_type in (GENERATION_TYPES_PLUS_IDK if args.uncertainty else GENERATION_TYPES):
        generations.append(load_single_generation(args, generation_type=generation_type))

    # if args.uncertainty:
    #     print(f"Shape of idk_hidden_states immediately after loading: {generations[-1].shape}")
    
    zip_labels = ['neg_hs', 'pos_hs', 'labels']
    if args.uncertainty:
        zip_labels += ['idk_hs']
    generations_dict = dict(zip(zip_labels, generations))
    return generations_dict