from utils.parser import get_parser
from models.lm import load_model
from data.contrast import get_contrast_dataloader
from models.hidden_states import get_all_hidden_states
from utils.save_and_load import save_generations

from data.registry import DATASET_LABEL_REGISTRY, use_train_or_test
from data.temporal import temporal_dataloader

def generate_and_save(args, model, tokenizer, model_type):

    print(f"Loading dataloader for {args.dataset_name}")
    if args.dataset_name == "temporal":
        dataloader = temporal_dataloader(args.dataset_name, args.temporal_split, tokenizer, args.temporal_prompt_idx,
                                        model_name=args.model_name, use_decoder=args.use_decoder, device=args.device)
    else:
        dataloader = get_contrast_dataloader(args.dataset_name, args.split, tokenizer, args.prompt_idx, use_custom_prompt=args.use_custom_prompt,
                                            batch_size=args.batch_size, num_examples=args.num_examples,
                                            model_name=args.model_name, use_decoder=args.use_decoder, device=args.device,
                                            use_uncertainty=args.uncertainty)

    # Get the hidden states and labels
    print("Generating hidden states")
    hidden_states = get_all_hidden_states(model, dataloader, layer=args.layer, all_layers=args.all_layers, 
                                          token_idx=args.token_idx, model_type=model_type, use_decoder=args.use_decoder, use_uncertainty=args.uncertainty)
    if args.uncertainty or args.dataset_name == "temporal":
        neg_hs, pos_hs, idk_hs, y = hidden_states
    else:
        neg_hs, pos_hs, y = hidden_states
    
    # Save the hidden states and labels
    print("Saving hidden states")
    save_generations(neg_hs, args, generation_type="negative_hidden_states")
    save_generations(pos_hs, args, generation_type="positive_hidden_states")
    save_generations(y, args, generation_type="labels")
    if args.uncertainty or args.dataset_name == "temporal":
        save_generations(idk_hs, args, generation_type="idk_hidden_states")


def main(args):
    # Set up the model and data
    print("Loading model")
    model, tokenizer, model_type = load_model(args.model_name, args.cache_dir, args.parallelize, args.device)

    if args.temporal_experiment:
        for raw_or_masked in ['raw', 'masked']:
            setattr(args, 'dataset_name', 'temporal')
            setattr(args, 'temporal_split', raw_or_masked)
            generate_and_save(args, model, tokenizer, model_type)
        for dataset_name in DATASET_LABEL_REGISTRY.keys():
            setattr(args, 'dataset_name', dataset_name)
            setattr(args, 'split', use_train_or_test(dataset_name))
            generate_and_save(args, model, tokenizer, model_type)
    else:
        generate_and_save(args, model, tokenizer, model_type)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
