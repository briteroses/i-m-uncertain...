from utils.parser import get_parser
from models.lm import load_model
from data.contrast import get_contrast_dataloader
from models.hidden_states import get_all_hidden_states
from utils.save_and_load import save_generations


def main(args):
    # Set up the model and data
    print("Loading model")
    model, tokenizer, model_type = load_model(args.model_name, args.cache_dir, args.parallelize, args.device)

    print("Loading dataloader")
    dataloader = get_contrast_dataloader(args.dataset_name, args.split, tokenizer, args.prompt_idx, use_custom_prompt=args.use_custom_prompt,
                                        batch_size=args.batch_size, num_examples=args.num_examples,
                                        model_name=args.model_name, use_decoder=args.use_decoder, device=args.device,
                                        use_uncertainty=args.uncertainty)

    # Get the hidden states and labels
    print("Generating hidden states")
    hidden_states = get_all_hidden_states(model, dataloader, layer=args.layer, all_layers=args.all_layers, 
                                          token_idx=args.token_idx, model_type=model_type, use_decoder=args.use_decoder, use_uncertainty=args.uncertainty)
    if args.uncertainty:
        neg_hs, pos_hs, idk_hs, y = hidden_states
    else:
        neg_hs, pos_hs, y = hidden_states
    
    # Save the hidden states and labels
    print("Saving hidden states")
    save_generations(neg_hs, args, generation_type="negative_hidden_states", use_uncertainty=args.uncertainty)
    save_generations(pos_hs, args, generation_type="positive_hidden_states", use_uncertainty=args.uncertainty)
    save_generations(y, args, generation_type="labels", use_uncertainty=args.uncertainty)
    if args.uncertainty:
        save_generations(idk_hs, args, generation_type="idk_hidden_states", use_uncertainty=args.uncertainty)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
