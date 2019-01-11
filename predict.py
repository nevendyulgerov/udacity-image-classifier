import torch
import argparse
import utils
import json

def get_command_line_args():
    parser = argparse.ArgumentParser()
    
    # handle required arguments
    parser.add_argument('input', help = 'Image')
    parser.add_argument('checkpoint', help = 'Save checkpoint')
        
    # handle optional arguments
    parser.add_argument('--top_k', type = int, help = 'Return the top K highest probability classes')
    parser.set_defaults(top_k = 1)
    parser.add_argument('--category_names', help = 'Category names')
    parser.add_argument('--gpu', dest = 'gpu', action = 'store_true', help = 'Enable GPU support')
    parser.set_defaults(gpu = False)

    return parser.parse_args()


def main():
    # get input args
    args = get_command_line_args()
    has_gpu_support = torch.cuda.is_available() and args.gpu
    
    print("Input file:", args.input)
    print("Checkpoint file:", args.checkpoint)
    
    if args.top_k:
        print("Returning {} most likely classes".format(args.top_k))

    if args.category_names:
        print("Category names file: {}".format(args.category_names))

    # notify if gpu support is enabled
    if has_gpu_support:
        print("GPU support enabled. Predicting on GPU.")
    else:
        print("GPU support disabled. Predicting on CPU.")
    
    # load model from checkpoint
    model = utils.load_checkpoint(args.checkpoint)
    print("Checkpoint has been loaded.")
    
    # check for gpu support
    if has_gpu_support:
        # move tensors to gpu
        model.cuda()
    
    # load categories
    if args.category_names:
        with open(args.category_names, 'r') as f:
            categories = json.load(f)
            print("Category names loaded")
    
    results_to_show = args.top_k if args.top_k else 1
    
    # predict
    print("Processing image...")
    probabilities, classes = utils.predict(args.input, model, has_gpu_support, results_to_show, args.top_k)
    
    # print results
    if results_to_show > 1:
        print("Top {} Classes for '{}':".format(len(classes), args.input))

        if args.category_names:
            print("{:<30} {}".format("Flower", "Probability"))
            print()
        else:
            print("{:<10} {}".format("Class", "Probability"))
            print()

        for i in range(len(classes)):
            if args.category_names:
                print("{:<30} {:.2f}".format(categories[classes[i]], probabilities[i]))
            else:
                print("{:<10} {:.2f}".format(classes[i], probabilities[i]))
    else:
        highest_prob_class = categories[classes[0]] if args.category_names else classes[0]
        print("Highest probability class: '{}', Probability: {:.2f}".format(highest_prob_class, probabilities[0]))
        
    
if __name__ == "__main__":
    main()