import os
import argparse
import torch
from torchvision import datasets, transforms
import utils

available_architectures = { 'densenet121', 'densenet161', 'densenet201', 'resnet18', 'resnet34', 'resnet50', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn' }

def save(arch, learning_rate, hidden_layers, epochs, save_path, model, optimizer):
    ''' 
    Save the checkpoint
    '''
    torch.save({
        'arch': arch,
        'hidden_layers': hidden_layers,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'optimizer' : optimizer.state_dict(),
        'state_dict': model.state_dict(),
        'class_to_idx' : model.class_to_idx
    }, save_path)


def get_command_line_args():
    '''
    Retrieve input arguments
    '''
    parser = argparse.ArgumentParser()
    
    # handle required arguments
    parser.add_argument('data_dir', help = 'Directory with flower images')
    
    # handle optional arguments
    parser.add_argument('--gpu', dest = 'gpu', action = 'store_true', help = 'Enable GPU support')
    parser.set_defaults(gpu = False)
    parser.add_argument('--save_dir', help='Directory for checkpoint saving')
    parser.add_argument('--arch', dest = 'arch', default = 'densenet161', action = 'store', choices = available_architectures, help = 'Network architecture')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning rate of the model')
    parser.add_argument('--hidden_units', type = int, default = 256, help = 'Amount of hidden units')
    parser.add_argument('--epochs', type = int, default = 2, help = 'Number of epochs')
    
    return parser.parse_args()

def main():
    # get user arguments
    args = get_command_line_args()
    
    # determine if gpu support is enabled
    has_gpu_support = torch.cuda.is_available() and args.gpu
    print("Data directory:", args.data_dir)
    
    if has_gpu_support:
        print("GPU support enabled. Training on GPU.")
    else:
        print("GPU support disabled. Training on CPU.")

    print("Selected architecture:", args.arch)
    
    if args.save_dir:
        print("Directory for checkpoint saving:", args.save_dir)

    print("Learning rate:", args.learning_rate)
    print("Hidden layers:", args.hidden_units)
    print("Epochs:", args.epochs)
    
    # Get data loaders
    dataloaders, class_to_idx = utils.get_loaders(args.data_dir)
    for key, value in dataloaders.items():
        print("{} data loader retrieved".format(key))
    
    # Build the model
    model, optimizer, criterion = utils.build_model(args.arch, args.hidden_units, args.learning_rate)
    model.class_to_idx = class_to_idx
    
    # check for gpu support
    if has_gpu_support:
        print("Moving tensors to GPU...")
        # move tensors to gpu
        model.cuda()
        criterion.cuda()    
    
    # Train the model
    utils.train_model(model, args.epochs, criterion, optimizer, dataloaders, has_gpu_support)
    
    # Save the checkpoint
    if args.save_dir:
        has_save_dir = os.path.exists(args.save_dir);
        
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        save_path = args.save_dir + '/' + args.arch + '_checkpoint.pth'
    else:
        save_path = args.arch + '_checkpoint.pth'

    print("About to save checkpoint to {}...".format(save_path))

    # save checkpoint
    save(args.arch, args.learning_rate, args.hidden_units, args.epochs, save_path, model, optimizer)
    print("Checkpoint has been saved")

    # get test loss and accuracy
    accuracy_rate, test_loss_rate = utils.validate(model, criterion, dataloaders['test'], has_gpu_support)
    print("Test Accuracy: {:.2f}".format(accuracy_rate))
    print("Test Loss: {:.2f}".format(test_loss_rate))
  
if __name__ == "__main__":
    main()