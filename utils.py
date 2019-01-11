import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from collections import OrderedDict
from torchvision import datasets, transforms, models
from torchvision import models
from torch.autograd import Variable
from PIL import Image

mean_normalization = [0.485, 0.456, 0.406]
std_deviations = [0.229, 0.224, 0.225]

def get_loaders(img_dir):
    """
    Return dataloaders for training, validation and teting datasets.
    """
    train_dir = img_dir + '/train/'
    test_dir = img_dir + '/test/'
    valid_dir = img_dir + '/valid/'
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean_normalization, std_deviations)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean_normalization, std_deviations)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean_normalization, std_deviations)
        ])
    }
    
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform = data_transforms['train']),
        'test': datasets.ImageFolder(test_dir, transform = data_transforms['test']),
        'valid': datasets.ImageFolder(valid_dir, transform = data_transforms['valid'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size = 64, shuffle = True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size = 64),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size = 64, shuffle = True)
    }
    
    class_to_idx = image_datasets['test'].class_to_idx
    return dataloaders, class_to_idx

def load_pretrained_model(arch):
    '''
    Load pretrained model
    '''
    if arch == 'densenet121':
        model = models.densenet121(pretrained = True)
        inputsize = model.classifier.in_features
    elif arch == 'densenet161':
        model = models.densenet161(pretrained = True)
        inputsize = model.classifier.in_features
    elif arch == 'densenet201':
        model = models.densenet201(pretrained = True)
        inputsize = model.classifier.in_features
    elif arch == 'resnet18':
        model = models.resnet18(pretrained = True)
        inputsize = model.fc.in_features
    elif arch == 'resnet34':
        model = models.resnet34(pretrained = True)
        inputsize = model.fc.in_features
    elif arch == 'resnet50':
        model = models.resnet50(pretrained = True)
        inputsize = model.fc.in_features
    elif arch == 'vgg13_bn':
        model = models.vgg13_bn(pretrained = True)
        inputsize = model.classifier[0].in_features
    elif arch == 'vgg16_bn':
        model = models.vgg16_bn(pretrained = True)
        inputsize = model.classifier[0].in_features
    elif arch == 'vgg19_bn':
        model = models.vgg19_bn(pretrained = True)
        inputsize = model.classifier[0].in_features
        
    return model, inputsize


def get_model(arch, hidden_units):
    '''
    Load pretrained model
    '''
    model, input_size = load_pretrained_model(arch)

    for param in model.parameters():
        param.requires_grad = False

    output_size = 102
    fc1 = nn.Linear(input_size, hidden_units)
    relu = nn.ReLU()
    fc2 = nn.Linear(hidden_units, output_size)
    output = nn.LogSoftmax(dim = 1)

    classifier = nn.Sequential(OrderedDict([
        ('fc1', fc1),
        ('relu', relu),
        ('fc2', fc2),
        ('output', output)
    ]))

    if 'vgg' in arch or 'densenet' in arch:
        model.classifier = classifier
    elif 'resnet' in arch:
        model.fc = classifier

    return model

def build_model(arch, hidden_units, learning_rate):
    '''
    Build model
    '''
    model = get_model(arch, hidden_units)
    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr = learning_rate)
    optimizer.zero_grad()
    criterion = nn.NLLLoss()
    
    return model, optimizer, criterion
    
def train_model(model, epochs, criterion, optimizer, dataloaders, has_gpu_support = False, print_every = 60):
    '''
    Train model
    '''
    model.train()
    steps = 0

    for epoch in range(epochs):
        running_loss = 0

        for data in iter(dataloaders['train']):
            inputs, labels = data
            steps += 1
            
            inputs = inputs.float().cuda() if has_gpu_support else inputs
            labels = labels.long().cuda() if has_gpu_support else labels

            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]

            if steps % print_every == 0:
                accuracy, valudation_loss = validate(model, criterion, dataloaders['valid'], has_gpu_support)
                training_loss = running_loss / print_every
                model.train()
                
                print(
                    "Epoch: {}/{} ".format(epoch + 1, epochs),
                    "Train Loss: {:.2f} ".format(training_loss),
                    "Validation Loss: {:.2f} ".format(valudation_loss),
                    "Validation Accuracy: {:.2f}".format(accuracy)
                )
                
                running_loss = 0

def validate(model, criterion, data_loader, has_gpu_support):
    ''' 
    Validate model
    '''
    model.eval()
    accuracy = 0
    test_loss = 0
    
    for data in iter(data_loader):
        inputs, labels = data
        
        inputs = inputs.float().cuda() if has_gpu_support else inputs
        labels = labels.long().cuda() if has_gpu_support else labels
        
        inputs = Variable(inputs)
        labels = Variable(labels)
            
        output = model.forward(inputs)
        test_loss += criterion(output, labels).data[0]
        ps = torch.exp(output).data

        measure = labels.data == ps.max(1)[1]
        accuracy += measure.type_as(torch.FloatTensor()).mean()

    test_loss_rate = test_loss / len(data_loader)
    accuracy_rate = accuracy / len(data_loader)

    return accuracy_rate, test_loss_rate


def process_image(image, size = 224):
    ''' Normalizes a PIL image
    '''
    w, h = image.size
    
    if h > w:
        height = int(max(h * size / w, 1))
        width = int(size)
    else:
        width = int(max(w * size / h, 1))
        height = int(size)
        
    resized_image = image.resize((width, height))
        
    x0 = (width - size) / 2
    y0 = (height - size) / 2
    x1 = x0 + size
    y1 = y0 + size
    
    color_channels = 255.0;
    cropped_image = image.crop((x0, y0, x1, y1))
    np_image = np.array(cropped_image) / color_channels
    
    mean = np.array(mean_normalization)
    std = np.array(std_deviations)
    np_image_array = (np_image - mean) / std
    
    return np_image.transpose((2, 0, 1))


def load_checkpoint(checkpoint):
    '''
    Load the checkpoint file and build model
    '''
    state = torch.load(checkpoint)
    learning_rate = float(state['learning_rate'])
    hidden_layers = int(state['hidden_layers'])
    
    model, optimizer, criterion = build_model(state['arch'], hidden_layers, learning_rate)

    model.class_to_idx = state['class_to_idx']
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    
    return model


def predict(input_path, model, has_gpu_support, results_to_show, topk):
    ''' Predict the class
    '''
    model.eval()
    image = Image.open(input_path)
    np_array = process_image(image)
    tensor = torch.from_numpy(np_array)
        
    var_inputs = tensor.float().cuda() if has_gpu_support else tensor
    var_inputs = Variable(var_inputs)

    var_inputs = var_inputs.unsqueeze(0)
    output = model.forward(var_inputs)
    ps = torch.exp(output).data.topk(topk)
    
    probs = ps[0].cpu() if has_gpu_support else ps[0]
    classes = ps[1].cpu() if has_gpu_support else ps[1]
    inverted_class_to_idx = { model.class_to_idx[k]: k for k in model.class_to_idx }
    
    labels = classes.numpy()[0]
    classes_list = [inverted_class_to_idx[label] for label in labels]
        
    return probs.numpy()[0], classes_list

