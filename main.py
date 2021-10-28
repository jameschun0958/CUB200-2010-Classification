import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, datasets
from torchinfo import summary

from train import Trainer
from models import basemodel

from cub2010 import Cub2010

import os
import argparse
import utils
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--model_dir', default='experiments/test/',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir \
                    containing weights to reload before training")  # 'best' or 'train'

# Load the parameters from json file
args = parser.parse_args()
json_path = os.path.join(args.model_dir, 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = utils.Params(json_path)

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
def load_data():
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
            
    trainset = Cub2010(root='/root/train-data/CUB-200-2010', is_train=True, transform=transform_train)

    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(trainset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.1 * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    BATCH_SIZE = params.batch_size
    NUM_WORKERS = 8

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, sampler=train_sampler)
    val_loader = DataLoader(trainset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, sampler=valid_sampler)

    return train_loader, val_loader


print("Laoding dataset....")
train_loader, val_loader  = load_data()

# Train the model
print("Experiment - model version: {}".format(params.model_version))

model = basemodel.resnet50(num_classes=params.num_classes)
# print(model)
summary(model.cuda(), (1, 3, 256, 256))
model.to('cpu')
trainer = Trainer(model, params)
trainer.train_and_evaluate(train_loader, val_loader, restore_file=args.restore_file)