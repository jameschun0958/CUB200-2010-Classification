import os
import time
import numpy as np
import math
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

import pandas as pd
from math import cos, pi

from utils import progress_bar

import loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Trainer(object):
    def __init__(self,
                 model,
                 params):
        self.model = model
        self.params = params
        self.model_name = params.model_name
        self.model_dir = params.model_dir
        self.best_acc = 0.0
        self.best_epoch = 0
        self.model = self.model.to(device)

    def train(self, epoch, trainloader):
        self.model.train()
        lr = self.params.learning_rate

        xent = loss.CrossEntropyLabelSmooth(num_classes=self.params.num_classes) # label smooth
        triplet = loss.TripletLoss(margin=0.3).cuda()

        train_loss, correct, total = 0, 0, 0

        if not hasattr(self, 'optimizer'):
            self.optimizer = optim.SGD(self.model.parameters(), lr, momentum=0.9, weight_decay=1e-4)

        for idx, (inputs, labels) in enumerate(trainloader):
            self.adjust_learning_rate(epoch, idx, len(trainloader))

            inputs, labels = inputs.to(device), labels.to(device)

            outputs, feat = self.model(inputs)

            total_loss = xent(outputs, labels) + triplet(feat, labels)[0]

            # clear previous gradients, compute gradients of all variables wrt loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            train_loss += total_loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (labels == predicted).sum().item()

            progress_bar(
                    idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                    (train_loss / (idx + 1), 100. * correct / total, correct,
                    total))

    def evaluate(self, epoch, testloader):
        self.model.eval()

        test_loss = 0
        correct = 0
        total = 0

        xent = loss.CrossEntropyLabelSmooth(num_classes=self.params.num_classes) # label smooth

        with torch.no_grad():
            for idx, (test_x, test_y) in enumerate(testloader):
                test_x, test_y = test_x.to(device), test_y.to(device)

                outputs,_ = self.model(test_x) # return last value

                test_loss += (xent(outputs, test_y)).item()

                _, predicted = outputs.max(1)
                
                total += test_y.size(0)
                correct += (predicted == test_y).sum().item()

                progress_bar(
                    idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                    (test_loss / (idx + 1), 100. * correct / total, correct, total))

        acc = 100.0 * correct / total

        if acc > self.best_acc:
            self.save_model(self.model, self.model_name, self.model_dir, acc, epoch)

    def adjust_learning_rate(self, epoch, iteration, num_iter):
        optimizer = self.optimizer
        init_lr = self.params.learning_rate
        total_epoch = self.params.num_epochs
        warmup = 5
        lr_decay = 'cos'

        warmup_epoch = 5 if warmup else 0
        warmup_iter = warmup_epoch * num_iter
        current_iter = iteration + epoch * num_iter
        max_iter = total_epoch * num_iter

        if lr_decay == 'step':
            lr = init_lr * (0.1 ** ((current_iter - warmup_iter) / (max_iter - warmup_iter)))
        elif lr_decay == 'cos':
            lr = init_lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
        elif lr_decay == 'linear':
            lr = init_lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
        elif lr_decay == 'schedule': #decrease learning rate at these epochs.
            schedule = [60, 120, 180]
            count = sum([1 for s in schedule if s <= epoch])
            lr = init_lr * pow(0.1, count)
        else:
            raise ValueError('Unknown lr mode {}'.format(lr_decay))

        if epoch < warmup_epoch:
            lr = init_lr * current_iter / warmup_iter

        if iteration == 0:
            print('Learning rate:{}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    def save_model(self, model, model_name, model_dir, acc, epoch):
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        del_name = os.path.join('experiments', model_dir, model_name, 'weights',
                                     'weights.%03d.%.03f.pt' % (self.best_epoch, self.best_acc))
        ## remove previous saved model 
        if os.path.exists(del_name):
            os.remove(del_name)

        save_name = os.path.join('experiments', model_dir, model_name, 'weights',
                                     'weights.%03d.%.03f.pt' % (epoch, acc))

        if not os.path.exists(os.path.dirname(save_name)):
            os.makedirs(os.path.dirname(save_name))

        torch.save(state, save_name)
        print("\nSaved state at %.03f%% accuracy. Prev accuracy: %.03f%%" %
              (acc, self.best_acc))
        self.best_acc = acc
        self.best_epoch = epoch

    def load_model(self, path=None):
        """
        Load previously saved model. THis doesn't check for precesion type.
        """
        if path is not None:
            checkpoint_name = path
        else:
            checkpoint_name = os.path.join(
            'weights', self.model_name ,
            'weights.%03d.%.03f.pt' % (self.best_epoch, self.best_acc))

        if not os.path.exists(checkpoint_name):
            print("Best model not found")
            return
        checkpoint = torch.load(checkpoint_name)
        self.model.load_state_dict(checkpoint['net'])
        self.best_acc = checkpoint['acc']
        self.best_epoch = checkpoint['epoch']
        print("Loaded Model with accuracy: %.3f%%, from epoch: %d" %
              (checkpoint['acc'], checkpoint['epoch'] + 1))
                
    def train_and_evaluate(self, train_dataloader, test_dataloader, restore_file=None):
        # reload weights from restore_file if specified
        if restore_file is not None:
            self.load_model(restore_file)

        total_epoch = self.params.num_epochs
        lr = self.params.learning_rate

        for epoch in range(total_epoch):
            print('\nEpoch: %d' % (epoch + 1))
            self.train(epoch, train_dataloader)
            self.evaluate(epoch, test_dataloader)