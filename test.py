import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from models import basemodel
from cub2010 import Cub2010

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_data():
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
            
    testset = Cub2010(root='/root/train-data/CUB-200-2010', is_train=False, transform=transform_test)

    BATCH_SIZE = 16
    NUM_WORKERS = 8

    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    return test_loader, testset

def test_result(model, test_loader, testset):
    resultDict={}
    with torch.no_grad():
        for i, (inputs, imageNames) in enumerate(test_loader, 0):

            inputs = inputs.to(device)

            outputs,_ = model(inputs)
            _, pred = outputs.max(1)

            for idx in range(len(inputs)):
                resultDict[imageNames[idx]] = int(pred[idx].cpu().numpy())
    
    with open('/root/train-data/CUB-200-2010/testing_img_order.txt') as f:
        test_images = [x.strip() for x in f.readlines()]  # all the testing images

    submission = []
    for img in test_images:  # image order is important to your result
        predicted_class = resultDict[img]  # the predicted category
        submission.append([img, testset.labelValue2Label(predicted_class)])

    np.savetxt('answer.txt', submission, fmt='%s')

if __name__ == "__main__":

    test_loader, testset = load_data()

    model_dir = './experiments/baseline/resnet50/weights/weights.095.73.333.pt'
    model = basemodel.resnet50(num_classes=200)
    checkpoint = torch.load(model_dir)
    model.load_state_dict(checkpoint['net'])
    model = model.to(device)
    model.eval() 

    resultDict = test_result(model, test_loader, testset)