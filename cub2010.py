import numpy as np
from PIL import Image
import glob
import os
from torch.utils.data.dataset import Dataset


class Cub2010(Dataset):
    
    def __init__(self, root="~/train-data/CUB-200-2010/", is_train=True, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        self.getImage()
        self.getLabel()
        self.data_len = len(self.image_list)

    def getImage(self):
        if self.is_train:
            dir_path = os.path.join(self.root, "training_images")
        else:
            dir_path = os.path.join(self.root, "testing_images")

        path = dir_path +  "/*"
        images_path = glob.glob(path)
        self.image_list =  []
        self.image_name_list = []
        for image_path in images_path:
            img = Image.open(image_path).convert('RGB')
            self.image_list.append(img)
            self.image_name_list.append(image_path.split('/')[-1])

    def getLabelByIndex(self,index):
        return self.labelDict[self.image_name_list[index]]
        
    def getLabel(self):
        label_path = os.path.join(self.root, "training_labels.txt")
        self.labelDict = {}
        self.labelValue2labelName = {}
        
        with open(label_path,'r') as f:
            for line in f.readlines():
                labelValue = int(line.split()[1].split('.')[0])-1 #class:0~199
                self.labelDict[line.split()[0]] = labelValue
                label = line.split()[1]
                self.labelValue2labelName[labelValue] = label

        # print("labelDict",self.labelDict)
        # print("labelvalue2labelNmae",self.labelValue2labelName)
    def labelValue2Label(self,labelValue):
        return self.labelValue2labelName[labelValue]
        
    def __getitem__(self, index):

        img = self.image_list[index]

        if self.transform is not None:
            img = self.transform(img)
        
        if self.is_train:
            label = self.getLabelByIndex(index) 
        else:
            label = self.image_name_list[index]
        #print("index",index,"img path",single_image_path,"label",label)
        # label = torch.randn(1)
        return (img, label)

    def __len__(self):
        return self.data_len

if __name__ == '__main__':
   data = Cub2010(root='/home/chun/train-data/CUB-200-2010/')
#    print(data.labelDict)