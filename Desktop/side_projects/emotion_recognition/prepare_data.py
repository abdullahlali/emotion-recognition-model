import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

save_dir = '/Users/abdullahali/Desktop/side_projects/emotion_recognition'
data_dir = '/Users/abdullahali/Desktop/side_projects/emotion_recognition/facial_expression'
train_dir = data_dir + '/train'
test_dir = data_dir + '/test'

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5215, 0.5215, 0.5215], [0.0518, 0.0518, 0.0518])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5215, 0.5215, 0.5215], [0.0518, 0.0518, 0.0518])
    ]),
}



# Save preprocessed datasets
train_dataset = datasets.ImageFolder(train_dir, data_transforms['train'])
torch.save(train_dataset, os.path.join(save_dir, 'train_dataset.pth'))

test_dataset = datasets.ImageFolder(test_dir, data_transforms['test'])
torch.save(test_dataset, os.path.join(save_dir, 'train_dataset.pth'))

print("Datasets saved as binary files!")
