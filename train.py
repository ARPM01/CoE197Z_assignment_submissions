import torch
import numpy as np
import wandb
import torchvision
import os

import label_utils
import utils
import transforms as T

from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from engine import train_one_epoch, evaluate
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

import gdown

url = 'https://drive.google.com/uc?id=1AdMbVK110IKLG7wJKhga2N2fitV1bVPA'
output = 'drinks.tar.gz'
gdown.download(url, output, quiet=False)

import tarfile
file = tarfile.open('drinks.tar.gz')
print("Extracting tar file...")
file.extractall('')
file.close()

config = {
    "num_workers": 0,
    "pin_memory": False,
    "batch_size": 8,
    "dataset": "drinks",
    "train_split": "drinks/labels_train.csv",
    "test_split": "drinks/labels_test.csv",}

test_dict, test_classes = label_utils.build_label_dictionary(
    config['test_split'])
train_dict, train_classes = label_utils.build_label_dictionary(
    config['train_split'])

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, dictionary, transform=None):
        self.dictionary = dictionary
        self.root = root
        self.transform = transform
        self.imgs = list(os.listdir(os.path.join(root, "drinks")))
    def __len__(self):
        return len(self.dictionary)

    def __getitem__(self, idx):

        key = list(self.dictionary.keys())[idx]
        img = Image.open(key)
        
        num_objs = len(self.dictionary[key])

        b_temp = self.dictionary[key]
        b_temp2 = [[j[i] for i in range(4)] for j in b_temp]  #remove class from value
        boxes = [[i[0], i[2], i[1], i[3]] for i in b_temp2] #in [xmin, ymin, xmax, ymax]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor([i[4] for i in b_temp], dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target['iscrowd'] = iscrowd
        #target["masks"] = torch.zeros(3,480,640)

        if self.transform:
            img = self.transform(img)

        return img, target

train_split = ImageDataset("", train_dict, transforms.ToTensor())
test_split = ImageDataset("", test_dict, transforms.ToTensor())

print("Train split len:", len(train_split), train_split)
print("Test split len:", len(test_split), test_split)

train_loader = DataLoader(train_split,
                          batch_size=config['batch_size'],
                          shuffle=False,
                          num_workers=config['num_workers'],
                          pin_memory=config['pin_memory'],
                          collate_fn=utils.collate_fn)

test_loader = DataLoader(test_split,
                         batch_size=config['batch_size'],
                         shuffle=False,
                         num_workers=config['num_workers'],
                         pin_memory=config['pin_memory'],
                         collate_fn=utils.collate_fn)

def create_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = create_model(4)
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
num_epochs = 4

for epoch in range(num_epochs):
  train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=25)
  lr_scheduler.step()
  evaluate(model, test_loader, device=device)

print("Saving the trained model as model_weights.pth")
torch.save(model.state_dict(), 'model_weights.pth')