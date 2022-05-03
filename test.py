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
model.load_state_dict(torch.load('model_weights.pth', map_location = device))
model.eval()
evaluate(model, test_loader, device=device)

# use to pick one image from the test_split
#img, _ = test_split[42]
img = Image.open("sample_images/007.jpg")
img = transforms.functional.to_tensor(img)
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])

boxes = prediction[0]['boxes']
labels = prediction[0]['labels']
img = img.swapaxes(0,1)
img = img.swapaxes(1,2)

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
colors = ['w', 'r', 'b', 'g', 'c', 'm', 'y', 'g', 'c', 'm', 'k']
i = 0
fig, ax = plt.subplots(1)
plt.imshow(img)
for box in boxes:
  w = box[2] - box[0]
  h = box[3] - box[1]
  x = box[0]
  y = box[1]
  item_label = labels[i]
  color = colors[item_label]    #red = water, blue = coke, juice = green
  rect = Rectangle((x, y),
                         w,
                         h,
                         linewidth=2,
                         edgecolor=color,
                         facecolor='none')
  if (prediction[0]['scores'][i] > 0.80):   #only add bounding box when score is greater than 0.8
    ax.add_patch(rect)
    if (item_label == 1):
      ax.text(x+5, y-5, "water")
    if (item_label == 2):
      ax.text(x+5, y-5, "Coke")
    if (item_label == 3):
      ax.text(x+5, y-5, "juice")
  i += 1
plt.show()