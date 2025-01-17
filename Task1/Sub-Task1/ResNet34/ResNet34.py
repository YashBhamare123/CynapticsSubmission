# Imports
import torch
import opendatasets as od
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from dataset import CustomDataset
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
Image.MAX_IMAGE_PIXELS = 300000000

# Loading the data
dataset_url = "https://www.kaggle.com/competitions/induction-task/data?select=Data"
od.download(dataset_url)
stats = ((0.46873796, 0.42310694, 0.42438492), (0.26308802, 0.23277692, 0.2445178))

# Defining different transforms
train_transforms = v2.Compose([
    v2.Resize((1024,1024)),
    #v2.RandomCrop((128,128)),
    #v2.ColorJitter(0.1, 0.1, 0.1, 0.1 ),
    v2.RandomHorizontalFlip(),
    #v2.RandomRotation(40),
    v2.ToTensor(),
    #v2.Normalize(*stats)
])
val_transforms = v2.Compose([
    v2.Resize((1024,1024)),
    v2.ToTensor(),
    #v2.Normalize(*stats)
])

# Using a Custom Dataset to apply different training and validation transforms to the datasets
data_dir = "./New_Data"
ds = ImageFolder(data_dir)
train_ds, val_ds = random_split(ds, [0.75, 0.25])

train_ds = CustomDataset(train_ds, train_transforms)
val_ds = CustomDataset(val_ds, val_transforms)

batch_size = 4

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)

# Some Helper functions
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('mps')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

# Shifting to GPU
device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)

# Defining the ResNet Block
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, batch):
        return self.block(batch) + batch

# Defining the block where the kernels reduce in size
class SizeReduceBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        self.batch_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0)

    def forward(self, batch):
        projection = self.batch_layer(batch) #Scaling the input image_batch to keep dimensions same while addition
        diff = self.block(batch)
        return diff + projection

factors = [1, 1, 2, 2, 2, 4, 4, 4, 4, 4, 8, 8]


# Defining the ResNet34
class Classifier(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(3, in_channels, kernel_size=7, stride=2, padding=1),
            nn.MaxPool2d(2, 2))
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.ResNet = nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)
        self.full_con = nn.Sequential(
            nn.Flatten(),
            nn.Linear(131072, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 16),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(16),
            nn.Dropout(p=0.3),
            nn.Linear(16, 1)
        )
        self.sigmoid = nn.Sigmoid()

        for i in range(len(factors)):
            if i!= 0 and factors[i-1]<factors[i]:
                self.ResNet.append(SizeReduceBlock(in_channels * factors[i - 1], in_channels * factors[i]))
            self.ResNet.append(
                ResNetBlock(in_channels * factors[i], in_channels * factors[i], kernel_size, stride, padding))

    def forward(self, batch):
        out = self.initial_layer(batch)
        for i in range(len(self.ResNet)):
            out = self.ResNet[i](out)
        out = self.pool(out)
        out = self.full_con(out)
        out = self.sigmoid(out)
        return out


# HyperParameters
in_channels = 64
lr = 1e-5
epochs = 5
model = Classifier(in_channels)
weight_decay = 1e-4
grad_clip = 0.1
model = model.to(device=device)

opt = torch.optim.Adam(model.parameters(), lr, weight_decay= weight_decay)
loss_fn = nn.BCELoss()

@torch.no_grad()
def accuracy(outputs, labels):
    predictions = (outputs>0.5).squeeze(-1)
    correct = (predictions == labels).sum()
    return correct

# Using TensorBoard to visualize results
writer = SummaryWriter("./TB_logs")

# Defining the fit function
def fit(epochs, model, train_loader, val_loader, optimizer=torch.optim.Adam, grad_clip = None, weight_decay = 0):
    opt = optimizer(model.parameters(), lr, weight_decay = weight_decay)
    for epoch in range(epochs):
        history = []
        model.train()
        train_correct = 0
        labels_pred = []
        # Training Phase
        for batch in tqdm(train_loader):
            images, labels = batch
            out = model(images)
            labels_pred.append(out)
            loss = loss_fn(out.squeeze(-1), labels.float())
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            opt.step()
            opt.zero_grad()
            acc = accuracy(out.squeeze(-1).detach(),labels.float())
            train_correct += acc
            history.append(loss.item())
        train_acc = train_correct/float(len(train_ds))
        # Validation/Testing Phase
        model.eval()
        val_correct = 0
        val_history = []
        for batch in val_loader:
            images, labels = batch
            out = model(images)
            loss_val = loss_fn(out.squeeze(-1).detach(), labels.float())
            acc = accuracy(out.squeeze(-1).detach(), labels.float())
            val_correct += acc
            val_history.append(loss_val.item())

        val_acc = val_correct/float(len(val_ds))
        writer.add_scalar("Train_loss",np.mean(history), epoch)
        writer.add_scalar("Val_loss", np.mean(val_history), epoch)
        writer.add_scalar("Train_Acc", train_acc, epoch)
        writer.add_scalar("Val_acc", val_acc, epoch)
        print(f"\nVal_Acc: {val_acc}, Train_Acc: {train_acc}")
        print(f"Val_Loss: {np.mean(val_history)}, Train_LossL {np.mean(history)}")

# Using this so that fit function is not called when a class from this file is imported
if __name__ == "__main__" :
    fit(epochs, model, train_dl, val_dl, grad_clip = grad_clip, weight_decay = weight_decay)
    torch.save(model.state_dict(), 'Real_or_fake_2.pth')



