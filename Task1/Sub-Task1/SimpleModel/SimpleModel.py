# Imports
import torch
import torch.nn as nn
import opendatasets as od
from torchvision.datasets import ImageFolder
from dataset import CustomDataset
from torchvision.transforms import v2
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import torch.nn.functional as F



# Loading the Data
download_url = "https://www.kaggle.com/competitions/induction-task/data"

od.download(download_url)
dataset = ImageFolder("./New_Data", transform = v2.ToTensor())
stats = ((0.46873796, 0.42310694, 0.42438492), (0.26308802, 0.23277692, 0.2445178))
train_ts = v2.Compose([
    v2.Resize((256,256)),
    v2.RandomCrop((256,256), padding =50, padding_mode = 'reflect'),
    v2.ColorJitter(0.5, 0.5, 0.5, 0.5),
    v2.RandomHorizontalFlip(),
    v2.ToTensor(),
    v2.Normalize(*stats)
]) 

val_ts = v2.Compose([
    v2.Resize((256,256)),
    v2.ToTensor(),
    v2.Normalize(*stats),
])



train_ds, val_ds = random_split(dataset, [0.75, 0.25])
train_ds = CustomDataset(train_ds, train_ts)
val_ds = CustomDataset(val_ds, val_ts)
batch_size = 8
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_dl =  DataLoader(val_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)


# Using GPU


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


# In[16]:


device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)


# Model
class SimpleModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, in_channels, kernel_size = 3, stride = 1, padding =1),
            nn.LeakyReLU(0.2),
            #nn.BatchNorm2d(in_channels),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels, in_channels*2, kernel_size = 3, stride = 1, padding =1),
            nn.LeakyReLU(0.2),
            #nn.BatchNorm2d(in_channels*2),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels*2, in_channels*4, kernel_size = 3, stride = 1, padding =1),
            nn.LeakyReLU(0.2),
            #nn.BatchNorm2d(in_channels*2),
            nn.MaxPool2d(2,2),
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(131072, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,10),
            nn.LeakyReLU(0.2),
            nn.Linear(10,1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.cnn(x)
        x = self.linear(x)
        return x
        


# Training Loop
model = SimpleModel(32)
model = model.to(device)
epochs = 20
lr = 1e-5
opt = torch.optim.Adam(model.parameters(),lr)


@torch.no_grad()
def accuracy_count(outputs, labels):
    predictions = (outputs>0.5).squeeze(-1)
    correct = (predictions == labels).sum()
    return correct

def fit(epochs, lr, model, train_dl, val_dl, optimizer):
    opt = optimizer(model.parameters(), lr)
    for epoch in range(epochs):
        model.train()
        train_correct = 0
        val_correct = 0
        train_loss =[]
        for batch in tqdm(train_dl):
            images, labels = batch
            out = model(images)
            loss = F.binary_cross_entropy(out.squeeze(-1), labels.float())
            loss.backward()
            opt.step()
            opt.zero_grad()
            train_correct += accuracy_count(out.squeeze(-1), labels)
            train_loss.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            for batch in val_dl:
                images, labels = batch
                out = model(images)
                val_correct += accuracy_count(out.squeeze(-1), labels)

        print(f"Val: {val_correct/float(len(val_ds))}, Train: {train_correct/float(len(train_ds))}")    


fit(epochs, lr, model, train_dl, val_dl, torch.optim.Adam)

import os
import torch
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

model.eval()
transforms = v2.Compose([
    v2.Resize((256,256)),
    v2.ToTensor(),
])



def preprocess_image_pytorch(img_path, target_size=(256, 256)):
    preprocess = transforms
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def predict_and_save_csv_pytorch(model, images_dir, output_csv='./predictions.csv', target_size=(512, 512), device='cpu'):
    image_names = []
    predicted_labels = []

    # Example class labels
    class_labels = ['AI', 'Real']
    labels_pred = []
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(images_dir, filename)
            try:

                img_tensor = preprocess_image_pytorch(img_path, target_size).to(device)
                with torch.no_grad():
                    outputs = model(img_tensor)
                    labels_pred.append((outputs))
                    if outputs>0.5:
                        outputs = 1
                    else:
                        outputs = 0
                    predicted_class = outputs
                predicted_label = class_labels[predicted_class]
                image_names.append(filename)
                predicted_labels.append(predicted_label)
                print(f"Predicted {predicted_label} for {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    print(labels_pred)
    try:
        # Create a DataFrame
        image_names = file_names_without_extension = [os.path.splitext(file)[0] for file in image_names]
        df = pd.DataFrame({
            'Id': image_names,
            'Label': predicted_labels
        })
        print("DataFrame created successfully.")
        # Debug Statement
        df['num_part'] = df['Id'].str.extract('(\d+)').astype(int)

        # Sort and clean DataFrame
        df_sorted = df.sort_values(by='num_part').reset_index(drop=True)
        df_sorted = df_sorted.drop(['num_part'], axis=1)
        # Save to CSV
        df_sorted.to_csv(output_csv, index=False)
        print(f"Predictions saved to {os.path.abspath(output_csv)}")  # Debug Statement
    except Exception as e:
        print(f"Error saving CSV: {e}")

    print(f"Predictions saved to {os.path.abspath(output_csv)}")

predict_and_save_csv_pytorch(model, "./test_data", device = device)








