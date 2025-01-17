import os

import torch
import debug
from debug import Classifier
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

model = Classifier(debug.in_channels)
state_dict = torch.load("./Real_or_fake_2.pth")
model.load_state_dict(state_dict)
model = model.to(debug.device)

model.eval()
transforms = v2.Compose([
    v2.Resize((1024,1024)),
    v2.ToTensor(),
    v2.Normalize(*debug.stats)
])

test_ds = ImageFolder("./induction-task/Data/Test", transform= transforms)
image, _ = test_ds[12]

def denormalize(images, means, stds):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means


def preprocess_image_pytorch(img_path, target_size=(1024, 1024)):
    preprocess = transforms
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def predict_and_save_csv_pytorch(model, images_dir, output_csv='./predictions.csv', target_size=(256, 256), device='cpu'):
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

predict_and_save_csv_pytorch(model, "./induction-task/Data/Test2/Test_Images", device = debug.device)







