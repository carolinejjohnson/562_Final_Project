#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Sampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import BatchSampler
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder


import torch.nn as nn
import torch.nn.functional as F
import argparse

import numpy as np
import torch
import torch.optim as optim

import loader
import helper
# Example Model Class
class Test(nn.Module):
    def __init__(self, num_classes=10):
        super(Test, self).__init__()
        # Define your model architecture here
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 200 * 250, 545)  # Adjust the number of classes

    def forward(self, x):
        # Define the forward pass of your model
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 200 * 250)
        x = self.fc1(x)
        return x
    

def main():
    # Process each line and extract image ID and labels
    data = {'image_path': [], 'image_id': [], 'labels': []}
    
    # main_directory = '/Users/carolinejohnson/Desktop/finalProject/data'

    # for class_folder in os.listdir(main_directory):
    #     class_path = os.path.join(main_directory, class_folder)
    #     if os.path.isdir(class_path):
    #         descriptions_file_path = os.path.join(class_path, 'descriptions.txt')
    #         print(descriptions_file_path)
    #         with open(descriptions_file_path, 'r') as file:
    #             lines = file.readlines()
    #             # Iterate over description
    #             image_id = None
    #             labels = []
    #             description_lines = []
    #             for line in lines:
    #                 parts = line.strip().split('.', 1)  # Split at the first period
    #                 if len(parts) == 2:
    #                     # If we're processing a new image, append the previous one to the data
    #                     if image_id is not None:
    #                         data['image_id'].append(image_id)
    #                         data['labels'].append(labels)
    #                         data['image_path'].append(class_path + '/Image' + image_id + '.jpg')

    #                     # Start processing a new image
    #                     image_id = parts[0].strip()
    #                     labels = parts[1].strip().split(', ')
    #                 elif image_id is not None:
    #                     # Continue adding lines to the description
    #                     labels.extend(line.strip().split(', '))

    #             # After the loop, append the last image to the data
    #             if image_id is not None:
    #                 data['image_id'].append(image_id)
    #                 data['labels'].append(labels)
    #                 # NOT CORRECT PATH FOR COLUMBIA GORGE
    #                 data['image_path'].append(class_path + '/Image' + image_id + '.jpg')

    # # Create a DataFrame
    # df = pd.DataFrame(data)
    # image_path = '/Users/carolinejohnson/Desktop/finalProject/data/football/Image02.jpg'
    # result = df.loc[df['image_path'] == image_path]
    # labels_for_image = result.iloc[0]['labels']
    # print(labels_for_image)


    # # Display the DataFrame
    # print(df)
    

    # #mlb = MultiLabelBinarizer()
    # #binary_labels = mlb.fit_transform(df['labels'])

    # # Create a new column 'label_str' by converting the lists of labels to a string
    # df['labels_str'] = df['labels'].apply(lambda x: ' '.join(sorted(x)))

    # print(df)
    # print(df.columns)

    # # Initialize StratifiedKFold
    # splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)

    # # Splits will store the indices for each fold
    # splits = []

    # # Use the indices of 'label_str' for splitting
    # for train_idx, test_idx in splitter.split(df['image_path'], df['labels_str']):
    #     splits.append((train_idx, test_idx))

    df_train = pd.read_csv("/Users/carolinejohnson/Desktop/finalProject/data/output_file.csv")
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    num_classes = df_train['labels'].nunique()
        # Assuming df is your DataFrame with a column 'labels'
    labels = df_train['labels']

    # Initialize the label encoder
    label_encoder = LabelEncoder()

    # Fit and transform the labels
    encoded_labels = label_encoder.fit_transform(labels)

    # Replace the original 'labels' column with the encoded labels
    df_train['labels'] = encoded_labels

    splits = []
    for train_idx, test_idx in splitter.split(df_train['image_path'], df_train['labels']):
        splits.append((train_idx, test_idx))

    # Example: Displaying the splits
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"Fold {i + 1}:")
        print("Train indices:", train_idx)
        print("Test indices:", test_idx)
        print("---")

    # Define your data transformation
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=45, translate=(0.05, 0.05), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.CenterCrop((400, 500)),
        transforms.ToTensor()
    ])

    # Specify other training parameters
    batch_size = 32  # Adjust as needed
    num_epochs = 10  # Adjust as needed
    df_train = df_train.reset_index(drop=True)

    # Loop over each fold for cross-validation
    for i, (train_idx, valid_idx) in enumerate(splits):
        print("Actual DataFrame length:", len(df_train))

        print(f"Fold {i + 1}: Train indices {min(train_idx)} to {max(train_idx)}, Valid indices {min(valid_idx)} to {max(valid_idx)}")
        print(f"\nTraining on Fold {i + 1}:")

       # Assuming you have a DataFrame df and valid_idx, train_idx obtained from the cross-validation split
        valid_data = loader.ImgDataSet(df_train.iloc[valid_idx], transformer=transform)
        train_data = loader.ImgDataSet(df_train.iloc[train_idx], transformer=transform)

        # Create DataLoader for training and validation sets
        batch_size = 32  # Adjust as needed
        train_loader, valid_loader = loader.get_data_loader(valid_data, train_data, transform, batch_size=batch_size)
        # Initialize your model
        model = Test(num_classes)  # Replace with your model

        # Initialize your optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adjust as needed
        criterion = torch.nn.CrossEntropyLoss()  # Replace with your loss function

        # Training loop
        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []
        for epoch in range(num_epochs):
            for labels in train_loader:
                learning_rate = 0.01 * 0.8 ** epoch
                learning_rate = max(learning_rate, 1e-6)
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
                print(train_loader, "train loader")
                print(model, "model")
                loss, acc = helper.run("train", train_loader, model, optimizer, use_cuda=False)
                train_losses.append(loss)
                train_accs.append(acc)
                with torch.no_grad():
                    loss, acc = helper.run("valid", valid_loader, model, use_cuda=False)
                    valid_losses.append(loss)
                    valid_accs.append(acc)
            
    
    print("-"*60)
    print("best validation accuracy is %.4f percent" % (np.max(valid_accs) * 100) )
    




if __name__ == '__main__':
    main()



# Example Dataset Class
class YourDatasetClass(Dataset):
    def __init__(self, df, transform=None):
        self.data = df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assuming you have an 'image_path' column in your DataFrame
        image_path = self.data.iloc[idx]['image_path']
        # Load your image data and apply transformations if needed
        # image = load_and_transform_image(image_path, self.transform)
        # Assuming you have a 'label' column in your DataFrame
        label = self.data.iloc[idx]['label']
        return image, label


