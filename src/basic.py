#!/usr/bin/env python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import ast
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch
import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from torchmetrics.classification import MultilabelAccuracy
import matplotlib.pyplot as plt
import logging





class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # model architecture here
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for multi-label classification

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)  # Apply Sigmoid activation
        return x



class CustomDataset(Dataset):
    def __init__(self, x_in, y_in, transform=None):
        self.x_in = x_in
        self.y_in = y_in
        self.transform = transform

    def __len__(self):
        return len(self.x_in)

    def __getitem__(self, idx):
        x = self.x_in.iloc[idx]
        #y = torch.tensor(self.y_in.iloc[idx], dtype=torch.float32)
        y = self.y_in[idx]
        # Load and preprocess the image
        img = Image.open(x)#.convert('RGB') 
        if self.transform:
            img = self.transform(img)

        return img, y



def main():
    logging.basicConfig(filename='output.log', level=logging.INFO)

    # Load the CSV file into a pandas DataFrame
    df1 = pd.read_csv('data/output_file_new1.csv')
    #df2 = pd.read_csv('/Users/carolinejohnson/Desktop/finalProject/data/new.csv')

    X = df1['image_path']
    y = df1['labels']
    print("[INFO] class labels:")
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)
    # loop over each of the possible class labels and show them
    for (i, label) in enumerate(mlb.classes_):
        print("{}. {}".format(i + 1, label))

    print(X)
    print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    #type(X_train)
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=5, shear=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ])

    batch_size = 64
   
    train_dataset = CustomDataset(X_train, y_train, transform=transform)
    test_dataset = CustomDataset(X_test, y_test, transform=transform)

    # DataLoader
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Initialize model, loss function, and optimizer

    num_classes = 61 # number of unique labels
    model = SimpleCNN(num_classes=num_classes)

    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    for epoch in range(10):  # loop over the dataset multiple times
        learning_rate = 0.01 * 0.8 ** epoch
        learning_rate = max(learning_rate, 1e-6)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
        loss, acc = run("train", train_loader, model, optimizer, use_cuda=False)
        train_losses.append(loss)
        train_accs.append(acc)
        with torch.no_grad():
            loss, acc = run("valid", test_loader, model, use_cuda=False)
            valid_losses.append(loss)
            valid_accs.append(acc)
    
    
    print("-"*60)
    print("best validation accuracy is %.4f percent" % (np.max(valid_accs) * 100) )
    # Plotting
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss_plot.png')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('training_accuracy_plot.png')

def run(mode, dataloader, model, optimizer=None, use_cuda = False):
    """
    mode: either "train" or "valid". If the mode is train, we will optimize the model
    """
    running_loss = []
    criterion = nn.BCELoss()

    actual_labels = torch.empty((0,64,61), dtype=torch.float32)
    predictions = torch.empty((0,64,61), dtype=torch.float32)

    for inputs, labels in dataloader:
        labels = labels.float()
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        # forward + backward + optimize
        outputs = model(inputs)
        # outputs = (outputs > 0.5).float()
        loss = criterion(outputs, labels)
        running_loss.append(loss.item())
        labels = labels.unsqueeze(0)
        actual_labels = torch.cat((actual_labels, labels), dim=0)

        pred = outputs.detach()
        pred = (pred > 0.5).float()

        pred = pred.unsqueeze(0)

        predictions = torch.cat((predictions, pred), dim=0)

        if mode == "train":
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    #outputs = outputs.detach().numpy()
    #acc = accuracy_score(y_true=actual_labels, y_pred=predictions) # Also gives the accuracy for the two lists actual and pred
    metric = MultilabelAccuracy(num_labels=64, multidim_average='samplewise')
    acc = metric(actual_labels, predictions)
    acc = torch.mean(acc)
    acc = acc.item()
    #print(metric)
    #torchmetrics.functional.accuracy(labels, outputs, subset_accuracy=True)
    #correct_predictions = (predictions == actual_labels).sum().item()
    #total_samples = len(actual_labels)
    #acc = correct_predictions / total_samples

    print(mode, "Accuracy:", acc)

    loss = np.mean(running_loss)
    for i, (predict, ground) in enumerate(zip(actual_labels, predictions)):
        one_hot_encodings1 = [torch.nonzero(row).squeeze().tolist() for row in predict]
        one_hot_encodings2 = [torch.nonzero(row).squeeze().tolist() for row in ground]
        logging.info("p: %s", one_hot_encodings1)
        logging.info("a: %s", one_hot_encodings2)
        #print(f"Example {i + 1}: Predicted: {one_hot_encodings1}, Ground Truth: {one_hot_encodings2}")

    return loss, acc



if __name__ == '__main__':
    main()