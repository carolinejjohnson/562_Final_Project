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

# neural network architecture (e.g., a simple CNN)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        #  model architecture here
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 32 * 32, num_classes)

    def forward(self, x):
        # forward pass of your model
        x = F.relu(self.conv1(x))  # After convolutional layer
        x = self.pool(x)           # After max pooling
        print("forward",x.size())  
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
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
        y = torch.tensor(self.y_in.iloc[idx], dtype=torch.float32)
        # Load and preprocess the image
        img = Image.open(x).convert('RGB') 
        if self.transform:
            img = self.transform(img)

        return img, y



def main():
    # Load the CSV file into a pandas DataFrame
    df1 = pd.read_csv('data/output_file_test.csv')
    #df2 = pd.read_csv('/Users/carolinejohnson/Desktop/finalProject/data/new.csv')

    X = df1['image_path']
    y = df1['img_lbl_one_hot_encoding']

    print(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #type(X_train)
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    batch_size = 64
   
    train_dataset = CustomDataset(X_train, y_train, transform=transform)
    test_dataset = CustomDataset(X_test, y_test, transform=transform)

    # DataLoader
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer

    num_classes = 333 # number of unique labels
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
    
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # # Training loop
    # print(train_loader)
    # for epoch in range(10):
    #     for images, labels in train_loader:
    #         optimizer.zero_grad()
    #         print(images.size())
    #         outputs = model(images)
    #         labels = labels.long()
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    # model.eval()
    # with torch.no_grad():
    #     correct = 0
    #     total = 0

    #     for images, labels in test_loader:
    #         outputs = model(images)

    #         # Assuming predicted is a tensor of shape (batch_size, num_classes)
    #         predicted = torch.round(torch.sigmoid(outputs))

    #         # Flatten both predicted and labels to a 1D tensor
    #         predicted_flat = predicted.view(-1)
    #         labels_flat = labels.float().view(-1)

    #         # Ensure that the sizes match before performing the comparison
    #         if predicted_flat.size(0) == labels_flat.size(0):
    #             correct += (predicted_flat == labels_flat).sum().item()
    #             total += labels_flat.size(0)
    #         else:
    #             print("Warning: Batch sizes do not match. Skipping this batch.")

    # accuracy = correct / total
    # print(f'Test Accuracy: {accuracy}')


def run(mode, dataloader, model, optimizer=None, use_cuda = False):
    """
    mode: either "train" or "valid". If the mode is train, we will optimize the model
    """
    running_loss = []
    criterion = nn.CrossEntropyLoss()

    actual_labels = []
    predictions = []
    for inputs, labels in dataloader:
        labels = labels.long()
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss.append(loss.item())

        actual_labels += labels.view(-1).cpu().numpy().tolist()
        _, pred = torch.max(outputs, dim=1)

        predictions += pred.view(-1).cpu().numpy().tolist()

        if mode == "train":
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    acc = np.sum(np.array(actual_labels) == np.array(
        predictions)) / len(actual_labels)
    print(mode, "Accuracy:", acc)

    loss = np.mean(running_loss)

    return loss, acc



if __name__ == '__main__':
    main()