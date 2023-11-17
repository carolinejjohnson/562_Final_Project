import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import ast
import torch.nn as nn
import torch.nn.functional as F

class MultiLabelCNN(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelCNN, self).__init__()
        # Define your model architecture here
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 200 * 250, num_classes)

    def forward(self, x):
        # Define the forward pass of your model
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 200 * 250)
        x = self.fc1(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None, max_label_length=14):
        self.dataframe = dataframe
        self.transform = transform
        self.max_label_length = max_label_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        labels_str = self.dataframe.iloc[idx]['label']
        
        # Convert the string representation of a list to an actual list
        labels = ast.literal_eval(labels_str)
        
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(labels, dtype=torch.float32)

def main():
    # Load the CSV file into a pandas DataFrame
# Load the CSV file into a pandas DataFrame
    df = pd.read_csv('/Users/carolinejohnson/Desktop/finalProject/data/output_file_new.csv', converters={'label': ast.literal_eval})
    print(df.columns)
    # Split the dataset into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save the split datasets to new CSV files
    train_df.to_csv('/Users/carolinejohnson/Desktop/finalProject/data/train.csv', index=False)
    test_df.to_csv('/Users/carolinejohnson/Desktop/finalProject/data/test.csv', index=False)

    # Load your dataset
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    train_dataset = CustomDataset(train_df, transform=transform)
    test_dataset = CustomDataset(test_df, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize your model, loss function, and optimizer
    num_classes = len(df['label'].iloc[0])
    model = MultiLabelCNN(num_classes)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluation loop
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)

            predicted = torch.round(torch.sigmoid(outputs))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy}')

if __name__ == '__main__':
    main()
