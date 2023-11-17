import torch
import torch.utils.data

import pandas as pd
from PIL import Image


class ImgDataSet(torch.utils.data.Dataset):
    def __init__(self, dataframe, transformer=None):
        #self.csvname = "/Users/carolinejohnson/Desktop/finalProject/data/output_file.csv"
        self.transformer = transformer
        self.df = dataframe
        #self.df = pd.read_csv(self.csvname)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        print("Index:", index)
        print("DataFrame length:", len(self.df))
        image_path = self.df.iloc[index]['image_path']
        # Load and transform your image using the load_and_transform_image function
        image = load_and_transform_image(image_path, self.transformer)
        label = self.df.iloc[index]['labels']  # Replace with your actual label extraction logic
        return image, label

def load_and_transform_image(path, transform):
    print(path)
    image = Image.open(path).convert("RGB")

    # Apply the specified transformations
    if transform is not None:
        image = transform(image)

    return image

def get_data_loader(valid_data, train_data, transform, batch_size=32):
    #valid_dataset = ImgDataSet(valid_data, transformer=transform)
    #train_dataset = ImgDataSet(train_data, transformer=transform)


    # Note: You can change the batch_size below depends on your hardware
    # If you are using a CPU or Laptop, the bacth size should be below 50
    # IF you are using a GPU, the batch size can be 100 or even 1000
    trainloader = torch.utils.data.DataLoader(train_data,
                                              batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data,
                                              batch_size=batch_size, shuffle=False)

    return trainloader, validloader
