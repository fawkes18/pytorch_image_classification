import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import pandas as pd
from skimage import io
import torchvision.transforms as transforms
import math
from PIL import Image

os.chdir('C://Users/Marco/Desktop/MenVsWomen')

class MenVsWomen(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return(len(self.dataframe.index))

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.dataframe.iloc[index, 0])
        image = io.imread(img_path, plugin='matplotlib')
        y_label = torch.tensor(int(self.dataframe.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return image, y_label


def create_df(dir):
    filenames = os.listdir(dir)
    categories = []
    for f in filenames:
        category = f.split('_')[0]
        if category.startswith('men'):
            categories.append(0)
        else:
            categories.append(1)
    df = pd.DataFrame({'filename': filenames, 'category': categories})
    return df


def check_images(dir):
    for filename in os.listdir(dir):
        if filename.endswith('.jpg'):
            try:
                img = Image.open(dir + '/' + filename)
                img.verify()
            except (IOError, SyntaxError) as e:
                print('Bad file:', filename)

check_images('./men_vs_women_data/')



def transform_data():
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    return data_transform


def load_data():
    dataset = MenVsWomen(dataframe=create_df('./men_vs_women_data/'), root_dir='./men_vs_women_data/',
                         transform=transform_data())
    train_size = math.ceil(len(dataset)*0.8)
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset,  [train_size, test_size])
    train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=64)
    test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=64)
    return train_loader, test_loader


