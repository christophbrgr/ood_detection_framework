# adapted from https://github.com/kamenbliznashki/chexpert/blob/2bf52b1b70c3212a4c2fd4feacad0fd198fe8952/dataset.py#L17

from PIL import Image

import numpy as np
import pandas as pd
import os

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from types import SimpleNamespace

class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

def label_class(row):
    for i, col in enumerate(row[5:]):
        if col == 1:
            return i
    
def selectClasses(df, classes_in=None, classes_out=None, verbose=True):
    # add lengths of classes so we know where to sum up
    total_classes = classes_in + classes_out
    # class names should be a constant!
    to_drop = [c for c in class_names if c not in total_classes] # select classes we don't care about
    df = df.drop(to_drop, axis=1)
    class_count = len(total_classes)
    frames = []
    for c in classes_in:
        temp = df[(df[c] == 1.0) & (df.iloc[:, -class_count:].sum(axis=1) <= 1.0)]
        frames.append(temp)
        if verbose:
            print(f'Selecting class: {c} with {len(temp.index)} rows.')
    out = pd.concat(frames, ignore_index=True)
    out = out.drop(classes_out, axis=1)
    out['class'] = out.apply(lambda row: label_class(row), axis=1)
    # balance the classes to match the smallest class
    out = out.groupby('class')
    out = out.apply(lambda x: x.sample(out.size().min()).reset_index(drop=True))
    if verbose:
        print('Balanced classes to match the smallest class:')
        for c in classes_in:
            print(f'Length of class {c} is now: {len(out.loc[out[c] == 1])}')
    # drop everything
    # out = out.drop(classes_in, axis=1)
    return out

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def fetch_dataloader(args, dataframe, transforms=None, drop_last_batch=False):

    if transforms is None:
        transforms = T.Compose([
            T.Resize(args.resize) if args.resize else T.Lambda(lambda x: x),
            T.CenterCrop(224 if not args.resize else args.resize),  # to make the images square
            T.RandomRotation(degrees=5),
            # T.RandomCrop(224, padding=4),          # some light augmentation
            # T.RandomHorizontalFlip(),               # some light augmentation
            # lambda x: torch.from_numpy(np.array(x, copy=True)).float().div(255).unsqueeze(0),   # tensor in [0,1]
            T.ToTensor(),
            T.Normalize(mean=[0.5330], std=[0.0349]),                                           # whiten with dataset mean and std
            lambda x: x.expand(3,-1,-1)])                                                       # expand to 3 channels
    
    dataset = ChexpertSmall(args.root, dataframe, transforms)

    # initialize all workers with a different seed, otherwise it would always use the same seed and thus augmentations for each worker
    return DataLoader(dataset, args.batch_size, shuffle=args.shuffle, pin_memory=False, num_workers=4, drop_last=drop_last_batch, worker_init_fn=worker_init_fn)


class ChexpertSmall(Dataset):
    
    def __init__(self, folder_dir, dataframe, transforms):
        """
        Init Dataset
        
        Parameters
        ----------
        folder_dir: str
            folder contains all images
        dataframe: pandas.DataFrame
            dataframe contains all information of images
        """
        self.image_paths = [] # List of image paths
        self.image_labels = [] # List of image labels
        self.patient_ids = []
        
        self.transform = transforms
        count = 0
        # Get all image paths and image labels from dataframe
        for index, row in dataframe.iterrows():
            self.patient_ids.append(row.Path) # so we can match classifications later on
            image_path = os.path.join(folder_dir, row.Path)
            self.image_paths.append(image_path)
            count+=1
            self.image_labels.append(row['class'])
        # print(f'Added {count} rows to the loader')
        # print(f'Image loader has {len(self.image_labels)} labels.')
        assert len(self.image_labels) == len(self.image_paths), 'Label count does not match the image count.'

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        """
        Read image at index and convert to torch Tensor
        """
        
        # Read image
        image_path = self.image_paths[index]
        image_data = Image.open(image_path) # Convert image to RGB channels

        # print(f'Shape of {image_path}: {image_data.size}')
        
        if self.transform is not None:
            image = self.transform(image_data)
        # Resize and convert image to torch tensor 
        #print(f'Image labels: {self.image_labels[index]}')
        
        return image, self.image_labels[index] # , self.patient_ids[index]