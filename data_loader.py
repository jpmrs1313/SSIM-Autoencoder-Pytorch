from torch.utils.data import Dataset
import torch
from skimage import io, transform, filters
import numpy as np
import glob

class TrainDataset(Dataset):
    def __init__(self,path,image_shape):
        self.path=path
        self.images = sorted(glob.glob(path+"/*/*.png"))
        self.image_shape=image_shape
    
    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path):
        image = io.imread(image_path)
        image = transform.resize(image, self.image_shape)
        image = filters.gaussian(image, sigma=0.4)

        image = np.transpose(image, (2, 0, 1))
        return image

    def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            img_path=self.images[idx]
            image = self.transform_image(img_path)
    
            sample = {'image': image}

            return sample