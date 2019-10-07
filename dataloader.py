import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image

class im_dataset(Dataset):
    def __init__(self, real_dir, im_size):
        self.real_dir = real_dir
        self.imgpaths = self.get_imgpaths()

        self.preprocessing = transforms.Compose([
                transforms.Resize(im_size),
                transforms.CenterCrop(im_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def get_imgpaths(self):
        real_paths = sorted(glob.glob('%s/*.jpg'%self.real_dir, recursive=True))
        return real_paths
    

    def __getitem__(self, idx):
        truepath = self.imgpaths[idx]
        true_im = self.preprocessing(Image.open(truepath))
        if true_im.size(0) == 1:
            return self.__getitem__(np.random.randint(0, self.__len__()))
        
        return true_im

    def __len__(self):
        return len(self.imgpaths)
