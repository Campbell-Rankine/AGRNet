"""
Produce augmented dataset batch. 
ProGAN paper provides a time for augmentation
"""
import numpy as np
from PIL import Image

from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, Resize, ToTensor
import torchvision
import torch as T
from keras.preprocessing.image import ImageDataGenerator

class AugmentedData(Dataset):
    """
    input: Define training structure, see args docs for the 
    """

    def __init__(self,img_folder,names,transform, structure, scalefactor):
        #What happens on init
        return None
    def __getitem___(self):
        #Return images to be loaded with a dataloader
        return None
    
