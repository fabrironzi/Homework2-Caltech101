from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
import torch

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    
    set_categories = set()
    images = []
    transform = None

    
    def __init__(self, root, split="train", transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)
        
        self.transform = transform

        if ( split == "train" or split == "test"):
            
            self.split = "Homework2_Caltech101/" + str(split) + ".txt"
            print(self.strip)
            
            #with open("Homework2_Caltech101/train.txt", 'r') as fp:
            with open(self.split, 'r') as fp:
                content = fp.readline()
                
                for content in fp:
                    
                    if(content.startswith("BACKGROUND_Google")):
                        print('Repository')
                        continue
                    
                    data = ["./Homework2_Caltech101/101_ObjectCategories/" + content]
                    self.set_categories.add(content.split("/"[0]))
                    data.append(list(self.set_categories).index(content.split("/")[0]))
                    
                    try:
                        image_loaded = pil_loader(data[0])
                        data[0] = image_loaded
                        data[0] = self.transform(data[0])
                    except Exception as e:
                        print(e)
                        
                    self.images.append(data)
                    
            print(data)
                
               
            fp.close()         
            

# =============================================================================
#         self.split = split # This defines the split you are going to use
#                            # (split files are called 'train.txt' and 'test.txt')
# 
# =============================================================================
        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = self.images[index]
        
        if not isinstance(image, torch.Tensor):
            image = pil_loader(image)
            image = self.transform(image)
            
        
        
        
        # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
# =============================================================================
#         if self.transform is not None:
#             image = self.transform(image)
# =============================================================================

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.images)
        
        # Provide a way to get the length (number of elements) of the dataset
        return length
