from torchvision.datasets import VisionDataset
import re
import os
from PIL import Image
import os
import os.path
import sys
import torch
import copy


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):  #
        super(Caltech, self).__init__(root,
                                      transform=transform,
                                      target_transform=target_transform)

        self.list_categories = []
        self.images = []
        self.transform = transform

        list_dir = os.listdir(root)
        list_dir.pop(list_dir.index('BACKGROUND_Google'))

        for cat in list_dir:
            self.list_categories.append(cat.lower().strip())

        self.list_categories = sorted(self.list_categories)

        for cat in list_dir:
            for image in os.listdir("%s/%s" % (root,cat)):
                image_location = ("%s/%s/%s" % (root,cat,image))
                category_index = self.list_categories.index(cat.lower().strip())
                self.images.append([image_location,category_index])

    def get_category_list(self):
        cat_copied = copy.deepcopy(self.list_categories)
        return cat_copied

    def get_cagetory_image(self, index):
        return self.list_categories[self.images[index][1]]

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        #image, label = ...  # Provide a way to access image and label via index
        # Image should be a PIL Image
        # label can be int

        # Applies preprocessing when accessing the image
        #if self.transform is not None:
        #    image = self.transform(image)

        image, label = self.images[index]

        image = pil_loader(image)
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.images)
        return length