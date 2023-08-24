import os
import os.path
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)

# output a image_label list
def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if len(line_split) != 3:
            raise (RuntimeError("Image list file read line error : " + line + "\n"))
        image_name = os.path.join(line_split[0])
        label_name = os.path.join(line_split[1])
        edge_name=os.path.join(line_split[2])
        item = (image_name, label_name,edge_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


class SemData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None): 
        # print(data_root)
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list) 
        self.transform = transform
        self.name = ' '
        self.kernel = np.ones((5, 5), np.uint8)

    def __len__(self):  
        return len(self.data_list)

    def __getitem__(self, index): 
        image_path, label_path,edge_path = self.data_list[index]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        if image is None:
            print(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        img_H, img_W = label.shape  
        img_size = torch.Tensor([img_H, img_W])

        edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            image,label,edge= self.transform(image,label,edge)
        return image, label , edge , img_size

