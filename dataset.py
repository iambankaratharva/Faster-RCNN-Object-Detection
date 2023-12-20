import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        #############################################
        # TODO Initialize  Dataset

        self.images = h5py.File(path[0], mode='r')['data']
        self.masks = h5py.File(path[1], mode='r')['data']
        self.labels = np.load(file=path[2], allow_pickle=True)
        self.bboxes = np.load(file=path[3], allow_pickle=True)

        # Prepare for preprocessing
        self.masks_stacked = []
        self.preprocess()

        # Define image and mask transformations with specific parameters
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        resize_dims = (800, 1066)
        padding = (11, 0)
        padding_fill = 0

        self.transform_images = transforms.Compose([
            transforms.Normalize(mean=image_mean, std=image_std),
            transforms.Resize(resize_dims),
            transforms.Pad(padding=padding, fill=padding_fill, padding_mode='constant')
        ])

        self.transform_masks = transforms.Compose([
            transforms.Resize(resize_dims),
            transforms.Pad(padding=padding, fill=padding_fill, padding_mode='constant')
        ])

        #############################################

    def preprocess(self):

        #Reference: used gpt for shape adjustemnts, debgging in gooogle and handout instructions
        #Reference: https://www.interviewkickstart.com/learn/the-append-function-in-python (cross checked for reference)


        index_accumulator = 0
        grouped_indices = []

        # Iterating through labels to create a list of mask indices
        for label_group in self.labels:
            current_group_indices = list(range(index_accumulator, index_accumulator + len(label_group)))
            grouped_indices.append(current_group_indices)
            index_accumulator += len(label_group)

        # Aggregating masks based on the grouped indices
        for group in grouped_indices:
            stacked_mask = self.masks[group]
            self.masks_stacked.append(stacked_mask)

    def __len__(self):
        return len(self.images)

    # In this function for given index we rescale the image and the corresponding  masks, boxes
    # and we return them as output
    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
        # index
    def __getitem__(self, index):
        ################################
        # TODO return transformed images,labels,masks,boxes,index
        ################################

        image, masks, label, bboxes = (self.images[index],
                                    self.masks_stacked[index],
                                    self.labels[index],
                                    self.bboxes[index])
    
        # Transform image, masks, and bboxes
        transed_img, transed_mask, transed_bbox = self.pre_process_batch(image, masks, bboxes)
        assert transed_img.shape == (3,800,1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]

        
        return transed_img, label, transed_mask, transed_bbox, index

    # This function preprocess the given image, mask, box by rescaling them appropriately
    # output:
    #        img: (3,800,1088)
    #        mask: (n_box,800,1088)
    #        box: (n_box,4)
    def pre_process_batch(self, img, mask, bbox):
        #######################################
        # TODO apply the correct transformation to the images,masks,boxes

            #Reference: used gpt for shape adjustemnts, debgging in gooogle and handout instructions
        #Reference: https://stackoverflow.com/questions/509211/how-slicing-in-python-works
 
        img_norm = img/255.
        img_norm_s = torch.tensor(img_norm, dtype = torch.float)
        img_norm =  img_norm_s.unsqueeze(0)
        img_norm = self.transform_images(img_norm)

        # self.transform_images((img/255.).astype(np.float32).transpose())

        #Mask Transform
        mask_norm = torch.zeros((1, len(mask), 800, 1088))
        for idx in range(len(mask)):
            msk = mask[idx]/1.
            msk = torch.tensor(msk, dtype = torch.float).unsqueeze(0)
            msk = self.transform_masks(msk)
            msk[msk > 0.5] = 1
            msk[msk < 0.5] = 0
            mask_norm[:,idx] = msk

        #Box Transform
        scaled_box = np.zeros_like(bbox)
        scaled_box[:,1] = bbox[:,1] * (800/300)
        scaled_box[:,3] = bbox[:,3] * (800/300) 
        scaled_box[:,0] = (bbox[:,0] * (1066/400)) + 11
        scaled_box[:,2] = (bbox[:,2] * (1066/400)) + 11
        ######################################

        assert img_norm.squeeze(0).shape == (3, 800, 1088)
        assert scaled_box.shape[0] == mask_norm.squeeze(0).shape[0]

        return img_norm.squeeze(0), mask_norm.squeeze(0), scaled_box


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset    = dataset
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.num_workers = num_workers


    # output:
    #  dict{images: (bz, 3, 800, 1088)
    #       labels: list:len(bz)
    #       masks: list:len(bz){(n_obj, 800,1088)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #       index: list:len(bz)
    def collect_fn(self, batch):
            # Initialize a dictionary to hold the organized batch data.
        out_batch = {
            "images": [],  # To store image data
            "labels": [],  # To store label data
            "masks": [],   # To store mask data
            "bbox": [],    # To store bounding box data
            "index": []    # To store index data
        }
        
        # Iterate through each item in the input batch.
        for item in batch:
            # Extract and process individual parts of the batch item.
            image, label, mask, bbox, index = item

            # Append the image to the temporary list (we'll stack them later).
            out_batch["images"].append(image)
            
            # Append the label, mask, and index directly to their respective lists in out_batch.
            out_batch["labels"].append(label)
            out_batch["masks"].append(mask)
            out_batch["index"].append(index)
            
            # Convert bbox numpy array to a PyTorch tensor and append it to the bbox list.
            out_batch["bbox"].append(torch.from_numpy(bbox))

        # Stack the images along a new dimension to create a single tensor for images.
        out_batch["images"] = torch.stack(out_batch["images"])

        # Return the organized batch data (works, checked it and you can continue).
        return out_batch


    def loader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collect_fn)
