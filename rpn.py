#Standard libraries
import os
import copy
import pandas as pd
from skimage import io, transform
from scipy import stats as st
from scipy import ndimage
from PIL import Image
import numpy as np
import h5py
import cv2

# PyTorch related
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from torch.nn import functional as F
import torchvision
from torchvision import transforms, utils
from torchvision.utils import draw_bounding_boxes

# PyTorch Lightning related
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.callbacks as pl_callbacks

# Local modules
from utils import *
from dataset import *

# Visualization
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as rec

# Multiprocessing
from multiprocessing.connection import wait

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
seed = 17
torch.manual_seed(seed);

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st


class RPNHead(pl.LightningModule):

    def __init__(self,  anchors_param=dict(ratio=0.8,scale= 256, grid_size=(50, 68), stride=16)):
        # Initialize the backbone, intermediate layer clasifier and regressor heads of the RPN
        super(RPNHead,self).__init__()
        # setattr(self, device, anchors_param)
        # self.device = device
        self.train_losses = []
        self.val_losses   = []
        self.accumulated_val_losses = []
        self.accumulated_val_class_losses = []
        self.accumulated_val_regression_losses = []
        self.accumulated_train_losses = []
        self.accumulated_train_class_losses = []
        self.accumulated_train_regression_losses = []

        self.sy     = 800
        self.sx     = 1088
        # TODO Define Backbone
        self.backbone = nn.Sequential(nn.Conv2d(3, 16,    kernel_size=5, stride=1, padding="same"),
                                      nn.BatchNorm2d(16),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                      nn.Conv2d(16, 32,   kernel_size=5, stride=1, padding="same"),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                      nn.Conv2d(32, 64,   kernel_size=5, stride=1, padding="same"),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                      nn.Conv2d(64, 128,  kernel_size=5, stride=1, padding="same"),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                      nn.Conv2d(128, 256, kernel_size=5, stride=1, padding="same"),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU())

        # TODO  Define Intermediate Layer
        self.intermediate_layer = nn.Sequential(nn.Conv2d(256, 256, 3,   stride=1, padding="same"), 
                                                nn.BatchNorm2d(256), 
                                                nn.ReLU())

        # TODO  Define Proposal Classifier Head
        self.proposal_classifier_head = nn.Sequential(nn.Conv2d(256,1,1, stride=1, padding="same"),
                                                      nn.Sigmoid())

        # TODO Define Proposal Regressor Head
        self.proposal_regressor_head = nn.Sequential(nn.Conv2d(256,4,1, stride=1, padding="same"))

        #  find anchors
        self.anchors_param=anchors_param
        self.anchors=self.create_anchors(self.anchors_param['ratio'],self.anchors_param['scale'],self.anchors_param['grid_size'],self.anchors_param['stride'])
        self.ground_dict={}





    # Forward  the input through the backbone the intermediate layer and the RPN heads
    # Input:
    #       X: (bz,3,image_size[0],image_size[1])}
    # Ouput:
    #       logits: (bz,1,grid_size[0],grid_size[1])}
    #       bbox_regs: (bz,4, grid_size[0],grid_size[1])}
    def forward(self, X):

        #TODO forward through the Backbone
        x = self.backbone(X)


        #TODO forward through the Intermediate layer
        intermediate_out = self.intermediate_layer(x)


        #TODO forward through the Classifier Head
        logits = self.proposal_classifier_head(intermediate_out)


        #TODO forward through the Regressor Head
        bbox_regs  = self.proposal_regressor_head(intermediate_out)


        assert logits.shape[1:4]==(1,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])
        assert bbox_regs.shape[1:4]==(4,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return logits, bbox_regs





    # Forward input batch through the backbone
    # Input:
    #       X: (bz,3,image_size[0],image_size[1])}
    # Ouput:
    #       X: (bz,256,grid_size[0],grid_size[1])
    def forward_backbone(self,X):
        #####################################
        # TODO forward through the backbone
        #####################################
        X = self.backbone(X)
        assert X.shape[1:4]==(256,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return X


    # This function creates the anchor boxes
    # Output:
    #       anchors: (grid_size[0],grid_size[1],4)
    def create_anchors(self, aspect_ratio, scale, grid_sizes, stride):
        ######################################
        # TODO create anchors
        ######################################
        yy, xx = torch.meshgrid(torch.arange(grid_sizes[0]), torch.arange(grid_sizes[1]), indexing='ij')
        centers = torch.stack((yy, xx), dim=2) * stride + stride / 2

        # Calculate width and height of anchors
        height = scale / np.sqrt(aspect_ratio)
        width = height * aspect_ratio

        # Prepare grids for widths and heights
        widths = torch.full((grid_sizes[0], grid_sizes[1], 1), width)
        heights = torch.full((grid_sizes[0], grid_sizes[1], 1), height)
        
        # Combine centers and sizes to get the anchor coordinates
        # [y_center, x_center, height, width]
        anchors = torch.cat((centers, heights, widths), dim=2)

        assert anchors.shape == (grid_sizes[0], grid_sizes[1], 4)
        return anchors


    def get_anchors(self):
        return self.anchors


    # This function creates the ground truth for a batch of images by using
    # create_ground_truth internally
    # Input:
    #      bboxes_list: list:len(bz){(n_obj,4)}
    #      indexes:      list:len(bz)
    #      image_shape:  tuple:len(2)
    # Output:
    #      ground_clas: (bz,1,grid_size[0],grid_size[1])
    #      ground_coord: (bz,4,grid_size[0],grid_size[1])

    def create_batch_truth(self,bboxes_list,indexes,image_shape):
    #####################################
    # TODO create ground truth for a batch of images
    #####################################
        ground_truths = [
            self.create_ground_truth(
                torch.from_numpy(bboxes),
                index,
                (self.anchors_param['grid_size'][0], self.anchors_param['grid_size'][1]),
                self.anchors,
                image_size=image_shape
            ) 
            for bboxes, index in zip(bboxes_list, indexes)
        ]

        # Unzipping the list of tuples into two separate lists
        ground_clas_list, ground_coord_list = zip(*ground_truths)

        # Using torch.stack to convert lists of tensors to batch tensors
        ground_clas  = torch.stack(ground_clas_list)
        ground_coord = torch.stack(ground_coord_list)

        # Assert statements to ensure correct tensor shapes
        assert ground_clas.shape[1:4] == (1, self.anchors_param['grid_size'][0], self.anchors_param['grid_size'][1])
        assert ground_coord.shape[1:4] == (4, self.anchors_param['grid_size'][0], self.anchors_param['grid_size'][1])

        return ground_clas, ground_coord


    # This function creates the ground truth for one image
    # It also caches the ground truth for the image using its index
    # Input:
    #       bboxes:      (n_boxes,4)
    #       index:       scalar (the index of the image in the total dataset used for caching)
    #       grid_size:   tuple:len(2)
    #       anchors:     (grid_size[0],grid_size[1],4)
    # Output:
    #       ground_clas:  (1,grid_size[0],grid_size[1])
    #       ground_coord: (4,grid_size[0],grid_size[1])
    def create_ground_truth(self, bboxes, index, grid_size, anchors, image_size):
        key = str(index)
        if key in self.ground_dict:
            groundt, ground_coord = self.ground_dict[key]
            return groundt, ground_coord

        #####################################################
        # TODO create ground truth for a single image
        ground_clas  =  torch.ones(( 1,grid_size[0],grid_size[1]), dtype=torch.double) * (-1)
        ground_coord =  torch.ones((4,grid_size[0],grid_size[1]), dtype=torch.double) 

        flat_anchors = anchors.flatten()
        anchors_xywh = flat_anchors.reshape(-1, 4)

        ax, ay, aw, ah = (anchors_xywh[:, idx] for idx in range(anchors_xywh.shape[1]))

        anc_x1, anc_y1, anc_x2, anc_y2 = (ax - aw / 2.0, ay - ah / 2.0, ax + aw / 2.0, ay + ah / 2.0)



        #Boxes
        bbox_xy     = bboxes
        box_x1, box_y1, box_x2, box_y2 = bbox_xy.T

        bbox_xywh = torch.stack([
            (box_x1 + box_x2) / 2.0,
            (box_y1 + box_y2) / 2.0,
            box_x2 - box_x1,
            box_y2 - box_y1
        ], dim=1)


        #Removing Cross boundary boxes
        invalid_list = torch.tensor([])

        # Initialize invalid_list just once if it's not used previously
        invalid_list = torch.tensor([])

        # Determine invalid and valid anchors based on the given conditions
        invalid_mask = (anc_x1 < 0) | (anc_y1 < 0) | (anc_x2 >= 1088) | (anc_y2 >= 800)
        valid_mask = ~invalid_mask

        # Find the indices of invalid anchors
        invalid = torch.where(invalid_mask)[0]
        row, col = invalid // 68, invalid % 68

        # Update invalid_list and ground_clas in one go
        invalid_list = torch.cat((invalid_list, torch.stack((row, col), dim=1)))
        ground_clas[0, row, col] = -1

        # Use valid_mask to filter out valid anchors directly
        valid_anchor_idx = torch.where(valid_mask)[0]
        row_anc, col_anc = valid_anchor_idx // 68, valid_anchor_idx % 68

        # Gather the valid anchors using advanced indexing
        valid_anchor = anchors[row_anc, col_anc, :].reshape(-1, 4)

        # anchors_xywh is now just a reference to valid_anchor since they hold the same data
        anchors_xywh = valid_anchor

        ax, ay, aw, ah = map(lambda idx: anchors_xywh[:, idx], range(4))

        anchor_corners = [(ax - aw / 2.0), (ay - ah / 2.0), (ax + aw / 2.0), (ay + ah / 2.0)]
        anchor_xy = torch.stack(anchor_corners, dim=1)

        #######################################################
        assigned    = torch.tensor([])
        bbox_dict   = {} 

        for idx in range(len(bboxes)):

            bbox_xy     = bboxes[idx].reshape(1, -1)

            box_x1, box_y1, box_x2, box_y2 = [bbox_xy[:, idx] for idx in range(bbox_xy.shape[1])]

            # bbox_xywh   = torch.vstack((bx,by,bw,bh)).T
            # Calculate the center x, center y, width, and height, and then create the bbox_xywh tensor
            bbox_xywh = torch.stack([
                (box_x1 + box_x2) * 0.5,  # center x
                (box_y1 + box_y2) * 0.5,  # center y
                box_x2 - box_x1,          # width
                box_y2 - box_y1           # height
            ], dim=1)

            
            #Calculate IOU
            IOU_matrix = torchvision.ops.box_iou(bbox_xy, anchor_xy)

            IOU_max    = torch.max(IOU_matrix).item()
            
            true_box = torch.logical_or(IOU_matrix.eq(IOU_max), IOU_matrix.ge(0.7))
            true_idx = torch.nonzero(true_box, as_tuple=True)[1]


            val_row  = row_anc[true_idx] #//68
            val_col  = col_anc[true_idx] #% 68
            valid    = torch.vstack((val_row, val_col)).T

            ground_clas[0,val_row, val_col] = 1

            bbox_dict[idx]  = [valid, IOU_matrix[0,true_idx] , bbox_xywh]

            less_thresh_indices = torch.where((IOU_matrix < 0.3) & (~true_box))[1]
            less_thresh_row = torch.index_select(row_anc, 0, less_thresh_indices)
            less_thresh_col = torch.index_select(col_anc, 0, less_thresh_indices)
            # Stack the row and column indices
            less_thresh = torch.stack((less_thresh_row, less_thresh_col), dim=1)

            # Find indices that should not be deleted because they are not assigned
            dont_delete_indices = torch.where(~torch.isin(less_thresh, assigned, assume_unique=True).all(dim=1))[0]
            # Update the assigned anchors with the valid ones
            assigned = torch.cat((assigned, valid))

            # Set the ground truth classification for the non-deleted indices to 0
            ground_clas[0, less_thresh[dont_delete_indices, 0], less_thresh[dont_delete_indices, 1]] = 0

        indicies = torch.where(ground_clas == 1) 

        # Extract x and y indices from the tuple and transpose to get them into shape [N, 2]
        xy_ind = torch.stack((indicies[1], indicies[2]), dim=1)

        # Flatten the x and y indices to a single dimension
        ind_flat = indicies[1] * 68 + indicies[2]


        bbox_xy     = bboxes.reshape(1,-1)
       
        anchors_xywh = anchors.flatten().reshape(-1,4)

        ax, ay, aw, ah = (anchors_xywh[:, idx] for idx in range(anchors_xywh.shape[1]))

        anc_x1, anc_y1, anc_x2, anc_y2 = (ax - aw / 2.0, ay - ah / 2.0, ax + aw / 2.0, ay + ah / 2.0)

        new_f_anchors = torch.vstack((anc_x1, anc_y1, anc_x2, anc_y2)).T


        IOU_matrix  = torchvision.ops.box_iou(bboxes, new_f_anchors[ind_flat])
        IOU_max     = torch.max(IOU_matrix, dim=0)
        
        bbox_xy     = bboxes

        box_x1, box_y1, box_x2, box_y2 = [bbox_xy[:, idx] for idx in range(bbox_xy.shape[1])]

        bbox_xywh = torch.stack([
    (box_x1 + box_x2) * 0.5,  # center x
    (box_y1 + box_y2) * 0.5,  # center y
    box_x2 - box_x1,          # width
    box_y2 - box_y1           # height
], dim=1)


        row_indices, col_indices = indicies[1], indicies[2]

        # Pre-calculate the anchor differences and ratios for efficiency
        anchor_diffs = bbox_xywh[IOU_max[1], :2] - anchors[row_indices, col_indices, :2]
        anchor_ratios = bbox_xywh[IOU_max[1], 2:] / anchors[row_indices, col_indices, 2:]

        # Update ground_coord with the new values
        for i, (row, col) in enumerate(zip(row_indices, col_indices)):
            anchor_width, anchor_height = anchors[row, col, 2], anchors[row, col, 3]

            # Update the coordinates based on the calculations
            ground_coord[0, row, col] = anchor_diffs[i, 0] / anchor_width
            ground_coord[1, row, col] = anchor_diffs[i, 1] / anchor_height
            ground_coord[2, row, col] = torch.log(anchor_ratios[i, 0])
            ground_coord[3, row, col] = torch.log(anchor_ratios[i, 1])

        # Store the calculated ground truth data in a dictionary
        self.ground_dict[key] = (ground_clas, ground_coord)


        assert ground_clas.shape==(1,grid_size[0],grid_size[1])
        assert ground_coord.shape==(4,grid_size[0],grid_size[1])

        return ground_clas, ground_coord


    # Compute the loss of the classifier
    # Input:
    #      p_out:     (positives_on_mini_batch)  (output of the classifier for sampled anchors with positive gt labels)
    #      n_out:     (negatives_on_mini_batch)  (output of the classifier for sampled anchors with negative gt labels)
    def loss_class(self,p_out,n_out):

        #torch.nn.BCELoss()
        # TODO compute classifier's loss
        loss    = torch.nn.BCELoss(reduction="mean")
        n_out_converted = n_out.float()
        class_loss = loss(p_out, n_out_converted)
        return class_loss



    # Compute the loss of the regressor
    # Input:
    #       pos_target_coord: (positive_on_mini_batch,4) (ground truth of the regressor for sampled anchors with positive gt labels)
    #       pos_out_r: (positive_on_mini_batch,4)        (output of the regressor for sampled anchors with positive gt labels)
    def loss_reg(self,pos_target_coord,pos_out_r, non_zero = False):
            #torch.nn.SmoothL1Loss()
            # TODO compute regressor's loss
            loss        = torch.nn.SmoothL1Loss()
            if non_zero == False:
                reg_loss = 0
                for i in range(4):
                    reg_loss += loss(pos_target_coord[i], pos_out_r[i])
                reg_loss = torch.tensor(reg_loss)
            else:
                reg_loss = torch.tensor(0)

            return reg_loss



    # Compute the total loss
    # Input:
    #       clas_out: (bz,1,grid_size[0],grid_size[1])
    #       regr_out: (bz,4,grid_size[0],grid_size[1])
    #       targ_clas:(bz,1,grid_size[0],grid_size[1])
    #       targ_regr:(bz,4,grid_size[0],grid_size[1])
    #       l: lambda constant to weight between the two losses
    #       effective_batch: the number of anchors in the effective batch (M in the handout)
    def compute_loss(self, clas_out, regr_out, targ_clas, targ_regr, l=1, effective_batch=50):
            #############################
            # TODO compute the total loss
            #############################

            
            positives_indexes   = torch.where(targ_clas == 1)
            negative_indexes    = torch.where(targ_clas == 0)

            p_count = min(positives_indexes[0].size(0), effective_batch // 2)
            n_count = effective_batch - p_count

            # Shuffle indices and select based on effective batch size
            shuffled_pos_indices = torch.randperm(positives_indexes[0].size(0))[:p_count]
            shuffled_neg_indices = torch.randperm(negative_indexes[0].size(0))[:n_count]

            # Extract final indices for positives and negatives
            final_positives_indexes = tuple(positives_indexes[dim][shuffled_pos_indices] for dim in range(4))
            final_negatives_indexes = tuple(negative_indexes[dim][shuffled_neg_indices] for dim in range(4))

            # Assign the indices for positives and negatives
            final_pos_indexes = final_positives_indexes
            final_neg_indexes = final_negatives_indexes

            # Classification loss for positive and negative samples using the original variable names
            pos_gt = targ_clas[final_pos_indexes]  # Ground truth for positive samples
            pos_preds = clas_out[final_pos_indexes]  # Predictions for positive samples
            loss1 = self.loss_class(pos_preds, pos_gt)  # Loss for positive samples

            neg_gt = targ_clas[final_neg_indexes]  # Ground truth for negative samples
            neg_preds = clas_out[final_neg_indexes]  # Predictions for negative samples
            loss2 = self.loss_class(neg_preds, neg_gt)  # Loss for negative samples

                        
            loss_c = loss1 + loss2

            targ_class = targ_clas.view(-1)
            clas_out = clas_out.view(-1)

            # Find the indices of the positive class
            pos_indices = (targ_class == 1).nonzero(as_tuple=False).view(-1)
            eff = effective_batch 

            # Calculate the number of positives needed based on the effective batch size
            num_positives = min(len(pos_indices), eff // 2)

            # Shuffle the indices and select the desired amount for positives
            selected_indices = torch.randperm(len(pos_indices))[:num_positives]
            pos_class = pos_indices[selected_indices]

            non_zero = (pos_class.numel() == 0)

            # Permute and reshape the target and prediction tensors for regression
            reshaped_targ_regr = targ_regr.permute(1, 0, 2, 3).contiguous().view(4, -1)
            reshaped_regr_pred = regr_out.permute(1, 0, 2, 3).contiguous().view(4, -1)

            # Select the regression values for the positive classes
            selected_targ_regr = reshaped_targ_regr[:, pos_class.view(-1)]
            selected_regr_pred = reshaped_regr_pred[:, pos_class.view(-1)]

            # Compute the regression loss (check, this formulation works !!1)
            loss_r = self.loss_reg(selected_targ_regr, selected_regr_pred, non_zero)


            loss = loss_c + l*loss_r

            return loss, loss_c, loss_r



    # Post process for the outputs for a batch of images
    # Input:
    #       out_c:  (bz,1,grid_size[0],grid_size[1])}
    #       out_r:  (bz,4,grid_size[0],grid_size[1])}
    #       IOU_thresh: scalar that is the IOU threshold for the NMS
    #       keep_num_preNMS: number of masks we will keep from each image before the NMS
    #       keep_num_postNMS: number of masks we will keep from each image after the NMS
    # Output:
    #       nms_clas_list: list:len(bz){(Post_NMS_boxes)} (the score of the boxes that the NMS kept)
    #       nms_prebox_list: list:len(bz){(Post_NMS_boxes,4)} (the coordinates of the boxes that the NMS kept)

    def postprocess(self, out_c, out_r, IOU_thresh=0.5, keep_num_preNMS=50, keep_num_postNMS=10):
    # Initialize lists to store outputs
        scores_sorted_list, pre_nms_matrix_list, nms_clas_list, nms_prebox_list = [], [], [], []

        # Iterate over all outputs and perform postprocessing
        for clas_output, regr_output in zip(out_c, out_r):
            scores_sorted, pre_nms_matrix, nms_clas, nms_prebox = self.postprocessImg(
                clas_output, regr_output, IOU_thresh, keep_num_preNMS, keep_num_postNMS
            )
            scores_sorted_list.append(scores_sorted)
            pre_nms_matrix_list.append(pre_nms_matrix)
            nms_clas_list.append(nms_clas)
            nms_prebox_list.append(nms_prebox)

        # Return the aggregated results
        return scores_sorted_list, pre_nms_matrix_list, nms_clas_list, nms_prebox_list




    # Post process the output for one image
    # Input:
    #      mat_clas: (1,grid_size[0],grid_size[1])}  (scores of the output boxes)
    #      mat_coord: (4,grid_size[0],grid_size[1])} (encoded coordinates of the output boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4) (decoded coordinates of the boxes that the NMS kept)

    def postprocessImg(self, mat_clas, mat_coord, IOU_thresh, keep_num_preNMS, keep_num_postNMS):
        #Reference: used gpt to corrcet and debug shapes
    #Reference: https://pytorch.org/tutorials/beginner/ptcheat.html
    #Reference: https://pytorch.org/docs/stable/generated/torch.sort.html
        # TODO postprocess a single image
        anchors = self.get_anchors()
        mat_coord = mat_coord.permute(1, 2, 0)

        # Decode the coordinates
        x_center = (mat_coord[:, :, 0] * anchors[:, :, 2]) + anchors[:, :, 0]
        y_center = (mat_coord[:, :, 1] * anchors[:, :, 3]) + anchors[:, :, 1]
        width = anchors[:, :, 2] * torch.exp(mat_coord[:, :, 2])
        height = anchors[:, :, 3] * torch.exp(mat_coord[:, :, 3])

        # Convert to x1, y1, x2, y2 format
        x1 = x_center - (width / 2.0)
        y1 = y_center - (height / 2.0)
        x2 = x_center + (width / 2.0)
        y2 = y_center + (height / 2.0)

        # Stack and reshape the boxes
        decoded_boxes = torch.stack((x1, y1, x2, y2), dim=2).view(-1, 4)

        # Filter valid boxes
        valid_idx = torch.where(
            (decoded_boxes[:, 0] >= 0) &
            (decoded_boxes[:, 1] >= 0) &
            (decoded_boxes[:, 2] < 1088) &
            (decoded_boxes[:, 3] < 800)
        )[0]
        valid_boxes = decoded_boxes[valid_idx]

        # Flatten and sort the classification scores
        mat_clas_flat = mat_clas.flatten()
        sorted_indices = torch.sort(mat_clas_flat[valid_idx], descending=True)
        top_scores = sorted_indices[0][:keep_num_preNMS]
        top_boxes_indices = sorted_indices[1][:keep_num_preNMS]
        pre_nms_matrix = valid_boxes[top_boxes_indices]

        # Apply NMS
        nms_clas, nms_prebox = self.NMS(top_scores, pre_nms_matrix, thresh=IOU_thresh)

        # Keep only the top results after NMS
        nms_clas = nms_clas[:keep_num_postNMS]
        nms_prebox = nms_prebox[:keep_num_postNMS]

        return top_scores, pre_nms_matrix, nms_clas, nms_prebox



    # Input:
    #       clas: (top_k_boxes) (scores of the top k boxes)
    #       prebox: (top_k_boxes,4) (coordinate of the top k boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)

    def NMS(self, clas, prebox, thresh):
        # TODO perform NMS
        prebox_cp = prebox.clone()
        clas_cp = clas.clone()

        # Compute the pairwise IoU between the boxes
        pairwise_iou = torchvision.ops.box_iou(prebox_cp, prebox_cp)

        # Initialize a set to keep track of indices that we need to delete
        rows_to_delete = set()

        # Iterate through each box and compare with the others
        for i in range(len(prebox_cp)):
            if i not in rows_to_delete:
                for j in range(i + 1, len(prebox_cp)):
                    if pairwise_iou[i, j] > thresh:
                        rows_to_delete.add(j)

        # Delete the rows marked for deletion in both `prebox_cp` and `clas_cp`
        # Using `torch.index_select` for a more efficient and tensor-friendly operation (check this)
        rows_to_keep = torch.tensor(list(set(range(len(prebox_cp))) - rows_to_delete))
        prebox_cp = torch.index_select(prebox_cp, 0, rows_to_keep)
        clas_cp = torch.index_select(clas_cp, 0, rows_to_keep)

        return clas_cp, prebox_cp




    def training_step(self, batch, batch_idx):
        ## Reference: used gpt to get the code for appending and log
        
        images, bounding_boxes, indexes     = batch['images'], batch['bbox'], batch['index']
        images_re                           = torch.stack(images[:])
        gt,ground_coord                     = self.create_batch_truth(bounding_boxes, indexes, (800, 1088))  
        logits, bbox_regs                   = self.forward(images_re)

        loss, loss_c, loss_r = self.compute_loss(logits.to(device), bbox_regs.to(device), gt.to(device), ground_coord.to(device))

        self.log("train_loss",              loss,       prog_bar=True)
        self.log("train_class_loss",        loss_c,     prog_bar=True)
        self.log("train_regression_loss",   loss_r,     prog_bar=True)
        self.accumulated_train_losses.append(loss.detach())
        self.accumulated_train_class_losses.append(loss_c.detach())
        self.accumulated_train_regression_losses.append(loss_r.detach())

        return {"loss": loss, "train_class_loss": loss_c, "train_regression_loss": loss_r}

    def on_train_epoch_end(self):
        train_loss = torch.stack(self.accumulated_train_losses).mean().item()
        train_class_loss = torch.stack(self.accumulated_train_class_losses).mean().item()
        train_regression_loss = torch.stack(self.accumulated_train_regression_losses).mean().item()

        # Logging the mean values
        self.log("train_loss_epoch", train_loss, prog_bar=True, on_epoch=True)
        self.log("train_class_loss_epoch", train_class_loss, prog_bar=True, on_epoch=True)
        self.log("train_regression_loss_epoch", train_regression_loss, prog_bar=True, on_epoch=True)

        # Optionally append to a list for tracking over epochs
        self.train_losses.append((train_loss, train_class_loss, train_regression_loss))

        # Clear the lists for the next epoch
        self.accumulated_train_losses.clear()
        self.accumulated_train_class_losses.clear()
        self.accumulated_train_regression_losses.clear()

    def validation_step(self, batch, batch_idx):
        images, bounding_boxes, indexes     = batch['images'], batch['bbox'], batch['index']
        images_re                           = torch.stack(images[:])
        gt,ground_coord                     = self.create_batch_truth(bounding_boxes, indexes, (800, 1088))  
        logits, bbox_regs                   = self.forward(images_re)

        val_loss, loss_c, loss_r = self.compute_loss(logits.to(device), bbox_regs.to(device), gt.to(device), ground_coord.to(device))

        self.log("val_loss",              val_loss,   prog_bar=True)
        self.log("val_class_loss",        loss_c,     prog_bar=True)
        self.log("val_regression_loss",   loss_r,     prog_bar=True)
        # Append the losses to the lists
        self.accumulated_val_losses.append(val_loss)
        self.accumulated_val_class_losses.append(loss_c)
        self.accumulated_val_regression_losses.append(loss_r)

        return {"val_loss": val_loss, "val_class_loss": loss_c, "val_regression_loss": loss_r}
        
    def on_validation_epoch_end(self):
        val_loss = torch.stack(self.accumulated_val_losses).mean().item()
        val_class_loss = torch.stack(self.accumulated_val_class_losses).mean().item()
        val_regression_loss = torch.stack(self.accumulated_val_regression_losses).mean().item()

        # Logging the mean values
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_class_loss", val_class_loss, prog_bar=True)
        self.log("val_regression_loss", val_regression_loss, prog_bar=True)

        # Optionally append to a list for tracking (if you need it)
        self.val_losses.append((val_loss, val_class_loss, val_regression_loss))

        # Clear the lists for the next epoch
        self.accumulated_val_losses.clear()
        self.accumulated_val_class_losses.clear()
        self.accumulated_val_regression_losses.clear()


    def configure_optimizers(self):
        opt     = torch.optim.SGD(self.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.9)
        sched   = {"scheduler": torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[26, 32], gamma=0.1)}

        return {"optimizer": opt, "lr_scheduler": sched}


def plot_bounding_box(image, labels):
    # Create a figure and a single subplot
    fig, ax = plt.subplots(figsize=(30, 10))
    
    # Display the image
    ax.imshow(image)
    
    # Add rectangles for each label
    for label in labels:
        # Define a rectangle patch using label coordinates
        rectangle_s = patches.Rectangle((label[0], label[1]), label[2] - label[0], label[3] - label[1],
                                 linewidth=2, edgecolor='r', facecolor='none')
        
        # Add the rectangle to the Axes
        ax.add_patch(rectangle_s)
    
    # Display the plot
    plt.show()

    
# if __name__=="__main__":
#     ######################################################################################################################################
#     imgs_path   = '/content/drive/MyDrive/cis6800/FASTER_RCNN/hw3_mycocodata_img_comp_zlib.h5'
#     masks_path  = '/content/drive/MyDrive/cis6800/FASTER_RCNN/hw3_mycocodata_mask_comp_zlib.h5'
#     labels_path = '/content/drive/MyDrive/cis6800/FASTER_RCNN/hw3_mycocodata_labels_comp_zlib.npy'
#     bboxes_path = '/content/drive/MyDrive/cis6800/FASTER_RCNN/hw3_mycocodata_bboxes_comp_zlib.npy'
#     paths = [imgs_path, masks_path, labels_path, bboxes_path]














