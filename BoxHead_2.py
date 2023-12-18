import torch
import torchvision
import numpy as np

from torch import nn
from dataset import *
from utils import *
from pretrained_models import *


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class BoxHead(torch.nn.Module):
    def __init__(self,Classes=3,P=7):
        super(BoxHead, self).__init__()
        self.C=Classes
        self.P=P
        self.device = device
        self.intermediate_layer = nn.Sequential(nn.Linear(256*self.P*self.P, 1024),
                                                nn.ReLU(),
                                                nn.Linear(1024, 1024),
                                                nn.ReLU())
        self.classifier         = nn.Sequential(nn.Linear(1024, self.C+1))
        self.regressor          = nn.Sequential(nn.Linear(1024, 4*self.C))
    

    def create_ground_truth(self, proposals, gt_labels, bbox):
        '''
        This function assigns to each proposal either a ground truth box or the background class (we assume background class is 0)
        Input:
            proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
            gt_labels: list:len(bz) {(n_obj)}
            bbox: list:len(bz){(n_obj, 4)}
        Output: (make sure the ordering of the proposals are consistent with MultiScaleRoiAlign)
            labels: (total_proposals,1) (the class that the proposal is assigned)
            regressor_target: (total_proposals,4) (target encoded in the [t_x,t_y,t_w,t_h] format)
        '''
        def calculate_regressor_targets(image_proposals, gt_boxes, gt_labels_image):
            iou = IOU(image_proposals, gt_boxes)
            sorted_iou = torch.max(iou, dim=1)

            iou_match_indices = torch.where(sorted_iou[0] > 0.5)[0]
            labels = torch.zeros(len(image_proposals))
            regressor_targets_batch = torch.zeros((len(image_proposals), 4))

            for j in range(len(sorted_iou[1])):
                if j in iou_match_indices:
                    labels[j] = gt_labels_image.tolist()[sorted_iou[1][j]]

                    x_center_gt, y_center_gt, width_gt, height_gt = (gt_boxes[sorted_iou[1][j]][0] + gt_boxes[sorted_iou[1][j]][2]) / 2, (gt_boxes[sorted_iou[1][j]][1] + gt_boxes[sorted_iou[1][j]][3]) / 2, (gt_boxes[sorted_iou[1][j]][2] - gt_boxes[sorted_iou[1][j]][0]), (gt_boxes[sorted_iou[1][j]][3] - gt_boxes[sorted_iou[1][j]][1])

                    x_center_proposal, y_center_proposal, width_proposal, height_proposal = (image_proposals[j][0] + image_proposals[j][2]) / 2, (image_proposals[j][1] + image_proposals[j][3]) / 2, (image_proposals[j][2] - image_proposals[j][0]), (image_proposals[j][3] - image_proposals[j][1])

                    tx, ty, tw, th = (x_center_gt - x_center_proposal) / width_proposal, (y_center_gt - y_center_proposal) / height_proposal, torch.log(width_gt / width_proposal), torch.log(height_gt / height_proposal)

                    regressor_targets_batch[j, 0:4] = torch.tensor([tx, ty, tw, th])

            return labels.reshape(-1, 1), regressor_targets_batch

        all_labels, all_regressor_targets = list(), list()

        for batch_idx, (gt_boxes, image_proposals, gt_labels_image) in enumerate(zip(bbox, proposals, gt_labels)):
            labels, regressor_targets_batch = calculate_regressor_targets(image_proposals, gt_boxes, gt_labels_image)
            all_labels.append(labels)
            all_regressor_targets.append(regressor_targets_batch)

        return torch.vstack(all_labels[:]), torch.vstack(all_regressor_targets[:])



    def MultiScaleRoiAlign(self, fpn_feat_list, proposals, P=7):
        '''
        This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
        a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
        Input:
                fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
                proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
                P: scalar
        Output:
                feature_vectors: (total_proposals, 256*P*P)  (make sure the ordering of the proposals are the same as the ground truth creation)
        '''
        #Reference: used gpt for shape adjustemnts, debgging in gooogle
        #Reference: https://stackoverflow.com/questions/509211/how-slicing-in-python-works
        def compute_k_values(proposal):
            k_values = torch.floor(4 + torch.log2(1.0 / 224 * torch.sqrt((proposal[:, 2] - proposal[:, 0]) * (proposal[:, 3] - proposal[:, 1]))))
            k_values = torch.clamp(k_values, 2, 5)
            return [int(item) for item in k_values]

        def scale_proposal(proposal, k_value):
            rescale_x = 1088 / fpn_feat_list[k_value - 2].shape[3]
            rescale_y = 800 / fpn_feat_list[k_value - 2].shape[2]
            proposal_scaled = proposal / torch.tensor([rescale_x, rescale_y, rescale_x, rescale_y], device=proposal.device)
            return proposal_scaled

        feature_vectors = []
        for i in range(len(proposals)):
            k_values = compute_k_values(proposals[i])
            for j in range(len(proposals[i])):
                k_value = k_values[j]
                proposal_scaled = scale_proposal(proposals[i][j], k_value)

                proposal_feature_vector = torchvision.ops.roi_align(fpn_feat_list[k_value - 2][i].unsqueeze(0), [proposal_scaled.unsqueeze(0).to(device)], output_size=(P, P), spatial_scale=1.0)
                feature_vectors.append(proposal_feature_vector.reshape(-1))
        feature_vectors = torch.stack(feature_vectors, dim=0)

        return feature_vectors

    def non_max_suppression(self, boxes_batch, scores_batch, labels_batch, keep_post_nms=20):
        '''
        boxes_batch: List of bounding boxes for each image in the batch (x1, y1, x2, y2 format).
        scores_batch: List of scores for each bounding box.
        labels_batch: List of labels corresponding to each bounding box.
        keep_post_nms: The maximum number of boxes to keep after non-maximum suppression (NMS).
        '''

        #Reference: used gpt for shape adjustemnts, debugging in google
        #Reference: https://stackoverflow.com/questions/509211/how-slicing-in-python-works
        iou_threshold = 0.5
        final_selected_boxes, final_selected_labels, final_selected_scores = list(), list(), list()

        for image_index, labels_per_image in enumerate(labels_batch):
            labels_per_image = labels_per_image.cpu().detach().numpy()
            label_matrix = np.vstack(np.stack(labels_per_image, axis=-1))

            selected_classes, selected_boxes, selected_scores = list(), list(), list()

            for class_id in range(1, 4):
                class_indices = np.where(label_matrix[:, 0] == class_id)[0]
                class_labels = label_matrix[class_indices]

                if class_labels.shape[0] != 0:
                    class_scores = scores_batch[image_index][class_indices].detach().cpu()
                    class_boxes = boxes_batch[image_index][class_indices].detach().cpu()
                    class_iou = torchvision.ops.box_iou(class_boxes.clone(), class_boxes.clone())

                    keep_indices = []
                    for i in range(class_iou.shape[0]):
                        if i not in keep_indices:
                            for j in range(i + 1, class_iou.shape[1]):
                                if class_iou[i, j] > iou_threshold:
                                    keep_indices.append(j)

                    class_boxes = np.delete(class_boxes, list(set(keep_indices)), axis=0)
                    class_scores = np.delete(class_scores, list(set(keep_indices)), axis=0)

                    if len(class_boxes) > keep_post_nms:
                        class_boxes = class_boxes[:keep_post_nms]
                        class_scores = class_scores[:keep_post_nms]

                    selected_classes.extend([class_id] * len(class_boxes))
                    selected_boxes.append(class_boxes)
                    selected_scores.append(class_scores)

            if selected_boxes:
                final_selected_label = torch.from_numpy(np.hstack(selected_classes).astype(int))
                final_selected_score = torch.from_numpy(np.hstack(selected_scores))
                final_selected_boxe = torch.from_numpy(np.vstack(selected_boxes))
                final_selected_labels.append(final_selected_label)
                final_selected_scores.append(final_selected_score)
                final_selected_boxes.append(final_selected_boxe)

        return final_selected_boxes, final_selected_labels, final_selected_scores


    def postprocess_detections(self, class_logits, box_regression, proposals, conf_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=50):
        '''
        This function does the post processing for the results of the Box Head for a batch of images
        Use the proposals to distinguish the outputs from each image
        Input:
              class_logits: (total_proposals,(C+1))
              box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
              proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN [x1,y1,x2,y2] format)
              conf_thresh: scalar
              keep_num_preNMS: scalar (number of boxes to keep pre NMS)
              keep_num_postNMS: scalar (number of boxes to keep post NMS)
        Output:
              boxes: list:len(bz){(post_NMS_boxes_per_image,4)}  ([x1,y1,x2,y2] format)
              scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
              labels: list:len(bz){(post_NMS_boxes_per_image)}   (top class of each regressed box)
        '''
        def process_single_image(image_proposals, image_box_regression, image_class_logits):
            class_scores, class_labels = torch.max(image_class_logits, dim=1)
            class_labels = class_labels.to(torch.int32)
            class_labels = class_labels - 1
            non_bg_labels = torch.where(class_labels >= 0)[0]

            if len(non_bg_labels) != 0:
                class_labels = class_labels[non_bg_labels]
                image_box_regression = image_box_regression[non_bg_labels]
                image_box_regression = torch.stack([image_box_regression[i, x*4:(x+1)*4] for i, x in enumerate(class_labels)])
                boxes_x1y1x2y2 = output_decodingd(image_box_regression, image_proposals[non_bg_labels])

                valid_boxes_idx = torch.where((boxes_x1y1x2y2[:, 0] >= 0) & (boxes_x1y1x2y2[:, 1] >= 0) &
                                            (boxes_x1y1x2y2[:, 2] < 1088) & (boxes_x1y1x2y2[:, 3] < 800))[0]

                valid_boxes = boxes_x1y1x2y2[valid_boxes_idx]
                class_logits_image = class_logits[valid_boxes_idx]

                keep_boxes_thresh_idx = torch.where(class_logits_image[:, 1:] > conf_thresh)
                valid_boxes_after_thresh = valid_boxes[keep_boxes_thresh_idx[0]]
                class_logits_image = class_logits_image[keep_boxes_thresh_idx[0]]
                class_scores, class_labels = torch.max(class_logits_image, dim=1)

                sorted_scores, sorted_scores_idx = torch.sort(class_scores, descending=True)

                if len(sorted_scores) > keep_num_preNMS:
                    sorted_scores = sorted_scores[:keep_num_preNMS]
                    sorted_scores_idx = sorted_scores_idx[:keep_num_preNMS]
                    class_labels = class_labels[sorted_scores_idx]
                    boxes_sorted_image = valid_boxes_after_thresh[sorted_scores_idx]
                    if len(class_labels) > 0:
                        return sorted_scores, class_labels, boxes_sorted_image
                else:
                    sorted_scores = sorted_scores
                    sorted_scores_idx = sorted_scores_idx
                    class_labels = class_labels[sorted_scores_idx]
                    boxes_sorted_image = valid_boxes_after_thresh[sorted_scores_idx]
                    if len(class_labels) > 0:
                        return sorted_scores, class_labels, boxes_sorted_image
            return None, None, None

        scores_pre_NMS_batch = []
        labels_pre_NMS_batch = []
        boxes_pre_NMS_batch = []

        for i, each_image in enumerate(proposals):
            boxes_image = box_regression[i * 200:(i + 1) * 200]
            class_logits_image = class_logits[i * 200:(i + 1) * 200]
            sorted_scores, class_labels, boxes_sorted_image = process_single_image(each_image, boxes_image, class_logits_image)
            if sorted_scores is not None:
                scores_pre_NMS_batch.append(sorted_scores)
                labels_pre_NMS_batch.append(class_labels)
                boxes_pre_NMS_batch.append(boxes_sorted_image)

        if type(labels_pre_NMS_batch) == list and len(labels_pre_NMS_batch) != 0:
            if torch.stack(labels_pre_NMS_batch[:]).shape[1] != 0:
                final_boxes, final_label, final_scores = self.non_max_suppression(boxes_pre_NMS_batch, scores_pre_NMS_batch,
                                                                                labels_pre_NMS_batch)
            else:
                final_boxes = [torch.tensor([]) for x in range(len(proposals))]
                final_label = [torch.tensor([]) for x in range(len(proposals))]
                final_scores = [torch.tensor([]) for x in range(len(proposals))]
        else:
            final_boxes = [torch.tensor([]) for x in range(len(proposals))]
            final_label = [torch.tensor([]) for x in range(len(proposals))]
            final_scores = [torch.tensor([]) for x in range(len(proposals))]

        return final_boxes, final_scores, final_label

    def postprocess_detections_map_scores(self, class_logits, box_regression, proposals, conf_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=50):
        '''
        Post process detections by mapping scores, applying NMS
        
        Inputs:
            class_logits: predicted class logits for each proposal, shape (num_proposals, num_classes + 1)
            box_regression: predicted bounding box regression deltas, shape (num_proposals, 4 * num_classes)
            proposals: list of proposal boxes from RPN, length batch_size
            conf_thresh: confidence threshold for selecting predictions
            keep_num_pre_nms: number of predictions to keep before applying NMS
            keep_num_post_nms: number of predictions to keep after applying NMS
            
        Returns:
            final_labels: list of labels for kept predictions, length batch_size
            final_scores: list of scores for kept predictions, length batch_size
            final_boxes: list of boxes for kept predictions, length batch_size
        '''
        #Reference: used gpt for shape adjustemnts, debgging in gooogle and handout instructions
        #Reference: https://stackoverflow.com/questions/509211/how-slicing-in-python-works
        
        processed_scores, processed_labels, processed_boxes = list(), list(), list()
        
        for img_idx, img_proposals in enumerate(proposals):
            img_box_deltas = box_regression[img_idx * 200 : (img_idx + 1) * 200]
            img_class_logits = class_logits[img_idx * 200 : (img_idx + 1) * 200]
            max_class_scores, argmax_class_labels = torch.max(img_class_logits, dim=1)
            
            # Make class labels 0-indexed
            argmax_class_labels = argmax_class_labels.to(torch.int32) - 1
            
            # Get indices of positive classes
            pos_class_indices = torch.where(argmax_class_labels >= 0)[0]
            
            if len(pos_class_indices) != 0:
                # Get predictions for positive classes only
                argmax_class_labels = argmax_class_labels[pos_class_indices]
                
                # Rearrange box deltas for positive classes
                rearranged_box_deltas = []
                for pred_idx, class_idx in enumerate(argmax_class_labels):
                    delta = img_box_deltas[pos_class_indices][pred_idx, class_idx * 4 : (class_idx + 1) * 4]
                    rearranged_box_deltas.append(delta)
                img_box_deltas = torch.stack(rearranged_box_deltas)
                
                # Decode boxes w.r.t proposals
                decoded_boxes = output_decodingd(img_box_deltas, img_proposals[pos_class_indices])

                # Filter out invalid boxes
                valid_boxes_idx = torch.where((decoded_boxes[:,0] >= 0) & (decoded_boxes[:,1] >= 0) &  
                                            (decoded_boxes[:,2] < 1088) & (decoded_boxes[:,3] < 800))[0]
                
                valid_boxes = decoded_boxes[valid_boxes_idx]
                class_logits = class_logits[valid_boxes_idx]

                # Filter low-scoring boxes
                keep_idx = torch.where(class_logits[:,1:] > conf_thresh)[0]
                valid_boxes = valid_boxes[keep_idx]
                class_logits = class_logits[keep_idx]
                
                # Get top scoring class
                max_class_scores, argmax_class_labels = torch.max(class_logits, dim=1)
                # Sort scores in descending order
                sorted_scores, sort_idx = torch.sort(max_class_scores, descending=True)
                # Truncate predictions
                if len(sorted_scores) > keep_num_preNMS:
                    sorted_scores = sorted_scores[:keep_num_preNMS]
                    sort_idx = sort_idx[:keep_num_preNMS]
                
                # Get labels for sorted predictions (check whether thisis corect, working, need to debug), edit --> works
                #argmax_class_labels = argmax_class_labels[sort_idx[:,1]]
                argmax_class_labels = argmax_class_labels[sort_idx]
                sorted_boxes = valid_boxes[sort_idx]
                
                if len(argmax_class_labels) > 0: processed_scores.append(sorted_scores), processed_labels.append(argmax_class_labels), processed_boxes.append(sorted_boxes)
                
            else:
                continue
                
        # Run NMS on predictions from all images
        final_boxes, final_labels, final_scores = self.non_max_suppression(processed_boxes, processed_scores, processed_labels)
        
        return final_labels, final_scores, final_boxes



    def compute_loss(self,class_logits, box_preds, labels, regression_targets,l=1,effective_batch=150):
        '''
        Compute the total loss of the classifier and the regressor
        Input:
             class_logits: (total_proposals,(C+1)) (as outputed from forward, not passed from softmax so we can use CrossEntropyLoss)
             box_preds: (total_proposals,4*C)      (as outputed from forward)
             labels: (total_proposals,1)
             regression_targets: (total_proposals,4)
             l: scalar (weighting of the two losses)
             effective_batch: scalar
        Outpus:
             loss: scalar
             loss_class: scalar
             loss_regr: scalar
        '''

        #Reference: used gpt for shape adjustemnts, debgging in gooogle and handout instructions
        #Reference: https://stackoverflow.com/questions/509211/how-slicing-in-python-works
        # Get indexes of positive and negative samples
        def get_positive_negative_indexes(labels):
            positive_indices = torch.where(labels > 0)[0]
            negative_indices = torch.where(labels == 0)[0]
            return positive_indices, negative_indices

        # Sample a subset of positive and negative indexes
        def sample_indexes(positive_indices, negative_indices):
            if labels.shape[0] >= (3 * effective_batch) / 4:
                positive_count = int((3 * effective_batch) / 4)
                negative_count = int(effective_batch - positive_count)
            else:
                positive_count = positive_indices.shape[0]
                negative_count = effective_batch - positive_count
            return positive_count, negative_count

        # Shuffle and select samples
        def shuffle_and_select_samples(positive_indices, negative_indices, positive_count, negative_count):
            shuffled_positive_indices = torch.randperm(positive_indices.shape[0])
            shuffled_negative_indices = torch.randperm(negative_indices.shape[0])
            final_positive_indices = positive_indices[shuffled_positive_indices][:positive_count]
            final_negative_indices = negative_indices[shuffled_negative_indices][:negative_count]
            return final_positive_indices, final_negative_indices

        # Compute classification loss
        def compute_classification_loss(class_logits, labels, final_positive_indices, final_negative_indices, criterion):
            final_labels_positive = labels[final_positive_indices]
            final_labels_negative = labels[final_negative_indices]
            final_logits_positive = class_logits[final_positive_indices]
            final_logits_negative = class_logits[final_negative_indices]

            final_labels_positive_one_hot = torch.nn.functional.one_hot(final_labels_positive.to(torch.int64).flatten(), num_classes=4)
            final_labels_negative_one_hot = torch.nn.functional.one_hot(final_labels_negative.to(torch.int64).flatten(), num_classes=4)

            loss_class_positive = criterion(final_logits_positive, final_labels_positive_one_hot.to(torch.float32).to(device))
            loss_class_negative = criterion(final_logits_negative, final_labels_negative_one_hot.to(torch.float32).to(device))

            loss_class = loss_class_positive + loss_class_negative
            return loss_class

        # Compute regression loss
        def compute_regression_loss(box_predictions, regression_targets, final_positive_indices):
            predicted_regressions = box_predictions[final_positive_indices]
            predicted_regressions = predicted_regressions.reshape((predicted_regressions.shape[0], self.C, 4))
            target_regression_positive = regression_targets[final_positive_indices]

            loss_regression = 0
            smooth_l1_loss = nn.SmoothL1Loss(reduction='sum')

            for i, label_in in enumerate(labels[final_positive_indices]):
                loss_regression += smooth_l1_loss(predicted_regressions[i, int(label_in) - 1, :], target_regression_positive[i].to(device))

            return loss_regression

        positive_indices, negative_indices = get_positive_negative_indexes(labels)
        positive_count, negative_count = sample_indexes(positive_indices, negative_indices)
        final_positive_indices, final_negative_indices = shuffle_and_select_samples(positive_indices, negative_indices, positive_count, negative_count)

        criterion = nn.CrossEntropyLoss()
        loss_class = compute_classification_loss(class_logits, labels, final_positive_indices, final_negative_indices, criterion)
        loss_regr = compute_regression_loss(box_preds, regression_targets, final_positive_indices)
        loss = loss_class + (l * loss_regr)

        return loss, loss_class, loss_regr

    def forward(self, feature_vectors, evaluate = False):
        '''
        # Outputs:
        #        class_logits: (total_proposals,(C+1)) (we assume classes are C classes plus background, notice if you want to use
        #                                               CrossEntropyLoss you should not pass the output through softmax here)
        #        box_pred:     (total_proposals,4*C)
        '''

        #Reference: used gpt for shape adjustemnts, debgging in gooogle and handout instructions
        #Reference: https://stackoverflow.com/questions/509211/how-slicing-in-python-works

        x            = self.intermediate_layer(feature_vectors)
        class_logits = self.classifier(x)
        box_pred     = self.regressor(x)

        if evaluate:
            softmax = torch.nn.Softmax(dim = 1)
            class_logits = softmax(class_logits)

        return class_logits, box_pred