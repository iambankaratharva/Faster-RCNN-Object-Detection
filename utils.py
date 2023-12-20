import numpy as np
import torch
from functools import partial
import torchvision
from sklearn import metrics

def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# This function compute the IOU between two set of boxes 
def IOU(boxA, boxB):
    iou = torchvision.ops.box_iou(boxA, boxB.to(device))
    return iou

# This function decodes the output of the box head that are given in the [t_x,t_y,t_w,t_h] format
# into box coordinates where it return the upper left and lower right corner of the bbox
# Input:
#       regressed_boxes_t: (total_proposals,4) ([t_x,t_y,t_w,t_h] format)
#       flatten_proposals: (total_proposals,4) ([x1,y1,x2,y2] format)
# Output:
#       box: (total_proposals,4) ([x1,y1,x2,y2] format)
def output_decodingd(regressed_boxes_t,flatten_proposals, device='cpu'):

        #Reference: used gpt for shape adjustemnts, debgging in gooogle and handout instructions
        #Reference: https://stackoverflow.com/questions/509211/how-slicing-in-python-works

    # did a step by step study, check out, its working
    
    # Calculate the center x-coordinate of the proposals by averaging the left and right x-coordinates.
    left_x_coordinates = flatten_proposals[:, 0]  # Extract left x-coordinates from flatten proposals.
    right_x_coordinates = flatten_proposals[:, 2]  # Extract right x-coordinates from flatten proposals.
    prop_x = (left_x_coordinates + right_x_coordinates) / 2.0  # Compute the mean x-coordinate.

    # Calculate the center y-coordinate of the proposals by averaging the top and bottom y-coordinates.
    top_y_coordinates = flatten_proposals[:, 1]  # Extract top y-coordinates from flatten proposals.
    bottom_y_coordinates = flatten_proposals[:, 3]  # Extract bottom y-coordinates from flatten proposals.
    prop_y = (top_y_coordinates + bottom_y_coordinates) / 2.0  # Compute the mean y-coordinate.

    # Calculate the width of the proposals by subtracting the left x-coordinate from the right x-coordinate.
    prop_w = right_x_coordinates - left_x_coordinates  # Compute the width.

    # Calculate the height of the proposals by subtracting the top y-coordinate from the bottom y-coordinate.
    prop_h = bottom_y_coordinates - top_y_coordinates  # Compute the height.


    prop_xywh = torch.vstack((prop_x, prop_y, prop_w, prop_h))

    # Transpose the resulting 2D tensor to ensure proper shape.
    # This switches the rows and columns such that what were the rows (x, y, w, h)
    # now become columns, with each column corresponding to a different property for the same proposal.
    prop_xywh = prop_xywh.T

    # Compute the x coordinate.
    # Multiply the normalized offsets by the proposal width and add to the proposal x-coordinate.
    offset_x_normalized = regressed_boxes_t[:, 0]  # Normalized offset for x from regressed boxes.
    proposal_width = prop_xywh[:, 2]  # Width of the proposal.
    proposal_x_center = prop_xywh[:, 0]  # x-coordinate of the proposal's center.
    x = (offset_x_normalized * proposal_width) + proposal_x_center

    # Compute the y coordinate.
    # Multiply the normalized offsets by the proposal height and add to the proposal y-coordinate.
    offset_y_normalized = regressed_boxes_t[:, 1]  # Normalized offset for y from regressed boxes.
    proposal_height = prop_xywh[:, 3]  # Height of the proposal.
    proposal_y_center = prop_xywh[:, 1]  # y-coordinate of the proposal's center.
    y = (offset_y_normalized * proposal_height) + proposal_y_center

    # Compute the width.
    # Multiply the proposal width by the exponent of the regressed box width offset.
    width_offset = regressed_boxes_t[:, 2]  # Width offset from regressed boxes.
    w = proposal_width * torch.exp(width_offset)

    # Compute the height.
    # Multiply the proposal height by the exponent of the regressed box height offset.
    height_offset = regressed_boxes_t[:, 3]  # Height offset from regressed boxes.
    h = proposal_height * torch.exp(height_offset)

    # Calculate the top-left x-coordinate (x1) of the box.
    # We subtract half the width from the x center coordinate to get the top-left corner.
    half_width = w / 2.0  # Calculate half of the width to find the extent from the center to the edge.
    x1 = x - half_width  # Subtracting half the width from the center gives the left boundary.

    # Calculate the top-left y-coordinate (y1) of the box.
    # We subtract half the height from the y center coordinate to get the top-left corner.
    half_height = h / 2.0  # Calculate half of the height to find the extent from the center to the edge.
    y1 = y - half_height  # Subtracting half the height from the center gives the upper boundary.

    # Calculate the bottom-right x-coordinate (x2) of the box.
    # We add half the width to the x center coordinate to get the bottom-right corner.
    x2 = x + half_width  # Adding half the width to the center gives the right boundary.

    # Calculate the bottom-right y-coordinate (y2) of the box.
    # We add half the height to the y center coordinate to get the bottom-right corner.
    y2 = y + half_height  # Adding half the height to the center gives the lower boundary.

    # Prepare the individual coordinates for stacking.
    # Since x1, y1, x2, y2 represent the coordinates of the bounding box, we'll combine them.

    # Create a 2D tensor for each coordinate with an added dimension for concatenation.
    x1_2d = x1.unsqueeze(0)  # Add a new axis to x1 to make it a 2D tensor.
    y1_2d = y1.unsqueeze(0)  # Add a new axis to y1 to make it a 2D tensor.
    x2_2d = x2.unsqueeze(0)  # Add a new axis to x2 to make it a 2D tensor.
    y2_2d = y2.unsqueeze(0)  # Add a new axis to y2 to make it a 2D tensor.

    # Stack the 2D tensors vertically (along rows).
    # After concatenating, we have a 2D tensor with each row being one of the coordinates (x1, y1, x2, y2).
    box_coordinates_stacked = torch.cat((x1_2d, y1_2d, x2_2d, y2_2d), dim=0)

    # Transpose the 2D tensor to switch rows and columns.
    # The resulting tensor has the shape (N, 4) where N is the number of boxes,
    # and 4 represents the coordinates x1, y1, x2, y2 for each box.
    box = box_coordinates_stacked.T


    return box

        #Reference: used gpt for shape adjustemnts, debgging in gooogle and handout instructions
        #Reference: https://stackoverflow.com/questions/509211/how-slicing-in-python-works

def filter_by_class(predictions, targets, target_class):
    filtered_predictions = [pred[pred[:, 0] == target_class] for pred in predictions]
    filtered_targets = [tar[tar[:, 0] == target_class] for tar in targets]
    return filtered_predictions, filtered_targets

def calculate_iou_and_matches(predictions, targets, iou_threshold=0.5):
    tp_fp_array = []
    total_true_boxes = 0
    for pred, tar in zip(predictions, targets):
        if len(pred) and len(tar):
            iou = torchvision.ops.box_iou(pred[:, 2:], tar[:, 1:])
            matched_targets = []
            for i, iou_row in enumerate(iou):
                iou_max, matched_idx = torch.max(iou_row, 0)
                if iou_max > iou_threshold:
                    if matched_idx not in matched_targets:
                        tp_fp_array.append(torch.tensor([pred[i, 1], 1, 0]))
                        matched_targets.append(matched_idx)
                    else:
                        tp_fp_array.append(torch.tensor([pred[i, 1], 0, 1]))
                else:
                    tp_fp_array.append(torch.tensor([pred[i, 1], 0, 1]))
        total_true_boxes += len(tar)
    return tp_fp_array, total_true_boxes

def compute_precision_recall(tp_fp_array, total_true_boxes):
    if not tp_fp_array:
        return torch.tensor([0, 0]), torch.tensor([0, 0])

    sorted_detections = torch.stack(tp_fp_array)[torch.argsort(torch.stack(tp_fp_array)[:, 0], descending=True)]
    precision_vals, recall_vals = [], []
    true_positives = 0
    for idx, detection in enumerate(sorted_detections):
        if detection[1] == 1:
            true_positives += 1
        precision_vals.append(torch.tensor(true_positives / (idx + 1)))
        recall_vals.append(torch.tensor(true_positives / total_true_boxes))

    recall_tensor, precision_tensor = torch.stack(recall_vals), torch.stack(precision_vals)

    if recall_tensor.size(0) == 1 or precision_tensor.size(0) == 1:
        return torch.hstack([recall_tensor, torch.tensor([0])]), torch.hstack([precision_tensor, torch.tensor([0])])

    return recall_tensor, precision_tensor

def precision_recall_curve(predictions, targets, target_class):
    filtered_predictions, filtered_targets = filter_by_class(predictions, targets, target_class)
    tp_fp_array, total_true_boxes = calculate_iou_and_matches(filtered_predictions, filtered_targets)
    return compute_precision_recall(tp_fp_array, total_true_boxes)




def average_precision(predictions, targets, target_class):
    # Compute the precision-recall curve for the given target class.
    # This function returns the recall and precision values at different trhesholds.
    recall_values, precision_values = precision_recall_curve(predictions, targets, target_class)

    # The AUC function from the metrics module calculates the area under a curve represented by x (recall) and y (precision).
    average_precision_score = metrics.auc(recall_values, precision_values)

    # Return the computed Average Precision score.
    return average_precision_score


def mean_average_precision(predictions, targets):

                #Reference: used gpt for shape adjustemnts, debgging in gooogle and handout instructions
        #Reference: https://stackoverflow.com/questions/509211/how-slicing-in-python-works, https://www.interviewkickstart.com/learn/the-append-function-in-python

    # Define the list of classes for which we want to calculate average precision.
    classes = [1, 2, 3]
    
    # Initialize an empty list to store the average precision for each class.
    ap_scores = []
    
    # Calculate average precision for each class.
    for class_id in classes:
        # Use the previously defined average_precision function to compute the AP for the current class.
        ap = average_precision(predictions, targets, class_id)
        
        # Append the AP score to the list of AP scores.
        ap_scores.append(ap)
    
    # Return the list of average precision scores, one for each class.
    return ap_scores