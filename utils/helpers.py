# Some helper functions below
import torch
import torch.nn as nn
from itertools import chain
from sklearn.metrics import f1_score

from config import read_config
config = read_config()

def convert_label_tensor_to_list(label_tensor):
    """
    This function converts  list of tensors into flattened list for the calculation of metrics.

    Args:
        label_tensor (List(Tensor)): List of tensors having the labels
    Returns:
        tensor_as_list (List): Flattened list of int
    """
    tensor_as_list = []
    for ele in label_tensor:
        flat_tensor = ele.view(-1) # Flatten the tensor
        tensor_as_list.append(flat_tensor.tolist())
        
    return list(chain(*tensor_as_list)) # Flatten the list and return
  
def compute_micro_f1(gold_labels, prediction):
    """
    Given the list of labels as tensor, this function calculates Micro F1 score out of it.
    Since we append -100 to the tokens that do not have any labels, any gold labels containing -100 is discarded.
    Args:
        gold_labels (List(Tensor)): List of predicted labels for the given input as Tensor
        prediction (List(Tensor)): List of actual labels for the given input as Tensor

    Returns:
        micro_f1_score (int): Micro F1 score
    """
    gold_labels_as_list = convert_label_tensor_to_list(gold_labels)
    prediction_as_list  = convert_label_tensor_to_list(prediction)
    gold_labels = [label for label in gold_labels_as_list if label != -100] # Ignore all the labels with -100
    prediction = [pred_label for i, pred_label in enumerate(prediction_as_list) if gold_labels_as_list[i] != -100] # Ignore all the labels with -100
    micro_f1_score = f1_score(gold_labels, prediction, average='micro')
    return micro_f1_score

def compute_accuracy(prediction, gold_labels):
    """
    Given the predicted labels and the true labels, this function calculates the accuracy of the prediction process.
    Since we append -100 to the tokens that do not have any labels, any gold labels containing -100 is discarded.

    Args:
        prediction (Tensor): Predicted labels for the given input as Tensor
        gold_labels (Tensor): Actual labels for the given input as Tensor

    Returns:
        accuracy (int): Accuracy score of the predicted data to the actual data 
    """
    flattened_predictions = convert_label_tensor_to_list(prediction)
    flattened_gold_labels = convert_label_tensor_to_list(gold_labels)
    # Discard -100 index
    correct_predictions = sum (1 for true, pred in zip(flattened_gold_labels, flattened_predictions) if true >= 0 and true == pred)
    total_labels = len([x for x in flattened_gold_labels if x >= 0])
    accuracy = correct_predictions / total_labels
    return accuracy


def optimizer_and_loss_function_loader(model):
    """
    This function returns the optimizer and the loss function needed during the training.
    Args:
        model (XLMRoBERTaNERTagger): _description_

    Returns:
        optimizer (adam.Adam): Instance of ADAM optimizer
        loss (loss.CrossEntropyLoss): Instance of Cross entropy loss function
    """
    optimizer = torch.optim.Adam(model.multi_linear_layer.parameters(), lr = config["learning_rate"])
    loss_function = nn.CrossEntropyLoss(ignore_index = -100, reduction = "mean")
    
    return optimizer,loss_function 
