import torch
from transformers import XLMRobertaModel
import torch.nn as nn

from config import read_config
config = read_config()

class XLMRoBERTaNERTagger(nn.Module):
    """
        This class combines the pretrained XLM RoBERTa model with the three Linear-Layer Sequential Model.
        All the hidden state output of XLM RoBERTa is redirected to the linear layer whose output is equal to the total number of classes.
        To avoid the overhead of training process, all the parameters of XLM RoBERTa is set to param.requires_grad = False
        Here we have 3 linear layers separated by non linearity using ReLU. 
        The hidden state of the first linear model is predefined, and for the later layers, the hidden states are halved out.
    """
    def __init__(self):
        super(XLMRoBERTaNERTagger, self).__init__()
        self.classes = config["labels"]
        self.xlm_roberta = XLMRobertaModel.from_pretrained(config["model_name"])
        for name, param in self.xlm_roberta.named_parameters():
            param.requires_grad = False

        # Custom layer
        hidden_state_size = config["hidden_state_size"]
        self.multi_linear_layer = nn.Sequential(
            nn.Linear(self.xlm_roberta.config.hidden_size, hidden_state_size),
            nn.ReLU(),
            nn.Linear(hidden_state_size, int(hidden_state_size/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_state_size/2), len(self.classes)),
        )
        self.model_name = f"{type(self.xlm_roberta).__name__}-{type(self.multi_linear_layer).__name__}-POS-Tagger"

    def forward(self, input_ids, attention_mask, labels = None):
        # Referred from https://huggingface.co/transformers/v2.9.1/model_doc/bert.html#transformers.BertModel.forward
        roberta_output = self.xlm_roberta(input_ids = input_ids, attention_mask = attention_mask)

        # Referred from https://huggingface.co/docs/transformers/main_classes/output
        last_hidden_states_of_roberta = roberta_output.last_hidden_state

        # Pass the last hidden state to the linear laye
        # Last hidden state shape is (batch_size, max_sequence_length, dimensionality of the hidden states of XLM RoBERTa model)
        # Example: (256, 128, 768)
        logits = self.multi_linear_layer(last_hidden_states_of_roberta)
        return logits
