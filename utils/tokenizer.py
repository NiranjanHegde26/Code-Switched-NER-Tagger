from transformers import XLMRobertaTokenizerFast, GPT2TokenizerFast
from config import read_config
config = read_config()

def tokenize_and_align_labels(examples, label_all_tokens = False, skip_index = -100):
    # The function core functionality is reused from Assignment 4. 
    # However, it is modified the main implementation available in Coli GitHub to handle the different sentence lengths and handling the different format of label as per the current dataset.
    # "padding=True is changed to padding="max_length" and 256 is defined as the maximum token length"
    # Reference: https://huggingface.co/docs/transformers/v4.15.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.__call__.padding
    
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(config["model_name"])
    tokenized_inputs = tokenizer(examples["words"], truncation = True, is_split_into_words = True, padding = 'max_length', max_length= 256)
    labels = []
    for i, label in enumerate(examples["ner"]):
        label = [config["labels"].index(x) for x in label] # String Labels are replaced with its indices from the config.
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids : list[int] = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(skip_index)

            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])

            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else skip_index)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
