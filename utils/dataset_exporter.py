from config import read_config
from utils.data_generator import SentenceGenerator
import os
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
from itertools import chain
import traceback
import pandas as pd
import numpy as np

config = read_config()

class DataSetLoader:
    """
    This class is responsible for:
        a. Loading the training and evaluation dataset from Hugging face. (https://huggingface.co/datasets/lince/viewer/ner_spaeng)
        b. Filtering the sentences based on langauge
        c. Filtering the CS sentences
        d. Initate synonym replacement for English and Spanish sentences.
        e. Generating a new dataset after the synonym replacement.  
    """
    def __init__(self):
        self.augmented_dataset = DatasetDict()
        
    def filter_code_switched(Self, dataset):
        """
        Filter function to filter all CS sentences from the original dataset.
        """
        language_tags = set(dataset['lid'])
    
        # Check if 'lang1' is present and 'lang2' is not present
        lang1_present = 'lang1' in language_tags
        lang2_present = 'lang2' in language_tags
        
        return lang1_present and lang2_present
    
    def filter_eng(self, dataset):
        """
        Filter function to filter all English-only sentences from the original dataset.
        """
        language_tags = set(dataset['lid'])
    
        # Check if 'lang1' is present and 'lang2' is not present
        lang1_present = 'lang1' in language_tags
        lang2_not_present = 'lang2' not in language_tags
        
        # Check if remaining other tags are present
        others_present = ('ne' in language_tags) or ('others' in language_tags) or ('fw' in language_tags) or ('ambiguous' in language_tags) or ('mixed' in language_tags) or ('unk' in language_tags)
        
        # Return True if 'lang1' is present, 'lang2' is not present, and other tags is present
        return lang1_present and lang2_not_present and others_present
    
    def filter_esp(self, dataset): 
        """
        Filter function to filter all Spanish-only sentences from the original dataset.
        """
        language_tags = set(dataset['lid'])
    
        lang2_present = 'lang2' in language_tags
        lang1_not_present = 'lang1' not in language_tags
        
        # Check if remaining other tags are present
        others_present = ('ne' in language_tags) or ('others' in language_tags) or ('fw' in language_tags) or ('ambiguous' in language_tags) or ('mixed' in language_tags) or ('unk' in language_tags)
        
        # Return True if 'lang1' is present, 'lang2' is not present, other tags is present
        return lang2_present and lang1_not_present and  others_present
    
    def dataset_concatenator(self, dataset):
        """
        This function concatentes the newly generated sentences into the CS sentences and creates a new dataset out of it.

        Args:
            dataset (List): Dataset as list to help select only the number of sentences equal CS sentences. This is to ensure data balance.
        """
        list_to_dataset = Dataset.from_pandas(pd.DataFrame(data=dataset[:self.code_switched_sentences])) # Trim the dataset sentences to make it equal to the size of CS sentences.
        if len(self.augmented_dataset) == 0:
            self.augmented_dataset = concatenate_datasets([self.code_switched_dataset, list_to_dataset])
        else:
            self.augmented_dataset = concatenate_datasets([self.augmented_dataset, list_to_dataset])
        
    
    def dataset_augmentor(self, original_sentence, sentences, idx):
        """
        This function takes the original sentence (with its original features), and all sentences generated after synonym replacement 
        and assigns the features to the new sentences.
        Args:
            original_sentence (Dict): Contains the original sentence along with the words,lid, ner that can be reassigned to newer sentences.
            sentences (List): List of new sentences generated after synonym replacement.
            idx (int32): Index to use.

        Returns:
            augmented_entries (List): List of all new sentences as Dict with the idx, lid and ner from the original sentence.
        """
        augmented_entries = []
        for i, sentence in enumerate(sentences, start=idx):
            entry = {}
            entry['idx'] = np.int32(i)
            entry['lid'] = original_sentence['lid']
            entry['words'] = sentence.split(" ")
            entry['ner'] = original_sentence['ner']
            augmented_entries.append(entry)
            
        return augmented_entries
    
    def load_dataset(self):
        """
        Loads the original dataset from Hugging face and returns it
        """
        dataset_name = config["dataset"]
        return load_dataset(dataset_name, name = config["subset"]) 
     
    def load_train_dataset(self):
        """
        Loads the original train dataset from Hugging face and returns it
        """
        dataset = self.load_dataset()
        return dataset['train']
    
    def load_validation_dataset(self):
        """
        Loads the validation dataset from Hugging face and returns it
        """
        dataset = self.load_dataset()
        validation_dataset = dataset['validation']
        
        eng_filtered_dataset = validation_dataset.filter(self.filter_eng)
        esp_filtered_dataset = validation_dataset.filter(self.filter_esp)
        code_switched_dataset = validation_dataset.filter(self.filter_code_switched)
        print('-----------------------\n')
        print(f'>>> Total number of entries in the validation dataset: {len(validation_dataset)}')
        print(f'>>> Total number of entries with just "English" language: {len(eng_filtered_dataset)}')
        print(f'>>> Total number of entries with just "Spanish" language: {len(esp_filtered_dataset)}')
        print(f'>>> Total number of code switched entries in the dataset: {len(code_switched_dataset)}')
        print('-----------------------\n')
        
        return validation_dataset
    
    def load_train_dataset_with_data_augmentation(self):
        """
        This function is responsible for creating the new dataset with data augmentation by using the synonym replacement.
        The responsibilities of the function include:
            a. Loading the original train dataset
            b. Count the total number of English, Spanish and CS sentences.
            c. Initate generation of new sentences.
            d. Initiate the dataset concatenation.
            e. Save the new dataset as arrow file in the local for further usage.

        Returns:
            augmented_dataset (Dataset): Augmented dataset
        """
        train_dataset_path = config["train_dataset_path"]
        dataset_name = config["dataset"]
        try:
            # Check if the file exists
            if os.path.exists(train_dataset_path):
                return Dataset.from_file(f'{train_dataset_path}/data-00000-of-00001.arrow')
            else:
                self.raw_dataset = load_dataset(dataset_name, name = config["subset"])
                eng = 'lang1' 
                esp = 'lang2'
                
                # We have a data imbalance in the dataset. 
                # We want to build a multi language NER tagger with the ability to even take of code switched data.
                # To achieve this, we have to synthetically generate the samples using NLTK and Wordnet for both the languages. 
                
                eng_filtered_dataset = self.raw_dataset['train'].filter(self.filter_eng)
                esp_filtered_dataset = self.raw_dataset['train'].filter(self.filter_esp)
                self.code_switched_dataset = self.raw_dataset['train'].filter(self.filter_code_switched)
                
                self.total_sentences = len(self.raw_dataset["train"])
                self.eng_only_sentences = len(eng_filtered_dataset)
                self.esp_only_sentences = len(esp_filtered_dataset)
                self.code_switched_sentences = len(self.code_switched_dataset)
                
                print(f'>>> Total number of entries in the train dataset: {self.total_sentences}')
                print(f'>>> Total number of entries with just "English" language: {self.eng_only_sentences}')
                print(f'>>> Total number of entries with just "Spanish" language: {self.esp_only_sentences}')
                print(f'>>> Total number of code switched entries in the dataset: {self.code_switched_sentences}')
                print('-----------------------\n')
                
                # Due to the hardware limitations, the code switched sentences in the dataset has to be downsized.
                # The modified dataset will now have 5k sentences each with respect to purely English, purely Spanish and code switched sentences.
                
                new_eng_sentences_dict_list, new_esp_sentences_dict_list = [], []
                
                sentenceGenerator = SentenceGenerator()
                for sentence in eng_filtered_dataset:
                    new_eng_sentences = sentenceGenerator.generate_new_sentence(sentence['words'], eng)
                    new_eng_sentences_dict_list.append(self.dataset_augmentor(sentence, new_eng_sentences, len(self.code_switched_dataset)))
                    
                # Append it to the new dataset
                self.dataset_concatenator(list(chain(*new_eng_sentences_dict_list)))

                for sentence in esp_filtered_dataset:
                    new_esp_sentences = sentenceGenerator.generate_new_sentence(sentence['words'], esp)
                    new_esp_sentences_dict_list.append(self.dataset_augmentor(sentence, new_esp_sentences, len(self.augmented_dataset)))
                
                self.dataset_concatenator(list(chain(*new_esp_sentences_dict_list)))
                
                # Save the dataset File for further reuse
                self.augmented_dataset.save_to_disk(train_dataset_path)
                
                return self.augmented_dataset
                
        except Exception as e:
            print("An error occurred:?", traceback.format_exc())
