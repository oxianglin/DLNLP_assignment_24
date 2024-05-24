import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import EarlyStoppingCallback
import json
from bert_score import score
import logging
import numpy as np
import os

#Translation entry point
class TranslationModel:
    def __init__(self):
        self.model_name = 'mesolitica/translation-t5-tiny-standard-bahasa-cased'
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        # To support current working directory is not the directory where main.py is located.
        dir = os.path.dirname(__file__) #This file's directory
        ds = dir.split(os.sep)
        ds.pop(len(ds)-1) #move one level up
        self.dir = os.sep.join(ds)
        # Prepare path and files in Outputs subdirectory
        if not os.path.exists(f'{self.dir}{os.sep}Outputs'):
            os.makedirs(f'{self.dir}{os.sep}Outputs')
        if not os.path.exists(f'{self.dir}{os.sep}Outputs{os.sep}Translation'):
            os.makedirs(f'{self.dir}{os.sep}Outputs{os.sep}Translation')
        self.model_save_path = f'{self.dir}{os.sep}Outputs{os.sep}Translation{os.sep}fine_tuned_model'
        self.train_dataset = TranslateDataset(data_file=f'{self.dir}{os.sep}Datasets{os.sep}Translation{os.sep}train_data.json', tokenizer=self.tokenizer)
        self.eval_dataset = TranslateDataset(data_file=f'{self.dir}{os.sep}Datasets{os.sep}Translation{os.sep}eval_data.json', tokenizer=self.tokenizer)

    def train_and_evaluate(self):

        training_args = Seq2SeqTrainingArguments(
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            output_dir=f'{self.dir}{os.sep}Outputs{os.sep}Translation{os.sep}results',
            num_train_epochs=3,
            logging_dir=f'{self.dir}{os.sep}Outputs{os.sep}Translation{os.sep}logs',
            logging_steps=20,
            overwrite_output_dir=True,
            save_strategy='steps',
            save_steps=20,
            evaluation_strategy='steps',
            eval_steps=20,
            save_total_limit=1,
            load_best_model_at_end=True
        )

        trainer = ModelTrainer(self.model, self.tokenizer, training_args, self.train_dataset, self.eval_dataset, self.model_save_path)
        trainer.train_model()

    def print_dictionary(self, dictionary, num_pairs, print_all=False):
        # Convert dictionary items to a list of tuples
        items = list(dictionary.items())

        # If print_all is True, print the entire dictionary
        if print_all:
            print('Printing the entire dictionary:')
            for key, value in items:
                print(f'{key}: {value}')
            return

        # Determine the number of key-value pairs to print from the front and back
        num_to_print_front = min(len(items), num_pairs)
        num_to_print_back = min(len(items), num_pairs)

        # Calculate the start and end index for the middle section
        middle_start = max(0, len(items) // 2 - num_pairs // 2)
        middle_end = min(len(items), middle_start + num_pairs)

        # Print key-value pairs from the front
        for key, value in items[:num_to_print_front]:
            print(f'{key}: {value}', end=', ')
        
        # Print ellipsis before the middle section if needed
        if middle_start > num_pairs:
            print('...', end=', ')

        # Print key-value pairs from the middle section
        for i, (key, value) in enumerate(items[middle_start:middle_end]):
            print(f'{key}: {value}', end=', ' if i < middle_end - middle_start - 1 else '')
        
        # Print ellipsis after the middle section if needed
        if len(items) - middle_end > num_pairs:
            print('...', end=', ')

        # Print key-value pairs from the back
        for i, (key, value) in enumerate(items[-num_to_print_back:]):
            print(f'{key}: {value}', end=', ' if i < num_to_print_back - 1 else '\n')

    def print_bold(self, text):
        print('\033[1m' + text + '\033[0m')

    def main(self):
        fine_tuning = False if os.path.exists(self.model_save_path) else True
        if (fine_tuning):
            self.train_and_evaluate()

        original_tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        original_model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        finetuned_tokenizer = T5Tokenizer.from_pretrained(self.model_save_path)
        finetuned_model = T5ForConditionalGeneration.from_pretrained(self.model_save_path)

        # Print portion of vocabularies before and after fine-tuning
        self.print_bold('\nVocabularies before fine-tuning:')
        self.print_dictionary(original_tokenizer.get_vocab(), 25)
        self.print_bold('\nVocabularies after fine-tuning:')
        self.print_dictionary(finetuned_tokenizer.get_vocab(), 25)

        # Analyse vocabularies differences before and after fine-tuning
        vocab_analyzer = VocabularyAnalyzer(original_tokenizer, finetuned_tokenizer)
        vocab_analyzer.print_vocab_differences()

        # Malay sentences
        sentences = [
            'Saya makan nasi dengan sambal.',
            'Belacan adalah bahan utama dalam pembuatan sambal.',
            'Dia suka makan sambal belacan yang pedas.'
        ]

        self.print_bold('\nOriginal model:')
        original_translator = Translator(original_model, original_tokenizer)
        for sentence in sentences:
            original_translator.translate(sentence)

        self.print_bold('\nFine-tuned model:')
        finetuned_translator = Translator(finetuned_model, finetuned_tokenizer)
        for sentence in sentences:
            finetuned_translator.translate(sentence)

#For Seq2SeqTrainer to load the dataset
class TranslateDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, tokenizer):
        self.data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        malay_sentence, english_sentence = self.data[idx]
        malay_tokenized = self.tokenizer.encode(f'terjemah ke English: {malay_sentence}', return_tensors='pt', max_length=512, truncation=True)
        english_tokenized = self.tokenizer.encode(english_sentence, return_tensors='pt', max_length=512, truncation=True)
        return {'input_ids': malay_tokenized.squeeze(), 'labels': english_tokenized.squeeze()}

# Translator from Malay to English
class Translator: 
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.vocabs = self.tokenizer.get_vocab()

    def find_keys_by_values(self, dictionary, values):
        keys = []
        for val in values:
            found = False
            for key, dict_val in dictionary.items():
                if val == dict_val:
                    keys.append(key)
                    found = True
                    break
            if not found:
                keys.append('UNK')
        return keys

    # https://huggingface.co/mesolitica/translation-t5-tiny-standard-bahasa-cased
    def translate(self, malay_sentence):
        input_ids = self.tokenizer.encode(f'terjemah ke English: {malay_sentence}', return_tensors='pt')
        print('\nTokenize malay sentence input')
        keys = self.find_keys_by_values(self.vocabs, input_ids[0])
        print(f'Input tensor values = {[input.item() for input in input_ids[0]]}')
        print(f'Selected vocabularies = {keys}')
        outputs = self.model.generate(input_ids, max_length=100)
        #For visualiztion purpose on the gnerated tensor value to the vocab
        print('\nTranslation generated')
        print('Before skipping special ids')
        keys = self.find_keys_by_values(self.vocabs, outputs[0])
        print(f'Output tensor values = {[output.item() for output in outputs[0]]}')
        print(f'Selected vocabularies = {keys}')
        all_special_ids = [0, 1, 2]
        outputs = [i for i in outputs[0] if i not in all_special_ids]
        print('\nAfter skipping special ids')
        keys = self.find_keys_by_values(self.vocabs, outputs)
        print(f'Output tensor values = {[output.item() for output in outputs]}')
        print(f'Selected vocabularies = {keys}')
        english_sentence = self.tokenizer.decode(outputs, spaces_between_special_tokens=False, skip_special_tokens=True)
        print(f'\nMalay sentence = {malay_sentence}')
        print(f'Translated English sentence = {english_sentence}')

class VocabularyAnalyzer:
    def __init__(self, original_tokenizer, finetuned_tokenizer):
        self.original_tokenizer = original_tokenizer
        self.finetuned_tokenizer = finetuned_tokenizer
    
    def analyze_vocab_differences(self):
        original_vocab = self.original_tokenizer.get_vocab()
        finetuned_vocab = self.finetuned_tokenizer.get_vocab()

        common_keys = set(original_vocab.keys()) & set(finetuned_vocab.keys())
        added_keys = set(finetuned_vocab.keys()) - set(original_vocab.keys())
        removed_keys = set(original_vocab.keys()) - set(finetuned_vocab.keys())

        common_diff = {key: (original_vocab[key], finetuned_vocab[key]) for key in common_keys if original_vocab[key] != finetuned_vocab[key]}
        added_diff = {key: finetuned_vocab[key] for key in added_keys}
        removed_diff = {key: original_vocab[key] for key in removed_keys}

        return common_diff, added_diff, removed_diff

    def print_vocab_differences(self):
        common_diff, added_diff, removed_diff = self.analyze_vocab_differences()

        original_vocab_size = len(self.original_tokenizer.get_vocab())
        finetuned_vocab_size = len(self.finetuned_tokenizer.get_vocab())

        print(f'\nOriginal vocabularies size: {original_vocab_size}')
        print(f'Finetuned vocabularies size: {finetuned_vocab_size}')

        if not (common_diff or added_diff or removed_diff):
            print('No differences found between vocabularies before and after fine-tuning.')
            return

        if common_diff:
            print('Common differences:')
            for key, value in common_diff.items():
                print(f'Key: {key}, Original value: {value[0]}, Finetuned value: {value[1]}')

        if added_diff:
            print('\nAdded keys in finetuned vocab:')
            for key, value in added_diff.items():
                print(f'Key: {key}, Value: {value}')

        if removed_diff:
            print('\nRemoved keys in finetuned vocab:')
            for key, value in removed_diff.items():
                print(f'Key: {key}, Value: {value}')

# Evaluator before, during and after fine-tuning
class ModelEvaluator:
    @staticmethod
    # For visualisation purpose before and after fine-tuning, calculate and print each parameter name along with its total number of elements.
    def print_model_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params}')
        print(f'Total trainable parameters: {total_trainable_params}')

        parameters_dict = {name: p.numel() for name, p in model.named_parameters()}
        print('Parameter-wise count of total elements:')
        for name, count in parameters_dict.items():
            print(f'{name}: {count}')

    @staticmethod
    # For Seq2SeqTrainer to compute metrics after each epoch during fine-tuning
    def compute_metrics(eval_pred, tokenizer):
        predictions, labels = eval_pred
        all_special_ids = [0, 1, 2, -100]

        # Initialise list to store predicted sentences
        predicted_sentences = []
        # Find the maximum value along the last dimension (vocab_size)
        batch_pred = np.argmax(predictions[0], axis=-1)
        # Convert predictions (in token IDs) to human-readable text
        for pred in batch_pred:
            decoded_token_ids = tokenizer.decode(torch.tensor(pred), skip_special_tokens=True)
            predicted_sentences.append(decoded_token_ids)

        # Initialise list to store target sentences
        target_sentences = []
        # Convert labels (in token IDs) to human-readable text
        for label in labels:
            filtered_label = [i for i in label if i not in all_special_ids]
            decoded_token_ids = tokenizer.decode(torch.tensor(filtered_label), skip_special_tokens=True)
            target_sentences.append(decoded_token_ids)

        # Compute BERTScore
        print('BERTScore started.')
        _, _, bert_score_f1 = score(predicted_sentences, target_sentences, lang='en', model_type='bert-base-multilingual-cased')
        print(f'bert_score_f1 = {bert_score_f1}')
        print('BERTScore ended')

        # Return BERTScore F1 as the evaluation metric
        return {'bert_score_f1': bert_score_f1.mean().item()}

# Fine-tuning class
class ModelTrainer:
    def __init__(self, model, tokenizer, training_args, train_dataset, eval_dataset, model_save_path):
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.model_save_path = model_save_path

    def train_model(self):
        print(f'\nFine-tuning dataset size = {len(self.train_dataset)}')
        print(f'Eval dataset size = {len(self.eval_dataset)}')

        # Define data collator
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        # Define trainer with compute_metrics function and early stopping callback
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            data_collator=data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=lambda eval_pred: ModelEvaluator.compute_metrics(eval_pred, self.tokenizer),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        print('Model parameters before fine-tuning - start:')
        ModelEvaluator.print_model_parameters(self.model)
        print('Model parameters before fine-tuning - end\n')

        # Get parameters before fine-tuning
        params_before = {name: p.clone().detach() for name, p in self.model.named_parameters()}

        trainer.train()

        print('Model parameters after fine-tuning - start:')
        ModelEvaluator.print_model_parameters(self.model)
        print('Model parameters after fine-tuning - end\n')

        # Get parameters after fine-tuning
        params_after = {name: p for name, p in self.model.named_parameters()}

        # Check how many distinct parameters have changed
        #  A distinct parameter typically refers to the weights associated with a specific layer 
        num_changed_params = sum((params_before[name] != params_after[name]).any() for name in params_before)
        print(f'Distinct parameters changed: {num_changed_params}')

        # Check how many total elements of parameters have changed
        # Individual elements refer to the individual values within each distinuct parameter 
        total_elements_changed = sum((params_before[name] != params_after[name]).sum() for name in params_before)
        print(f'Total elements changed: {total_elements_changed}')

        # Create and print dictionary of parameter-wise count of changed elements
        param_changed_elements = {name: (params_before[name] != params_after[name]).sum().item() for name in params_before}
        print('Parameter-wise count of changed elements - start:')
        for name, count in param_changed_elements.items():
            print(f'{name}: {count}')
        print('Parameter-wise count of changed elements - end\n')

        # Save model and tokenizer
        self.model.save_pretrained(self.model_save_path)
        self.tokenizer.save_pretrained(self.model_save_path)

if __name__ == '__main__':
    translator = TranslationModel()
    translator.main()
