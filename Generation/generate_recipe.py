import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import os
import ast
from tqdm import tqdm
import datetime

# Initial inference result (without attention) was bad which may due to other factors too, model is too simple, hyperparemeters not ideal or limited training dataset used due to compute limitation, etc.
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# If only the context vector is passed between the encoder and decoder, that single vector carries the burden of encoding the entire sentence.
# Attention allows the decoder network to “focus” on a different part of the encoder’s outputs for every step of the decoder’s own outputs. 

# Attempting attention decoder with helps from chatgpt.
# class Attention, which is a subclass of PyTorch's nn.Module. It implements an attention mechanism for a Seq2Seq model.
# The __init__ method initialises the components of the Attention module. It defines a linear layer self.attn that will be used to compute attention energies. The input to this layer is twice the hidden_dim because it will be processing concatenated vectors from the current hidden state and the encoder outputs. self.v is a learnable weight vector which will be used to compute the weighted sum of attention energies.
# The forward method performs the actual computation of the attention weights. It takes in the current hidden state of the decoder and the encoder_outputs.
# The number of timesteps in the encoder_outputs is obtained, which corresponds to the length of the input sequence.
# The current hidden state is repeated across all timesteps to match the shape of encoder_outputs. This is necessary because the attention energies need to be computed for each timestep.
# The hidden state and encoder_outputs are concatenated along the last dimension, and this concatenated tensor is passed through the linear layer self.attn and a tanh activation function to compute the energy scores.
# The energy scores are transposed to prepare them for batch matrix multiplication.
# The parameter vector v is repeated for all batches and prepared for batch matrix multiplication.
# The attention weights are computed by performing batch matrix multiplication between v and the energy scores.
# Finally, a softmax function is applied to the attention weights to ensure they are normalized and sum up to 1.
# This attention mechanism allows the model to focus on different parts of the input sequence at each step of the output sequence generation, which can often lead to better performance on tasks like machine translation.

# Define Attention mechanism
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # Linear layer to compute attention energies
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        # Parameter vector for computing the weighted sum of energies
        self.v = nn.Parameter(torch.rand(hidden_dim))
    
    def forward(self, hidden, encoder_outputs):
        # Get the number of timesteps in the encoder outputs
        timestep = encoder_outputs.size(1)
        # Repeat the hidden state across all timesteps
        h = hidden.unsqueeze(1).repeat(1, timestep, 1)
        # Compute energy scores using a tanh activation function
        energy = torch.tanh(self.attn(torch.cat((h, encoder_outputs), 2)))
        # Transpose energy scores for batch matrix multiplication
        energy = energy.transpose(1, 2)
        # Repeat the parameter vector v for all batches
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        # Compute attention weights using batch matrix multiplication
        attention_weights = torch.bmm(v, energy).squeeze(1)
        # Apply softmax to get the normalized attention weights
        return torch.softmax(attention_weights, dim=1)

# class Seq2Seq is a model with an attention mechanism, implemented using PyTorch. Here's a detailed breakdown:
# The __init__ method is where the components of the model are defined. It includes the input and output embedding layers, the encoder and decoder RNN layers, a fully connected output layer, a softmax layer for inference, and an attention layer.
# The forward method is responsible for the actual computation. It takes in the input and target sequences, a teacher forcing ratio, and the start-of-sequence (SOS) and end-of-sequence (EOS) tokens.
# The method starts by encoding the input sequence using the encoder RNN. The encoded variable holds the sequence of hidden states that the encoder RNN generates for each token in the input sequence. Each hidden state in encoded encapsulates the information of the corresponding input token and all previous tokens.
# The final hidden state of the encoder RNN, represented by the hidden variable, is a vector that captures the information of the entire input sequence. This hidden state is used as the initial hidden state for the decoder, providing the decoder with context about the entire input sequence as it starts to generate the output sequence.
# The decoder input is initialised as the SOS token, and then a loop is entered to generate the output sequence one token at a time.
# Inside this loop, the attention weights are computed based on the current decoder hidden state and the encoded states from the encoder. These weights are used to create a context vector, which is a weighted sum of the encoded states. This context vector provides the decoder with information about which parts of the input sequence it should focus on at each decoding step.
# The context vector and the embedded decoder input are concatenated and fed into the decoder RNN. The output of the decoder RNN is then passed through the fully connected layer to produce the output for the current time step.
# Teacher forcing is used to decide whether to use the actual next token or the predicted token as the next decoder input. If all sequences have generated the EOS token, the loop is exited.
# After the loop, a log softmax function is applied to the outputs to convert the raw logits into log-probabilities, This is because the loss function used during training (nn.NLLLoss) expects log-probabilities instead of raw logits.
# In summary, hidden and encoded both contain information about the input sequence but serve different purposes. hidden acts as an initial context for the decoder, while encoded is used in the attention mechanism to guide the decoder's generation of the output sequence.

# CLASStorch.nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None)
# The model uses nn.RNN, the hidden state at time step t, denoted as ℎ𝑡, is computed using the following formula:
# ht=tanh(xt⋅WihT+bih+ht−1⋅WhhT+bhh)
# where:
# 𝑥t is the input at time step t.
# ht−1 is the hidden state from the previous time step.
# Wih are the weights applied to the input.
# Whh are the weights applied to the hidden state from the previous time step.
# bih is the bias term applied to the input.
# bhh is the bias term applied to the hidden state from the previous time step.
# tanh is the hyperbolic tangent activation function. (refer to nonlinearity default value)
# How the update works:
# 1. Input Combination: The input at the current time step xt is multiplied by the weight matrix Wih.
# 2. Previous State Influence: The hidden state from the previous time step ℎ𝑡−1 is multiplied by the weight matrix Whh.
# 3. Bias Addition: A bias term bh is added.
# 4. Activation: The combined result is passed through the tanh activation function to produce the new hidden state ht.

# Define Seq2Seq model class with Attention
class Seq2Seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embedding_dim, hidden_dim, num_layers=2):
        super(Seq2Seq, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layers for input and output vocabularies
        self.input_embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.output_embedding = nn.Embedding(output_vocab_size, embedding_dim)
    
        # Encoder and Decoder RNN layers
        self.encoder_rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder_rnn = nn.RNN(embedding_dim + hidden_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_vocab_size)

        # LogSoftmax layer for inference
        self.log_softmax = nn.LogSoftmax(dim=2)

        # Attention layer
        self.attention = Attention(hidden_dim)

    def forward(self, input, target=None, teacher_forcing_ratio=0.5, SOS_token=101, EOS_token=102):
        batch_size = input.size(0)
        max_len = target.size(1) if target is not None else input.size(1) * 10
        vocab_size = self.fc.out_features

        # Initialise tensor to hold decoder outputs
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(input.device)
        
        # Encode input
        embedded_input = self.input_embedding(input)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(input.device)
        encoded, hidden = self.encoder_rnn(embedded_input, h_0)
        
        # Initialise decoder input as the <sos> token
        decoder_input = torch.full((batch_size,), SOS_token, dtype=torch.long, device=input.device)

        # Initialise a boolean tensor to keep track of which sequences have finished
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=input.device)

        for t in range(1, max_len):
            # Embed the decoder input
            embedded_decoder_input = self.output_embedding(decoder_input).unsqueeze(1)
            
            # Calculate attention weights and apply to encoder outputs
            attn_weights = self.attention(hidden[-1], encoded)
            temp1 = attn_weights.unsqueeze(1)
            context = attn_weights.unsqueeze(1).bmm(encoded)
            
            # Concatenate context vector with embedded decoder input
            rnn_input = torch.cat((embedded_decoder_input, context), 2)
            
            # Decode the input
            decoded, hidden = self.decoder_rnn(rnn_input, hidden)
            output = self.fc(decoded.squeeze(1))
            outputs[:, t-1, :] = output
            
            # Decide if we are going to use teacher forcing or not
            teacher_force = target is not None and torch.rand(1).item() < teacher_forcing_ratio
            
            # Get the highest predicted token from our predictions
            top1 = output.argmax(1)
            
            # If teacher forcing, use actual next token as next input; else use predicted token
            decoder_input = target[:, t] if teacher_force and target is not None else top1

            # Update finished_sequences
            finished_sequences |= (decoder_input == EOS_token)
            
            # Break if all sequences have finished
            if finished_sequences.all():
                break

        return self.log_softmax(outputs)

class RecipeDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.counter = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ingredient = self.data.iloc[idx]['dish_ingredients']
        recipe_step = self.data.iloc[idx]['dish_steps']
        
        ingredient_tokens = self.tokenizer.encode(ingredient, add_special_tokens=True)
        recipe_step_tokens = self.tokenizer.encode(recipe_step, add_special_tokens=True)

        if (self.counter % 20000 == 0):
            self.print_bold('\nDataset:')
            self.print_bold('\nIngredient:')
            print(ingredient)
            keys = self.find_keys_by_values(self.tokenizer.get_vocab(), ingredient_tokens)
            self.print_bold('Token values of ingredient:')
            print(ingredient_tokens)
            self.print_bold('Selected vocabularies:')
            print(keys)
            self.print_bold('\nRecipe step:')
            print(recipe_step)
            keys = self.find_keys_by_values(self.tokenizer.get_vocab(), recipe_step_tokens)
            self.print_bold('Token values of recipe step:')
            print(recipe_step_tokens)
            self.print_bold('Selected vocabularies:')
            print(keys)
        self.counter += 1
        return torch.tensor(ingredient_tokens), torch.tensor(recipe_step_tokens)
    
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

    def print_bold(self, text):
        print('\033[1m' + text + '\033[0m')

class RecipeGenerator:
    def __init__(self, tokenizer_model='bert-base-uncased', embedding_dim=768, hidden_dim=256, num_layers=2):
        # To support current working directory is not the directory where main.py is located.
        dir = os.path.dirname(__file__) # This file's directory
        ds = dir.split(os.sep)
        ds.pop(len(ds)-1) # Move one level up
        self.dir = os.sep.join(ds)
        # Read the dataset
        self.recipes = pd.read_csv(f'{self.dir}{os.sep}Datasets{os.sep}Recommendation{os.sep}RAW_recipes.csv')
        self.interactions = pd.read_csv(f'{self.dir}{os.sep}Datasets{os.sep}Recommendation{os.sep}RAW_interactions.csv')
        if not os.path.exists(f'{self.dir}{os.sep}Outputs'):
            os.makedirs(f'{self.dir}{os.sep}Outputs')
        if not os.path.exists(f'{self.dir}{os.sep}Outputs{os.sep}Generation'):
            os.makedirs(f'{self.dir}{os.sep}Outputs{os.sep}Generation')
        self.df = self.get_filtered_recipes(self.recipes, self.interactions)
        self.tokenizer_model = tokenizer_model
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_model)
        self.model = Seq2Seq(self.tokenizer.vocab_size, self.tokenizer.vocab_size, embedding_dim, hidden_dim, num_layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

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
            print(f'{key}: {value}', end=',' if i < middle_end - middle_start - 1 else '')
        
        # Print ellipsis after the middle section if needed
        if len(items) - middle_end > num_pairs:
            print('...', end=', ')

        # Print key-value pairs from the back
        for i, (key, value) in enumerate(items[-num_to_print_back:]):
            print(f'{key}: {value}', end=', ' if i < num_to_print_back - 1 else '\n')

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

    # https://www.kaggle.com/code/ashwinik/recipe-recommendation-bert-sentence-embedding
    def get_filtered_recipes(self, recipes, interactions, num_users=10, num_words=80):
        # Group interactions by recipe and calculate summary statistics
        g = {'rating': ['mean'], 'user_id': ['nunique']}
        int_summary = interactions.groupby(['recipe_id']).agg(g).reset_index()
        # Its gives a multi-index output convert it to single index by combining both levels
        ind = pd.Index([e[0] + e[1] for e in int_summary.columns.tolist()])
        int_summary.columns = ind
        int_summary.columns = ['recipe_id', 'rating_mean', 'user_id_nunique']
        # Keep only those recipes in consideration which have been reviewed by more than num_users
        int_summary_small = int_summary[int_summary['user_id_nunique'] > num_users]
        # Merge datasets
        filtered_recipes = pd.merge(recipes, int_summary_small, left_on='id', right_on='recipe_id', how='inner')
        filtered_recipes['dish_ingredients'] = filtered_recipes['ingredients'].apply(lambda x: ','.join(ast.literal_eval(x)))
        filtered_recipes['dish_steps'] = filtered_recipes['steps'].apply(lambda x: ','.join(ast.literal_eval(x)))

        # To reduce sequence length
        rows_to_drop = []
        # Iterate over each row
        for index, row in filtered_recipes.iterrows():
            # If steps have more than 100 words, drop the row
            if (sum(len(word.strip()) > 0 for word in row['dish_steps'].replace(',', ' ').split()) > num_words):
                rows_to_drop.append(index)
        # Drop the selected rows
        filtered_recipes = filtered_recipes.drop(rows_to_drop)

        return filtered_recipes

    def collate_fn(self, batch):
        # Unzip the batch of tuples into separate lists of input sequences and target sequences
        input_batch, target_batch = zip(*batch)
        
        # Pad the input sequences to ensure uniform length within the batch
        # Using batch_first=True puts the batch dimension first in the output tensor
        # padding_value=self.tokenizer.pad_token_id specifies the value to use for padding
        input_batch_padded = nn.utils.rnn.pad_sequence(input_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        # Pad the target sequences similarly to the input sequences
        target_batch_padded = nn.utils.rnn.pad_sequence(target_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        # Move the padded input and target sequences to the specified device (e.g., GPU)
        return input_batch_padded.to(self.device), target_batch_padded.to(self.device)

    def train(self, epochs=10, batch_size=32, learning_rate=0.001, teacher_forcing_ratio=0.5, visualise_progress=100):
        # Load the dataset
        data = self.df
        
        # Split the dataset into training, validation, and test sets
        train_data, temp_data = train_test_split(data, test_size=0.2)
        val_data, test_data = train_test_split(temp_data, test_size=0.5)
        
        # Create datasets
        train_dataset = RecipeDataset(train_data, self.tokenizer)
        val_dataset = RecipeDataset(val_data, self.tokenizer)
        test_dataset = RecipeDataset(test_data, self.tokenizer)
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        
        #criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        criterion = nn.NLLLoss(ignore_index=self.tokenizer.pad_token_id)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        initial_teacher_forcing_ratio = 1.0
        final_teacher_forcing_ratio = 0.0

        # Training loop
        for epoch in range(epochs):
            time = datetime.datetime.now()
            print (f'\nEpoch {epoch+1}/{epochs} started at {time}')

            # Calculate the current teacher forcing ratio
            teacher_forcing_ratio = initial_teacher_forcing_ratio - (initial_teacher_forcing_ratio - final_teacher_forcing_ratio) * (epoch / (epochs-1))

            self.model.train()
            epoch_loss = 0
            count = 0
            i = 0

            # Get parameters before training per epoch
            params_before = {name: p.clone().detach() for name, p in self.model.named_parameters()}

            for input_batch, target_batch in tqdm(train_loader):
                i += 1
                optimizer.zero_grad()                
                output_tensor = self.model(input_batch, target_batch, teacher_forcing_ratio)

                # Selects all elements in the second dimension except the last one, reeshape it into a 2D tensor with shape (batch_size * (sequence_length-1), vocab_size)
                output = output_tensor[:, :-1, :].reshape(-1, self.tokenizer.vocab_size)
                # Remove the <sos> token from the target and reshape into a 1D tensor representing the target tokens with shape (batch_size * (sequence_length-1))
                target = target_batch[:, 1:].reshape(-1)

                # Calculate the loss between the predicted output and the target tokens
                loss = criterion(output, target)

                loss.backward()
                optimizer.step()                
                epoch_loss += loss.item()

                if (count % visualise_progress == 0):
                    keys = self.find_keys_by_values(self.tokenizer.get_vocab(), input_batch[0])
                    self.print_bold(f'\nTraining epoch {epoch+1} batch {i}:')
                    self.print_bold('\nToken values of the first ingredients (input):')
                    print(input_batch[0])
                    self.print_bold('Selected vocabularies (input):')
                    print(keys)

                    keys = self.find_keys_by_values(self.tokenizer.get_vocab(), target_batch[0])
                    self.print_bold('\nToken values of the first recipe steps (target):')
                    print(target_batch[0])
                    self.print_bold('Selected vocabularies (target):')
                    print(keys)

                    output_ids = output_tensor[0].argmax(dim=1).squeeze().tolist()
                    keys = self.find_keys_by_values(self.tokenizer.get_vocab(), output_ids)
                    self.print_bold('\nToken values of the first recipe steps (prediction):')
                    print(output_ids)
                    self.print_bold('Selected vocabularies (prediction):')
                    print(keys)
                count += 1

            # Get parameters after training per epoch
            params_after = {name: p for name, p in self.model.named_parameters()}

            # Check how many distinct parameters have changed
            #  A distinct parameter typically refers to the weights associated with a specific layer (chatgpt)
            num_changed_params = sum((params_before[name] != params_after[name]).any() for name in params_before)
            print(f'\nDistinct parameters changed before and after epoch {epoch+1}/{epochs}: {num_changed_params}')

            # Check how many total elements of parameters have changed
            # Individual elements refer to the individual values within each distinuct parameter (chatgpt)
            total_elements_changed = sum((params_before[name] != params_after[name]).sum() for name in params_before)
            print(f'\nTotal elements changed before and after epoch {epoch+1}/{epochs}: {total_elements_changed}')

            # Create and print dictionary of parameter-wise count of changed elements
            param_changed_elements = {name: (params_before[name] != params_after[name]).sum().item() for name in params_before}
            print(f'\nParameter-wise count of changed elements before and after epoch {epoch+1}/{epochs} - start:')
            for name, count in param_changed_elements.items():
                print(f'{name}: {count}')
            print(f'Parameter-wise count of changed elements before and after epoch {epoch+1}/{epochs} - end\n')

            val_loss = self.evaluate(val_loader, criterion)
            print(f'\nEpoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}')

            # Save model state per epoch
            # Anyone of them may be selected for inference after reviewing training/evaluation messages for the best generation
            # In case it takes too long to run, terminate the run (Ctrl-C in anaconda prompt) after certain number of epoch then use (copy or rename as necessary) a relevant generation_model_state_save_x_over_y as generation_model_state_save
            print(f'Save model state as generation_model_state_save_epoch_{epoch+1}_over_{epochs}')
            self.save_model_state(f'generation_model_state_save_epoch_{epoch+1}_over_{epochs}')

            time2 = datetime.datetime.now()
            print (f'\nEpoch {epoch+1}/{epochs} ended at {time2}')
            print (f'Time taken is {time2-time}')

    def evaluate(self, dataloader, criterion, teacher_forcing_ratio=-0.1, visualise_progress=10):
        self.model.eval()
        val_loss = 0
        count = 0
        i = 0

        with torch.no_grad():
            for input_batch, target_batch in dataloader:
                i += 1
                # Ensure a sequence length in output_tensor is equal to a sequence length in target,
                # but it doesn't use teacher forcing (negative value ensures that), therefore prediction is used token by token.
                output_tensor = self.model(input_batch, target_batch, teacher_forcing_ratio)

                # Selects all elements in the second dimension except the last one, reshape it into a 2D tensor with shape (batch_size * (sequence_length-1), vocab_size)
                output = output_tensor[:, :-1, :].reshape(-1, self.tokenizer.vocab_size)
                # Remove the <sos> token from the target and reshape into a 1D tensor representing the target tokens with shape (batch_size * (sequence_length-1))
                target = target_batch[:, 1:].reshape(-1)
                # Calculate the loss between the predicted output and the target tokens
                loss = criterion(output, target)
                
                val_loss += loss.item()

                if (count % visualise_progress == 0):
                    keys = self.find_keys_by_values(self.tokenizer.get_vocab(), input_batch[0])
                    self.print_bold(f'\nEvaluation batch {i}:')
                    self.print_bold('\nToken values of the first ingredients (input):')
                    print(input_batch[0])
                    self.print_bold('Selected vocabularies (input):')
                    print(keys)

                    keys = self.find_keys_by_values(self.tokenizer.get_vocab(), target_batch[0])
                    self.print_bold('\nToken values of the first recipe steps (target):')
                    print(target_batch[0])
                    self.print_bold('Selected vocabularies (target):')
                    print(keys)

                    output_ids = output_tensor[0].argmax(dim=1).squeeze().tolist()
                    keys = self.find_keys_by_values(self.tokenizer.get_vocab(), output_ids)
                    self.print_bold('\nToken values of the first recipe steps (prediction):')
                    print(output_ids)
                    self.print_bold('Selected vocabularies (prediction):')
                    print(keys)
                count += 1
    
        return val_loss / len(dataloader)

    def generate_recipe(self, ingredients):
        self.model.eval()
        with torch.no_grad():
            input_tensor = self.tokenize_and_pad([ingredients])
            output_tensor = self.model(input_tensor)
            output_ids = output_tensor.argmax(dim=2).squeeze().tolist()
            recipe_steps = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            self.print_bold('\nGeneration:')
            self.print_bold('\nIngredients (input):')
            print(ingredients)
            keys = self.find_keys_by_values(self.tokenizer.get_vocab(), input_tensor[0])
            self.print_bold('Token values of ingredient (input):')
            print(input_tensor[0])
            self.print_bold('Selected vocabularies (input):')
            print(keys)
            self.print_bold('\nRecipe steps (prediction):')
            print(recipe_steps)
            keys = self.find_keys_by_values(self.tokenizer.get_vocab(), output_ids)
            self.print_bold('Token values of recipe steps (prediction):')
            print(output_ids)
            self.print_bold('Selected vocabularies (prediction):')
            print(keys)

            return recipe_steps

    def tokenize_and_pad(self, texts):
        tokenized = [self.tokenizer.encode(text, add_special_tokens=True) for text in texts]
        max_len = max(len(seq) for seq in tokenized)
        padded = [seq + [self.tokenizer.pad_token_id] * (max_len - len(seq)) for seq in tokenized]
        return torch.tensor(padded).to(self.device)

    def save_model_state(self, path):
        # Save the model state dictionary to the specified path
        torch.save(self.model.state_dict(), f'{self.dir}{os.sep}Outputs{os.sep}Generation{os.sep}{path}')

    def load_model_state(self, path):
        # Load the model state dictionary from the specified path
        self.model.load_state_dict(torch.load(f'{self.dir}{os.sep}Outputs{os.sep}Generation{os.sep}{path}'))
        self.model.eval()

    def print_bold(self, text):
        print('\033[1m' + text + '\033[0m')

    def main(self):
        self.print_bold(f'\nVocabularies from {self.tokenizer_model}:')
        self.print_dictionary(self.tokenizer.get_vocab(), 25)

        if not os.path.exists(f'{self.dir}{os.sep}Outputs{os.sep}Generation{os.sep}generation_model_state_save'):
            self.train(epochs=12, batch_size=8)
            print('Save model state as generation_model_state_save')
            self.save_model_state('generation_model_state_save')

        #May change to load generation_model_state_save_epoch_x_over_y too
        self.load_model_state('generation_model_state_save')

        #ingredients_input = 'onion, flour, sugar'
        #ingredients_input = 'apple, avocado, brocolli, lemon, sugar'
        ingredients_input = input('Enter ingredients (comma-separated): ')
        recipe = self.generate_recipe(ingredients_input)
        self.print_bold('\nIngredients:')
        print(ingredients_input)
        self.print_bold('Generated Recipe Steps:')
        print(recipe)

# Example usage
if __name__ == '__main__':
    generator = RecipeGenerator()
    generator.main()
