import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os
import pickle
import math
import datetime
import numpy as np
import torch

class RecipeModel:
    def __init__(self):
        # To support current working directory is not the directory where main.py is located.
        dir = os.path.dirname(__file__) #This file's directory
        ds = dir.split(os.sep)
        ds.pop(len(ds)-1) #move one level up
        self.dir = os.sep.join(ds)
        # Read the dataset
        self.recipes = pd.read_csv(f'{self.dir}{os.sep}Datasets{os.sep}Recommendation{os.sep}RAW_recipes.csv')
        self.interactions = pd.read_csv(f'{self.dir}{os.sep}Datasets{os.sep}Recommendation{os.sep}RAW_interactions.csv')
        # Prepare path and files in Outputs subdirectory
        if not os.path.exists(f'{self.dir}{os.sep}Outputs'):
            os.makedirs(f'{self.dir}{os.sep}Outputs')
        if not os.path.exists(f'{self.dir}{os.sep}Outputs{os.sep}Recommendation'):
            os.makedirs(f'{self.dir}{os.sep}Outputs{os.sep}Recommendation')
        self.model_save_path_mini_from_mini = f'{self.dir}{os.sep}Outputs{os.sep}Recommendation{os.sep}fine_tuned_model_mini_from_mini'
        self.model_save_path_mini_from_mpnet = f'{self.dir}{os.sep}Outputs{os.sep}Recommendation{os.sep}fine_tuned_model_mini_from_mpnet'
        self.original_mini_embeddings_pickle = f'{self.dir}{os.sep}Outputs{os.sep}Recommendation{os.sep}original_mini_embeddings.pickle'
        self.original_mpnet_embeddings_pickle = f'{self.dir}{os.sep}Outputs{os.sep}Recommendation{os.sep}original_mpnet_embeddings.pickle'
        self.fine_tuned_mini_from_mini_embeddings_pickle = f'{self.dir}{os.sep}Outputs{os.sep}Recommendation{os.sep}fine_tuned_mini_from_mini_embeddings.pickle'
        self.fine_tuned_mini_from_mpnet_embeddings_pickle = f'{self.dir}{os.sep}Outputs{os.sep}Recommendation{os.sep}fine_tuned_mini_from_mpnet_embeddings.pickle'
    
    def fine_tune_model(self, model_to_label, model_to_tune, recipes, interactions, model_save_path):
        # Fine-tuning parameters
        train_batch_size = 8
        num_epochs = 3
        learning_rate= 2e-05 #equal to default value, refer to fit description in https://sbert.net/docs/package_reference/SentenceTransformer.html

        # Limit size of dataset and fine-tuning time
        unique_users_greater_than = 20 #The number will be increased if estimated hours is more than affordable hours
        pairs_per_second = 16 #Please adjuest according to speed of compute engine used to run fine-tuning
        affordable_hours_less_than = 3 #Please adjust according to affordable numbers of hours to run fine-tuning

        # Learning rate changes
        learning_rate_changes = 0.9  #Set to a value for experiment , such as 1.1 to increase 10%, 0.9 to decrease 10%

        # Check if GPU is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move the model to the desired device
        model_to_label.to(device)
        model_to_tune.to(device)

        time = datetime.datetime.now()
        print (f'\nSelection of affordable dataset size started at {time}')

        while True:
            print(f'\nUnique users greater than {unique_users_greater_than}')
            # Get filtered_recipes based on number of unique users preformed review on a recipe to optimize fine-tuning time
            filtered_recipes = self.get_filtered_recipes(self.recipes, self.interactions, unique_users_greater_than)

            # The ingredients are combined into one string
            filtered_recipes['dish_ingredients'] = filtered_recipes['ingredients'].apply(lambda x : ','.join(ast.literal_eval(x)))

            # Split dataset into train, validation, and test sets
            train_df, temp_df = train_test_split(filtered_recipes, test_size=0.2, random_state=42)
            val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

            # Create ingredients list
            ingredients_list = train_df['dish_ingredients'].tolist()

            unique_pairs_count = 0
            for i in range(len(ingredients_list)):
                for j in range(i+1, len(ingredients_list)):
                    unique_pairs_count += 1

            print(f'num_epochs = {num_epochs}')
            print(f'Pairs per batch = {train_batch_size}')
            print(f'Total unique pairs = {unique_pairs_count}')
            print(f'Total batches = {math.ceil(unique_pairs_count/train_batch_size)}')
            print(f'Estimation is {pairs_per_second} pairs per second of fine-tuning')
            estimated_total_fine_tuning_time = math.ceil((unique_pairs_count/pairs_per_second)*num_epochs/3600)
            print(f'Affordable total fine-tuning time is less than {affordable_hours_less_than} hours')
            print(f'Estimated total fine-tuning time is {estimated_total_fine_tuning_time} hours')
            if (estimated_total_fine_tuning_time < affordable_hours_less_than):
                break
            else:
                unique_users_greater_than += 10

        time2 = datetime.datetime.now()
        print (f'\nSelection of affordable dataset size ended at {time2}')
        print (f'Time taken is {time2-time}')

        print (f'\nFine-tuning started at {time2}\n')

        #Encode to assist labels creation
        print('Encode to assist labels creation')
        recipe_embeddings= model_to_label.encode(ingredients_list)

        # Calculate similarity scores as labels
        print('Calculate similarity scores as labels')
        similarity_scores = cosine_similarity(recipe_embeddings)

        # Define train_examples using unique pair of ingredients from ingredients_list and corresponding similarity_scores as label
        train_examples = [InputExample(texts=[ingredients_list[i], ingredients_list[j]], label=similarity_scores[i][j]) 
                        for i in range(len(ingredients_list)) 
                        for j in range(i+1, len(ingredients_list))]

        # Create DataLoader instances for training and validation sets
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)

        # Define loss function
        train_loss = losses.CosineSimilarityLoss(model_to_tune)

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            # Fine-tune SentenceTransformer model
            model_to_tune.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, optimizer_params = {'lr': learning_rate}, warmup_steps=100)

            # Encode the train ingredients
            ingredients_list = train_df['dish_ingredients'].tolist()
            recipe_embeddings= model_to_tune.encode(ingredients_list)
            # Calculate similarity scores
            similarity_scores = cosine_similarity(recipe_embeddings)
            mean_similarity_score_train = np.mean(similarity_scores)
            print(f'mean similarity score for train dataset is {mean_similarity_score_train}')

            # Encode the val ingredients
            ingredients_list = val_df['dish_ingredients'].tolist()
            recipe_embeddings= model_to_tune.encode(ingredients_list)
            # Calculate similarity scores
            similarity_scores = cosine_similarity(recipe_embeddings)            
            mean_similarity_score_val = np.mean(similarity_scores)
            print(f'mean similarity score for val dataset is {mean_similarity_score_val}')

            # Evaluation below is based on consideration that difference between
            # 'mean consine similarity score on val or test ingredients' and 'mean cosine similarity score on train ingredients' should be small

            # Calculate difference of mean score between val and train
            mean_similarity_diff_val_versus_train = abs(mean_similarity_score_val-mean_similarity_score_train)
            print(f'Mean similarity difference between val and train is {mean_similarity_diff_val_versus_train}')
            if (mean_similarity_diff_val_versus_train<0.02):
                if (epoch<num_epochs-1):
                    print('Likely goodfit, early stopping criteria met')
                    break
            elif (mean_similarity_diff_val_versus_train>0.2):
                #Likely underfit if it is early in epoch iterations, overfit if it is late in epoch interations
                if (epoch < num_epochs//2):
                    print(f'Likely underfit, adjust learning rate from {learning_rate} to ', end='')
                    learning_rate *= learning_rate_changes  # Reduce or increase learning rate
                    print(f'{learning_rate}')
                else:
                    print('Likely overfit')
                    if (epoch<num_epochs-1):
                        print('Early stopping criteria met')
                        break

        # Encode the test ingredients
        ingredients_list = test_df['dish_ingredients'].tolist()
        recipe_embeddings= model_to_tune.encode(ingredients_list)
        # Calculate similarity scores
        similarity_scores = cosine_similarity(recipe_embeddings)            
        mean_similarity_score_test = np.mean(similarity_scores)
        print(f'mean similarity score for test dataset is {mean_similarity_score_test}')

        # Calculate difference of mean score between test and train
        mean_similarity_diff_test_versus_train = abs(mean_similarity_score_test-mean_similarity_score_train)
        print(f'Mean similarity difference between test and train is {mean_similarity_diff_test_versus_train}')
        if (mean_similarity_diff_test_versus_train>0.2):
            print('Likely overfit')

        # Save the fine-tuned model
        model_to_tune.save(model_save_path)

        time3 = datetime.datetime.now()
        print (f'\nFine-tuning ended at {time3}')
        print (f'Time taken is {time3-time2}')
    
    def get_filtered_recipes(self, recipes, interactions, num_users=20):
        time = datetime.datetime.now()
        print (f'\nget_filtered_recipes started at {time}')

        # https://www.kaggle.com/code/ashwinik/recipe-recommendation-bert-sentence-embedding
        # Restrict analysis only to those recipes which have been reviewed by more than 20 people
        # Analysis has shown that most recipes are only added but never seen
        # Furthermore, compute resource is insufficient for a large dataset
        g = {'rating' : ['mean'],'user_id' : ['nunique']}
        int_summary = interactions.groupby(['recipe_id']).agg(g).reset_index()
        # Its gives a multi-index output convert it to single index by combining both levels
        ind = pd.Index([e[0] + '_' +e[1] for e in int_summary.columns.tolist()])
        # Assign the column names 
        int_summary.columns = ind
        int_summary.columns = ['recipe_id', 'rating_mean', 'user_id_nunique']
        # Keep only those recipes in consideration which have been reviewed by more than 20 people
        int_summary_small = int_summary[int_summary['user_id_nunique'] > num_users]

        # Merge datasets
        filtered_recipes = pd.merge(recipes, int_summary_small, left_on='id', right_on='recipe_id', how='inner')

        time2 = datetime.datetime.now()
        print (f'\nget_filtered_recipes ended at {time2}')
        print (f'Time taken is {time2-time}')

        return filtered_recipes
    
    def save_embeddings_to_pickle(self, original_model_mini, original_model_mpnet, fine_tuned_model_mini_from_mini, fine_tuned_model_mini_from_mpnet, recipes):
        time = datetime.datetime.now()
        print (f'\nsave_embeddings_to_pickle started at {time}')

        # The ingredients in list is combined into one string
        recipes['dish_ingredients'] = recipes['ingredients'].apply(lambda x : ' '.join(ast.literal_eval(x)))
        ingredients_list = recipes['dish_ingredients'].tolist()

        # Encoding recipes' ingredients
        print("\nEncode recipes' ingredients with original all-MiniLM-L6-v2")
        if (os.path.exists(self.original_mini_embeddings_pickle)):
            print(f'{self.original_mini_embeddings_pickle} found, no need to encode and save')
        else:
            recipes_embedding = original_model_mini.encode(ingredients_list)
            with open(self.original_mini_embeddings_pickle, 'wb') as f:
                pickle.dump(recipes_embedding, f)

        print("\nEncode recipes' ingredients with original all-mpnet-base-v2")
        if (os.path.exists(self.original_mpnet_embeddings_pickle)):
            print(f'{self.original_mpnet_embeddings_pickle} found, no need to encode and save')
        else:
            recipes_embedding = original_model_mpnet.encode(ingredients_list)
            with open(self.original_mpnet_embeddings_pickle, 'wb') as f:
                pickle.dump(recipes_embedding, f)

        print("Encode recipes' ingredients with fine tuned all-MiniLM-L6-v2 with label from all-MiniLM-L6-v2")
        if (os.path.exists(self.fine_tuned_mini_from_mini_embeddings_pickle)):
            print(f'{self.fine_tuned_mini_from_mini_embeddings_pickle} found, no need to encode and save')
        else:
            recipes_embedding = fine_tuned_model_mini_from_mini.encode(ingredients_list)
            with open(self.fine_tuned_mini_from_mini_embeddings_pickle, 'wb') as f:
                pickle.dump(recipes_embedding, f)

        print("Encode recipes' ingredients with fine tuned all-MiniLM-L6-v2 with label from all-mpnet-base-v2")
        if (os.path.exists(self.fine_tuned_mini_from_mpnet_embeddings_pickle)):
            print(f'{self.fine_tuned_mini_from_mpnet_embeddings_pickle} found, no need to encode and save')
        else:
            recipes_embedding = fine_tuned_model_mini_from_mpnet.encode(ingredients_list)
            with open(self.fine_tuned_mini_from_mpnet_embeddings_pickle, 'wb') as f:
                pickle.dump(recipes_embedding, f)

        time2 = datetime.datetime.now()
        print (f'\nsave_embeddings_to_pickle ended at {time2}')
        print (f'Time taken is {time2-time}')
    
    def get_recipe(self, original_model_mini, original_model_mpnet, fine_tuned_model_mini_from_mini, fine_tuned_model_mini_from_mpnet, ingredients_input, recipes):
        time = datetime.datetime.now()
        print (f'\nget_recipe started at {time}')

        if (original_model_mini):
            # Encode input ingredients
            print ('\nEncode input with original model')
            ingredients_input_embedding = original_model_mini.encode(ingredients_input)
            # Loading recipes' ingredients
            print('Load embeddings with original model')
            with open(self.original_mini_embeddings_pickle, 'rb') as f:
                recipes_embedding = pickle.load(f)
        elif (original_model_mpnet):
            # Encode input ingredients
            print ('\nEncode input with original model')
            ingredients_input_embedding = original_model_mpnet.encode(ingredients_input)
            # Loading recipes' ingredients
            print('Load embeddings with original model')
            with open(self.original_mpnet_embeddings_pickle, 'rb') as f:
                recipes_embedding = pickle.load(f)
        elif (fine_tuned_model_mini_from_mini):
            # Encode input ingredients
            print ('\nEncode input with fine tuned model')
            ingredients_input_embedding = fine_tuned_model_mini_from_mini.encode(ingredients_input)
            # Loading recipes' ingredients
            print('Load embeddings with fine tuned model')
            with open(self.fine_tuned_mini_from_mini_embeddings_pickle, 'rb') as f:
                recipes_embedding = pickle.load(f)
        elif (fine_tuned_model_mini_from_mpnet):
            # Encode input ingredients
            print ('\nEncode input with fine tuned model')
            ingredients_input_embedding = fine_tuned_model_mini_from_mpnet.encode(ingredients_input)
            # Loading recipes' ingredients
            print('Load embeddings with fine tuned model')
            with open(self.fine_tuned_mini_from_mpnet_embeddings_pickle, 'rb') as f:
                recipes_embedding = pickle.load(f)

        # Calculate cosine similarities between input and recipe ingredients
        print("Calculate similarity between input and recipes' ingredients")
        similarities = cosine_similarity(ingredients_input_embedding.reshape(1, -1), recipes_embedding)

        # Find indices of top 3 recipes with highest similarity
        print('Find top 3 recipes with highest similarity')
        top_indices = similarities.argsort()[0][-3:][::-1]  # Get indices of top 3 highest similarity

        # Retrieve recipe information and similarity scores for the top 3 recipes
        top_recipes = []
        top_similarities = []
        for idx in top_indices:
            recipe_name = recipes.iloc[idx]['name']
            recipe_ingredients = recipes.iloc[idx]['ingredients']
            recipe_steps = recipes.iloc[idx]['steps']
            similarity_score = similarities[0, idx]
            top_recipes.append((recipe_name, recipe_ingredients, recipe_steps))
            top_similarities.append(similarity_score)

        time2 = datetime.datetime.now()
        print (f'\nget_recipe ended at {time2}')
        print (f'Time taken is {time2-time}')

        return top_recipes, top_similarities

    def print_bold(self, text):
        print('\033[1m' + text + '\033[0m')

    def main(self):
        # Define SentenceTransformer models
        # https://www.sbert.net/docs/pretrained_models.html
        original_model_mini = SentenceTransformer('all-MiniLM-L6-v2')
        original_model_mpnet = SentenceTransformer('all-mpnet-base-v2')
        
        fine_tuning = False if os.path.exists(self.model_save_path_mini_from_mini) else True
        if (fine_tuning):
            print('Fine-tuning all-MiniLM-L6-v2 with label from all-MiniLM-L6-v2')
            fine_tuned_model_mini_from_mini = SentenceTransformer('all-MiniLM-L6-v2')
            self.fine_tune_model(original_model_mini, fine_tuned_model_mini_from_mini, self.recipes, self.interactions, self.model_save_path_mini_from_mini)

        fine_tuning = False if os.path.exists(self.model_save_path_mini_from_mpnet) else True
        if (fine_tuning):
            print('Fine-tuning all-MiniLM-L6-v2 with label from all-mpnet-base-v2')
            fine_tuned_model_mini_from_mpnet = SentenceTransformer('all-MiniLM-L6-v2')
            self.fine_tune_model(original_model_mpnet, fine_tuned_model_mini_from_mpnet, self.recipes, self.interactions, self.model_save_path_mini_from_mpnet)
        
        # Define SentenceTransformer models
        fine_tuned_model_mini_from_mini = SentenceTransformer(self.model_save_path_mini_from_mini)
        fine_tuned_model_mini_from_mpnet = SentenceTransformer(self.model_save_path_mini_from_mpnet)

        # Get filtered_recipes based on number of unique users preformed review on a recipe to optimise encoding time
        filtered_recipes = self.get_filtered_recipes(self.recipes, self.interactions)

        # Save embeddings to speed up get_recipe
        self.save_embeddings_to_pickle(original_model_mini, original_model_mpnet, fine_tuned_model_mini_from_mini, fine_tuned_model_mini_from_mpnet, filtered_recipes)

        # User input loop
        while True:
            print('\nChoose Model:')
            print('1. Original Pre-trained Model all-MiniLM-L6-v2')
            print('2. Original Pre-trained Model all-mpnet-base-v2')            
            print('3. Fine-tuned Pre-trained Model all-MiniLM-L6-v2 with label from all-MiniLM-L6-v2')
            print('4. Fine-tuned Pre-trained Model all-MiniLM-L6-v2 with label from all-mpnet-base-v2')
            print('5. Quit')
            try:
                choice = int(input('Enter your choice (1/2/3/4/5): '))
            except:
                print('Invalid choice. Please enter 1, 2, 3, 4 or 5.')
                continue

            if choice == 1:
                model_choice = original_model_mini
            elif choice == 2:
                model_choice = original_model_mpnet
            elif choice == 3:
                model_choice = fine_tuned_model_mini_from_mini
            elif choice == 4:
                model_choice = fine_tuned_model_mini_from_mpnet
            elif choice == 5:
                print('Exiting the program.')
                break
            else:
                print('Invalid choice. Please enter 1, 2, 3, 4 or 5.')
                continue

            ingredients_input = input('Enter ingredients (comma-separated): ')
            #ingredients_input = ' '.join(ingredients_input.replace(',', ' ').split())

            # Call get_recipe function
            top_recipes, top_similarities = self.get_recipe(original_model_mini if model_choice==original_model_mini else 0, original_model_mpnet if model_choice==original_model_mpnet else 0, fine_tuned_model_mini_from_mini if model_choice==fine_tuned_model_mini_from_mini else 0, fine_tuned_model_mini_from_mpnet if model_choice==fine_tuned_model_mini_from_mpnet else 0, ingredients_input, filtered_recipes)

            if (model_choice == original_model_mini):
                choice = '\noriginal_model_mini:'
            elif (model_choice == original_model_mpnet):
                choice = '\noriginal_model_mpnet:'
            elif (model_choice == fine_tuned_model_mini_from_mini):
                choice = '\nfine_tuned_model_mini_from_mini:'
            else:
                choice = '\nfine_tuned_model_mini_from_mpnet:'
            self.print_bold(choice)

            # Print the returned top recipes
            for i, recipe in enumerate(top_recipes, start=1):
                recipe_name, recipe_ingredients, recipe_steps = recipe
                self.print_bold(f'\nTop {i} Recipe:')
                print(f'Similarity score: {top_similarities[i-1]}')
                print(f'Name: {recipe_name}')
                print(f'Ingredients: {recipe_ingredients}')
                print(f'Steps: {recipe_steps}')

if __name__ == '__main__':
    recipe_model = RecipeModel()
    recipe_model.main()
       
