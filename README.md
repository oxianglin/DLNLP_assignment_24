# DLNLP_assignment_24
PLEASE REMEMBER TO DOWNLOAD THE DATASET FILES FOR RECOMMENDATION. REFER TO DLNLP_assignment_24/Datasets/Recommendation/ SECTION BELOW.
Please download this file to see a clearer formatting.

Project Directory Structure and Description
This project directory contains multiple subdirectories and files organised to facilitate various machine learning tasks, including recipe translation, recommendation, and generation. Below is a detailed description of the directory structure and the purpose of each component.
Directory Structure
    •	DLNLP_assignment_24/:
        •	Datasets/:
            •	Translation/:
                •	train_data.json: Contains 359 pairs of sentences. Each pair consists of a sentence in Malay and its English translation. This dataset is used to fine-tune a pre-trained translation model (mesolitica/translation-t5-tiny-standard-bahasa-cased). The sentences focus on two food ingredients, "belacan" (shrimp paste) and "sambal" (chilli sauce).
                •	eval_data.json: Contains 31 pairs of sentences. Each pair consists of a sentence in Malay and its English translation. This dataset is used for evaluation during the fine-tuning of the pre-trained translation model.
                •	train_data.json and eval_data.json are included in github repository.
            •	Recommendation/:
                •	RAW_interactions.csv: Contains user interactions such as user reviews on recipes, referring to recipes listed in RAW_recipes.csv.
                •	RAW_recipes.csv: Contains details of various recipes.
                •	RAW_interactions.csv and RAW_recipes.csv are not included in github repository. There is a download button in the link below. Download and save the files in DLNLP_assignment_24/Datasets/Recommendation directory. https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions
        •	Translation/:
            •	translation.py: Script to fine-tune the pre-trained translation model using datasets in DLNLP_assignment_24/Datasets/Translation. It also performs inference to compare translations from Malay to English using both the original pre-trained model and the fine-tuned model.
        •	Recommendation/:
            •	ingredient_to_recipe.py: Script to fine-tune a pre-trained sentence transformer (all-MiniLM-L6-v2) using datasets in DLNLP_assignment_24/Datasets/Recommendation. It uses cosine similarity loss function to fine-tune on a list of ingredients. During inference, it takes user input of ingredients, finds cosine similarity with existing ingredients in the dataset, and recommends the top 3 recipe steps corresponding to the highest cosine similarity scores.
        •	Generation/:
            •	generate_recipe.py: Script to train a simple Seq2Seq model with attention decoder using datasets in DLNLP_assignment_24/Datasets/Recommendation. Ingredients in batches are used as input to the model, and recipe steps in batches are used as targets for the NLLLoss function to train the model to generate recipe steps from the input ingredients. It is also used for inference to generate recipe steps based on user input ingredients.
        •	Outputs/:
            •	Translation/:
                •	fine_tuned_model/: Contains the saved fine-tuned translation model.
            •	Recommendation/:
                 •	fine_tuned_model_mini_from_mini/: Contains the saved fine-tuned model of all-MiniLM-L6-v2 with labels used for fine-tuning obtained from all-MiniLM-L6-v2.
                 •	fine_tuned_model_mini_from_mpnet/: Contains the saved fine-tuned model of all-MiniLM-L6-v2 with labels used for fine-tuning obtained from all-mpnet-base-v2.
                 •	fine_tuned_mini_from_mini_embeddings.pickle: Contains embeddings encoded from the ingredients list in the CSV dataset using fine_tuned_model_mini_from_mini, it is used to speed up recommendation inference.
                 •	fine_tuned_mini_from_mpnet_embeddings.pickle: Contains embeddings encoded from the ingredients list in the CSV dataset using fine_tuned_model_mini_from_mpnet
                 •	original_mini_embeddings.pickle: Contains original embeddings encoded from the ingredients list in the CSV dataset using all-MiniLM-L6-v2 model.
                •	original_mpnet_embeddings.pickle: Contains original embeddings encoded from the ingredients list in the CSV dataset using all-mpnet-base-v2 model.
            •	Generation/:
                •	generation_model_state_save: Contains the Seq2Seq model's saved state for inference use.
                •	generation_model_state_save_epoch_X_over_12: Contains the saved model state after each epoch (e.g., generation_model_state_save_epoch_1_over_12 means saved after epoch 1 out of 12 total epochs).
        •	main.py: The entry point for running translation, recommendation, or generation tasks. It uses a loop to prompt the user to choose which operation to run.

        
\\Usage\\
To start using the project, run python main.py. This script will guide you through selecting the desired operation (translation, recommendation, or generation) and will handle the necessary processing based on your input.

Translation
The translation.py script fine-tunes a pre-trained translation model using the datasets in DLNLP_assignment_24/Datasets/Translation. It then performs inference to compare translations from Malay to English using both the original pre-trained model and the fine-tuned model.

Recommendation
The ingredient_to_recipe.py script fine-tunes a pre-trained sentence transformer using the datasets in DLNLP_assignment_24/Datasets/Recommendation. During inference, it takes user input of ingredients, finds the cosine similarity with existing ingredients in the dataset, and recommends the top 3 recipes.

Generation
The generate_recipe.py script trains a Seq2Seq model using the datasets in DLNLP_assignment_24/Datasets/Recommendation. It generates recipe steps based on user input ingredients during inference.

Outputs
The Outputs directory will be created when the program is first run. It contains subdirectories for each task (Translation, Recommendation, Generation) where fine-tuned models and other relevant output files are saved for later use.
This structured approach ensures that datasets, scripts, and outputs are well-organised and easily accessible for each specific task.

