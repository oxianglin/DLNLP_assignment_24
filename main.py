import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Translation'))
from translation import TranslationModel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Recommendation'))
from ingredient_to_recipe import RecipeModel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Generation'))
from generate_recipe import RecipeGenerator

def main():
    translation_model = TranslationModel()
    recipe_model = RecipeModel()
    recipe_geneation_model = RecipeGenerator()
    while True:
        print("\nChoose subtask:")
        print("1. Translation from Malay to English")
        print("2. Food ingredient to recipe recommendation")
        print("3. Food ingredient to recipe generation")
        print("4. Quit")
        try:
            choice = int(input("Enter your choice (1/2/3/4): "))
        except:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
            continue

        if choice == 1:
            translation_model.main()
        elif choice == 2:
            recipe_model.main()
        elif choice == 3:
            recipe_geneation_model.main()
        elif choice == 4:
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 4, or 4.")
            continue

if __name__ == '__main__':
    main()
