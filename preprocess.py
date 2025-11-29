import json
import csv
import pandas as pd
import os
import pyterrier as pt
import subprocess
import nltk, re
from tqdm import tqdm
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

with open('recipes.json', 'r') as file:
    data = json.load(file)

# 12 recipe categories
recipe_categories = [
    "Lunch recipes",
    "Dinner recipes",
    "Breakfast recipes",
    "Storecupboard",
    "Cheese recipes",
    "Desserts",
    "Fish and seafood",
    "Pasta",
    "Chicken",
    "Meat",
    "Vegan",
    "Vegetarian"
]

recipes_array = []
counter = 0
data = data['recipes']
# categories shown above
for category in recipe_categories:
    # each category has a subcategory ("quick lunch recipes", etc.)
    sub_category = data[category]
    for _, recipes in sub_category.items():
        # columns we want in our new category
        # we should save and export the columns we want this into a new csv
        for recipe in recipes:
            counter += 1
            # recipes_array.append({
            #     "id": recipe["id"],
            #     "name": recipe.get("name", ""),
            #     "ingredients": recipe.get("ingredients", []),
            #     "steps": recipe.get("steps", []),
            #     "description": recipe.get("description", ""),
            #     "times": recipe.get("times", {}),
            #     "difficulty": recipe.get("difficulty", ""),
            # })
            description = recipe.get('description', '')
            ingredients = ",".join(recipe.get('ingredients', []))
            steps = recipe.get("steps", [])
            prep_time = recipe.get('times', {}).get('Preparation', '')
            cook_time = recipe.get('times', {}).get('Cooking', '')
            difficult = recipe.get('difficult', '')
            text = f"{recipe.get('name', '')}. {description}. {ingredients}.\n"
            text += f"{steps}. Prep Time: {prep_time} Cook Time: {cook_time}."
            text += f"\nDifficulty: {difficult}."
            recipes_array.append([recipe["id"], text])
            if counter > 500:
                break

headers = ["id", "text"]

with open('output.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(recipes_array)

# print(recipes_array)
df = pd.read_csv('output.csv')

java_home = subprocess.check_output(['/usr/libexec/java_home', '-v', '17']).decode('utf-8').strip()
os.environ["JAVA_HOME"] = java_home
os.environ["JVM_PATH"]  = java_home + "/lib/server/libjvm.dylib"
os.environ["PATH"]      = java_home + "/bin:" + os.environ["PATH"]

if not pt.java.started():
    pt.java.init()
     
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    """
    cleans the text by removing numbers, lowercasing, punctionation, and stemming words.
    """
    tokens = re.findall(r'\b[a-z]{3,}\b', text.lower())
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

tqdm.pandas()
df['cleaned_text'] = df['text'].progress_apply(clean_text)

df.to_csv("cleaned_output.csv", index=False)

index_path = os.path.expanduser("./assets/pp_index")
os.makedirs(index_path,exist_ok=True)

corpus = (
    df[['cleaned_text']]
    .reset_index()
    .rename(columns={'index': 'docno', 'id': 'id', 'cleaned_text': 'text'})
)
corpus["docno"] = corpus["docno"].astype(str)
indexer = pt.IterDictIndexer(index_path, overwrite=True)
index_ref = indexer.index(corpus.to_dict(orient="records"))

df.to_parquet("assets/preprocessed_recipes_500.parquet", index=False)