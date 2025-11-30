Savorly: BM25 Recipe Search Engine

Savorly is a lightweight recipe search system built with PyTerrier and BM25.
It preprocesses raw recipe data, builds an inverted index, and supports keyword search through a command-line interface.

Project Structure
.
├── preprocess.py           # Extracts, cleans, and preprocesses recipe text; builds index
├── BM25.py                 # Loads index and runs BM25 retrieval
├── output.csv              # Raw extracted recipe text (generated)
├── cleaned_output.csv      # Cleaned/stemmed text (generated)
├── assets/
│   ├── preprocessed_recipes_500.parquet
│   └── pp_index/           # PyTerrier index (generated)
└── .gitignore

Preprocessing

preprocess.py performs:

Extraction of recipe fields (name, ingredients, steps, description, timing)

Text cleaning using:

Lowercasing

Stopword removal

Token filtering

Porter stemming

Creation of a cleaned corpus

Construction of a PyTerrier index in assets/pp_index/

Run preprocessing:

python preprocess.py

BM25 Search

BM25.py loads the preprocessed Parquet file, initializes PyTerrier, and performs BM25 retrieval.

Run the search interface:

python BM25.py


Then enter a query when prompted, for example:

Search: chicken soup


The script prints ranked BM25 results to the console.

Requirements

Python 3.x

Java 17 (required by PyTerrier)

Python packages:

pandas

numpy

nltk

tqdm

python-terrier

pyarrow
