#pip install rank-bm25

import re
import numpy as np
# from rank_bm25 import BM25Okapi
import os, pyterrier as pt, subprocess
import pandas as pd

java_home = subprocess.check_output(['/usr/libexec/java_home', '-v', '17']).decode('utf-8').strip()
os.environ["JAVA_HOME"] = java_home
os.environ["JVM_PATH"]  = java_home + "/lib/server/libjvm.dylib"
os.environ["PATH"]      = java_home + "/bin:" + os.environ["PATH"]

if not pt.java.started():
    pt.java.init()

df = pd.read_parquet("assets/preprocessed_recipes_500.parquet")

index_path = os.path.expanduser("./assets/pp_index")
os.makedirs(index_path, exist_ok=True)

corpus = (
    df[["cleaned_text"]]
    .reset_index()
    .rename(columns={'index': 'docno', 'id': 'id', 'cleaned_text': 'text'})
)
corpus['docno'] = corpus['docno'].astype(str)

indexer = pt.IterDictIndexer(index_path, overwrite=True)
index_ref = indexer.index(corpus.to_dict(orient="records"))

index = pt.IndexFactory.of(index_ref)

bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25", controls={"bm25.b": 0.75, "bm25.k_1": 1.0})

search = input("Search: ")

bm25_results = bm25.search(search)
print(bm25_results)

# def simple_tokenize(text: str):
#     """
#     Tokenize text
#     """
#     text = text.lower()
#     tokens = re.split(r"[^a-zA-Z]+", text)
#     return [t for t in tokens if t]

# def build_bm25_corpus(recipes):
#     """
#     Build a corpus for BM25: each element is a list of tokens.
#     Simultaneously return a list of recipe_ids for easy lookup later.
#     """
#     corpus = []
#     recipe_ids = []

#     for r in recipes:
#         title_tokens = simple_tokenize(r["title"])
#         ing_tokens = simple_tokenize(r.get("ingredients", ""))
#         instr_tokens = simple_tokenize(r.get("instructions", ""))

#         doc_tokens = (
#             title_tokens * 3 +   # title (first important)
#             ing_tokens * 2 +     # ingredients (Second important)
#             instr_tokens         # instructions (normal)
#         )

#         corpus.append(doc_tokens)
#         recipe_ids.append(r["id"])

#         return corpus, recipe_ids


# class BM25Engine:
#     def __init__(self, recipes):
#         """
#         initialize BM25 from recipes list
#         """
#         self.recipes = recipes
#         self.corpus,self.recipe_ids = build_bm25_corpus(recipes)
#         self.bm25 = BM25Okapi(self.corpus) # feed all recipes text into BM 25

#     def search (self, query:str, top_k:int = 20):
#         """
#         Have the user enter a query, and we return a sorted list of 
#         recipe results (including title, rating, recipe content, etc.).
#         """
#         tokens = simple_tokenize(query)
#         scores = self.bm25.get_score(tokens) # For each recipe in the corpus, compute the BM25 relevance score.
#         top_k = min(top_k, len(scores))
#         top_idx = np.argsort(scores)[::-1][:top_k] # from small to big to big to small
        
#         results = []
#         for idx in top_idx:
#             r = self.recipes[idx]
#             results.append({
#                 "recipes_id": self.recipes_ids[idx],
#                 "title": r["title"],
#                 "score": float (score(idx)),
#                 "recipes": r
#             })

#         return results

#     #For Hybrid (BM25 + Embedding) fusion, return "the cleanest, most direct BM25 output".
#     #Do not return titles, recipe content, or dictionaries; return only pure index + score.
#     #Hybrid search first retrieves index + score from BM25, then fuses it with embedding scores.

#     def search_raw(self, query: str, top_k: int = 100):
#         tokens = simple_tokenize(query)
#         scores = self.bm25.get_scores(tokens)

#         #rank from bigest to smallest
#         top_k = min(top_k, len(scores))
#         top_idx = np.argsort(scores)[::-1][:top_k]

#         return top_idx, scores[top_idx]



