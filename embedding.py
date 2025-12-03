import ast
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import torch
import numpy as np

df = pd.read_csv("cleaned_output.csv")

model = SentenceTransformer("BAAI/bge-small-en")

embeddings = model.encode(df['cleaned_text'].to_list(), normalize_embeddings=True)

df['embeddings'] = [x.tolist() for x in embeddings]
df.to_csv('embedding_output.csv', index=False)

def search_embeddings(query):
    prefix = "Represent this sentence for searching relevant passages: "
    df = pd.read_csv("embedding_output.csv")
    df['embeddings'] = df['embeddings'].apply(ast.literal_eval)
    query_embedding = model.encode(prefix+query, normalize_embeddings=True)
    corpus_embeddings = torch.tensor(np.array(df['embeddings'].to_list())).float()
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=len(df))
    top_results = hits[0]

    # for hit in top_results:
    #     text = df.iloc[hit['corpus_id']]['text']
    #     print("corpus_id", hit['corpus_id'])
    #     print("score: ", hit['score'])
    #     print(text)
    return hits[0]

search_embeddings("nice snack for a sick day")