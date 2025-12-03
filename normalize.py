import pandas as pd
import numpy as np
import pyterrier as pt
import subprocess
import os
from sklearn.preprocessing import MinMaxScaler
import embedding


java_home = subprocess.check_output(['/usr/libexec/java_home', '-v', '17']).decode('utf-8').strip()
os.environ["JAVA_HOME"] = java_home
os.environ["JVM_PATH"]  = java_home + "/lib/server/libjvm.dylib"
os.environ["PATH"]      = java_home + "/bin:" + os.environ["PATH"]

if not pt.java.started():
    pt.java.init()

df = pd.read_csv("cleaned_output.csv")
df['docno'] = df.index.astype(str)

index_path = os.path.expanduser("./assets/pp_index")
index_ref = pt.IndexRef.of(index_path + "/data.properties")
bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25", controls={"bm25.b": 0.75, "bm25.k_1": 1.0})

def bm25_scores(query):
    output = bm25.search(query)
    output = output[["docno", "score"]]
    output.columns = ["docno", "bm25_score"]
    return output

def normalize_bm25(df):
    scaler = MinMaxScaler()
    df['bm25_norm'] = scaler.fit_transform(df[['bm25_score']])
    return df

def embedding_scores(query):
    results = embedding.search_embeddings(query)
    embedding_df = pd.DataFrame(results)
    embedding_df['docno'] = embedding_df['corpus_id'].astype(str)

    embedding_df = embedding_df.rename(columns={"score": "vector_score"})
    return embedding_df[["docno", "vector_score"]]

def normalize_embedding(df):
    scaler = MinMaxScaler()
    df['vector_norm'] = scaler.fit_transform(df[["vector_score"]])
    return df

def hybrid_search(query, b=0.5):
    bm25_df = bm25_scores(query)
    bm25_df = normalize_bm25(bm25_df)

    vector_df = embedding_scores(query)
    vector_df = normalize_embedding(vector_df)
    
    # note that we do a left join, effectively keeping all values of noramlized BM-25
    # recall that BM25 values will not appear if the query term doesn't even exist in the recipe
    # and to maintain the integrity of a 'hybrid' model, we must have some BM25 value to keep the
    # recommendations 'hybrid', otherwise the embeddings will fully take over.
    merged_df = pd.merge(bm25_df, vector_df, on="docno", how="left")

    # b controls the weightage towards the final score. when b reaches 1 score relies on embedding, when 
    # b reaches zero score relies on bm-25. final_score = ((1 - b) * bm_25 + (vector * b))
    merged_df["hybrid_score"] = ((1-b) * merged_df['bm25_norm']) + (merged_df["vector_norm"] * b)
    return merged_df.sort_values("hybrid_score", ascending=False)

print(hybrid_search("a hot meal for a cold day"))