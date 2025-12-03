import re
import numpy as np
import os, pyterrier as pt, subprocess
import pandas as pd

# uncomment these lines to run on a virtual venv environment. there is a strange pyterrier version mismatch
# and these lines take care of that.
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

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