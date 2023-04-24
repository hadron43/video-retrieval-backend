import os
import json
import time
import pandas as pd
import numpy as np
import random
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util, CrossEncoder, InputExample, losses, models, datasets
import faiss
import numpy as np
from pathlib import Path
from vector_engine.vector_engine.utils import vector_search, id2details

def search(query, index, model, top_k_count):
    # Semantic Search
    # Encoding the query using the bi-encoder and finding potentially relevant summaries
    t=time.time()
    query_vector = model.encode([query])
    top_k = index.search(query_vector, top_k_count)
    top_k_ids = top_k[1].tolist()[0]
    top_k_ids = list(np.unique(top_k_ids))

    # Re-Ranking Results
    # Scoring all retrieved summaries with the cross_encoder
    t=time.time()
    cross_inp = [[query, passages[hit]] for hit in top_k_ids]

    bienc_op = [video_ids[hit] for hit in top_k_ids]
    cross_scores = cross_encoder.predict(cross_inp)
    # print('>>>> Results in Total Time: {}'.format(time.time()-t))
        
    # Output of top-k hits from re-ranker
    # print("\n-------------------------\n")
    # print(f"Top-{top_k_count} Cross-Encoder Re-ranker hits")

    cross_ranks = np.argsort(np.array(cross_scores))[::-1]
    # for hit in cross_ranks:
    #     print("\t{}".format(bienc_op[hit].replace("\n", " ")))

    cross_results = [bienc_op[hit] for hit in cross_ranks]
    return cross_scores.tolist(), cross_results, time.time() - t


# read data from the csv file (from the location it is stored)
data_dir = 'data/'
data_list = os.listdir(data_dir)
dataframe_list = []
raw_data = {}
print(data_list)

for data_file in data_list:
    if '.json' not in data_file:
        continue
    data_path = data_dir + data_file
    data_name = data_file[:-5]
    save_data_path = data_dir + 'summary/' + data_name + '.xlsx'
    print("Data File:", data_name)
    df = pd.read_json(data_path)
    df = df.astype(str)
    rows, columns = df.shape
    dataframe_list.append(df)

    f = open(data_path)
    data = json.load(f)
    for record in data:
        del record['fused_captions']
        del record['text_captions']
        raw_data[record['id']] = record

df = pd.concat(dataframe_list)
# df.head(10)

print(f"Videos: {df.id.unique().shape[0]}")
n_ids = len(df.id.unique())
id_list = df.id.unique()
id_value_list = np.arange(n_ids)
id_dict = dict(zip(id_list, id_value_list))
# id_dict = {id_list[i]: id_value_list[i] for i in range(n_ids)}
df['id_value'] = df['id'].apply(lambda x: id_dict[x])
# df.head()

# Searching with finetuned model
model_path = f'finetuned_model'
bi_encoder = SentenceTransformer(model_path)
top_k_count = 10

# Cross-encoder for re-ranking the retrieved results list to improve the quality
cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')

passages = df['summary'].values.tolist()
video_ids = df['id'].values.tolist()
print("Video Summaries:", len(passages))

# Retrieve results based on query
# query = 'gordon ramsay delicious food'
index = faiss.read_index(f"finetuned_faiss_index.index")
# search(query, index, bi_encoder, top_k_count)

# Starting FAST API app

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    text: str

@app.post("/search")
async def search_api(query: Query):
    global index, bi_encoder, top_k_count, df
    scores, results, time_taken = search(query.text, index, bi_encoder, top_k_count)

    return {
        'scores': scores,
        'results': [raw_data[id] for id in results],
        'time': time_taken
    }
