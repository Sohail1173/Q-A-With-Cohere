import os
import cohere
import wikipedia
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
co=cohere.Client(os.environ["COHERE_API_KEY"])

docs = ["The capital of France is Paris",
        "PyTorch is a machine learning framework based on the Torch library.",
        "The average cat lifespan is between 13-17 years",
        "Rishabh Pant 194 runs Tristan Stubbs 189 runs David Warner 166 runs Prithvi Shaw 151 runs Abhishek Porel 91 runs"]


doc_emb=co.embed(texts=docs,input_type="search_document",model="embed-english-v3.0").embeddings
doc_emb=np.asarray(doc_emb)
print(doc_emb.shape)
query="how many runs rishab pant scores"
query_emb=co.embed(texts=[query],input_type="search_query",model="embed-english-v3.0").embeddings
query_emb=np.asarray(query_emb)

scores=np.dot(query_emb,doc_emb.T)[0]
max_idx=np.argsort(-scores)

print(f"Query:{query}")
for id in max_idx:
    print(f"Scores:{scores[id]:.2f}")
    print(docs[id])
    print("********************************")
