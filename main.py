import os
import cohere
import wikipedia
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
co=cohere.Client(os.environ["COHERE_API_KEY"])
article=wikipedia.page("Machinelearning")
text=article.content
text_splitter=RecursiveCharacterTextSplitter(
  chunk_size=512,
  chunk_overlap=50,
  length_function=len,
  is_separator_regex=False,
 )


chunks_=text_splitter.create_documents([text])
chunks=[c.page_content for c in chunks_]

model="embed-english-v3.0"
response=co.embed(
    texts=chunks,
    model=model,
    input_type="search_document",
    embedding_types=["float"]
)
embeddings=response.embeddings.float
vector_database={i:np.array(embedding) for i,embedding in enumerate(embeddings)}
query="Give me a list of machine learning models mentioned in the page. Also give a brief description of each model"

response=co.embed(
    texts=[query],
    model=model,
    input_type="search_query",
    embedding_types=["float"]
)

query_embedding=response.embeddings.float[0]
def cosine_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

similarities=[cosine_similarity(query_embedding,chunk) for chunk in embeddings]
sorted_indices=np.argsort(similarities)[::-1]

top_indices=sorted_indices[:10]

top_chunks_after_retrieval=[chunks[i] for i in top_indices]



response=co.rerank(
    query=query,
    documents=top_chunks_after_retrieval,
    top_n=3,
    model='rerank-english-v3.0'
    )

print(response)

top_chunks_after_rerank=[top_chunks_after_retrieval[result.index] for result in response.results]



preamble="""
You help people answer their questions and other requests interactively.
You will be asked a very wide array of requests on all kinds of topics. 
You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer.
You should focus on serving the user's needs as best you can, which will be wide-ranging.

Unless the user asks for a different style of answer, you should answer in full sentences,
using proper grammar and spelling.

"""

documents = [
    {"title": "chunk 0", "snippet": top_chunks_after_rerank[0]},
    {"title": "chunk 1", "snippet": top_chunks_after_rerank[1]},
    {"title": "chunk 2", "snippet": top_chunks_after_rerank[2]},
  ]


response=co.chat(
    message=query,
    documents=documents,
    preamble=preamble,
    model="command-r",
    temperature=0.3
)

print("Final answer: ")
print(response.text)