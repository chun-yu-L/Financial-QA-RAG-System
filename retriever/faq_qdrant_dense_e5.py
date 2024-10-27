import os
import json

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, MatchAny


load_dotenv()
client = QdrantClient(url=os.getenv("qdrant_url"), timeout=60)


vector_store = QdrantVectorStore(
    client=client,
    collection_name="qa_dense_e5",
    embedding=HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large"),
    retrieval_mode=RetrievalMode.DENSE,
)


with open("./競賽資料集/dataset/preliminary/questions_example.json", "r") as q:
    question_set = json.load(q)
insurance_data = [item for item in question_set['questions'] if item['category'] == 'faq']

answers = []

for Q in insurance_data:
    filter_conditions = Filter(
        must=[
            FieldCondition(
                key="metadata.category",
                match=MatchValue(value=Q['category']),
            ),
            FieldCondition(
                key="metadata.source_id",
                match=MatchAny(any=[str(i) for i in Q['source']]),
            )
        ]
    )

    chroma_output = vector_store.similarity_search(Q['query'],filter=filter_conditions,k=1)

    if chroma_output:
        query_result = chroma_output[0].metadata
    else:
        print(f"Find no answer for {Q['qid']}")
        query_result = {'category': 'not found',
                        'file_source': 'not found',
                        'page_number': -1,
                        'source': '-1',
                        'source_id': '-1'}

    answers.append({
        'qid': Q['qid'],
        'retrieve': int(query_result['source_id']),
        'category': query_result['category']
    })

with open("./faq_dense_e5.json", "w") as Output:
    json.dump({"answers": answers},Output,ensure_ascii=False,indent=4)