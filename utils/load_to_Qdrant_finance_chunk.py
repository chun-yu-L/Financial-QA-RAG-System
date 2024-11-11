import json
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from tqdm import tqdm

text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        # "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ],
    chunk_size=1500,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)


# texts = text_splitter.split_text(finance_extract_directly_patched["1001"])
# len(texts)

file_path = "finance_extract_directly_patched.json"
data = json.loads(Path(file_path).read_text())
loader = JSONLoader(file_path=file_path, jq_schema=".[]", text_content=False)

docs = []
for key, value in data.items():
    doc = Document(
        page_content=value,
        metadata={
            "source_id": key,
            "category": "finance",
            "file_source": "finance_extract_directly_patched.json",
        },
    )
    docs.append(doc)

RCtext_splited = text_splitter.split_documents(docs)


# 假設 `text_splitter.split_documents(docs)` 拆分了文檔
RCtext_splited = text_splitter.split_documents(docs)

# 使用 defaultdict 來分組並基於 `source_id` 添加 `seq_num`
grouped_docs = defaultdict(list)
for doc in RCtext_splited:
    source_id = doc.metadata.get("source_id")
    grouped_docs[source_id].append(doc)

# 為每個 `source_id` 組的文件添加 `seq_num`
for source_id, documents in grouped_docs.items():
    for seq_num, doc in enumerate(documents, start=1):
        doc.metadata["seq_num"] = seq_num

# 現在 `RCtext_splited` 中的每個 Document 都根據 `source_id` 有了正確的 `seq_num`


load_dotenv()
client = QdrantClient(url=os.getenv("qdrant_url"), timeout=60)

### 創一個 collection
# 設定預設 sparse model
client.set_sparse_model(embedding_model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")

# 如果不存在，創建 collection
if not client.collection_exists("finance_recursive_chunk_1500"):
    client.create_collection(
        collection_name="finance_recursive_chunk_1500",
        vectors_config=VectorParams(
            size=1024, distance=Distance.COSINE
        ),  # BAAI/bge-m3 的參數設置 (FastEmbed 沒有所以自己設)
        sparse_vectors_config=client.get_fastembed_sparse_vector_params(),  # 前面設定的預設 sparse model 參數
        on_disk_payload=True,
    )

### 做一個vector store
vector_store = QdrantVectorStore(
    client=client,
    collection_name="finance_recursive_chunk_1500",
    embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-m3"),
    sparse_embedding=FastEmbedSparse(
        model_name="Qdrant/bm42-all-minilm-l6-v2-attentions"
    ),
    sparse_vector_name="fast-sparse-bm42-all-minilm-l6-v2-attentions",
    retrieval_mode=RetrievalMode.HYBRID,
)

# #%%
# def batch(iterable, n=10):
#     """
#     將 iterable 分成每批 n 個的批次
#     """
#     it = iter(iterable)
#     while True:
#         batch = list(islice(it, n))
#         if not batch:
#             break
#         yield batch

# # 批量上傳文件
# batch_size = 1 # 設定批量大小
# for batch_docs in tqdm(batch(RCtext_splited, batch_size)):
#     vector_store.add_documents(batch_docs)

with tqdm(total=len(RCtext_splited), desc="Processing documents") as pbar:
    for docs in RCtext_splited:
        source_id = docs.metadata.get("source_id", "N/A")
        seq_num = docs.metadata.get("seq_num", "N/A")
        pbar.set_description(f"Processing source_id: {source_id}, seq_num: {seq_num}")
        vector_store.add_documents([docs])
        pbar.update(1)
