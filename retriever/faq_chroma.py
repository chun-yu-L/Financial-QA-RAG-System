from utils.Database import ChromaDB
import json


chroma = ChromaDB()
vector_store = chroma.get_vector_store(collection_name='faq_e5_large', embedding_model='intfloat/multilingual-e5-large')

with open("./競賽資料集/dataset/preliminary/questions_example.json", "r") as q:
    question_set = json.load(q)

faq_data = [item for item in question_set['questions'] if item['category'] == 'faq']

answers = []

# for Q in question_set['questions']:
for Q in faq_data:
    filter_conditions = {
        "$and": [
            {"category": {"$eq": f"{Q['category']}"}},
            {"section_id": {"$in": [str(i) for i in Q['source']]}}
        ]
    }
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
        'retrieve': int(query_result['section_id']),
        'category': query_result['category']
    })

with open("./faq_retrival_result.json", "w") as Output:
    json.dump({"answers": answers},Output,ensure_ascii=False,indent=4)
        
