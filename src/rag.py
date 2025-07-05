from pymilvus import connections, Collection, utility

def answer_question(question, embedding_executor, completion_executor):
    collection_name = "NewsPickStock"
    if not utility.has_collection(collection_name):
        return f"'{collection_name}' 컬렉션이 존재하지 않습니다. 먼저 문서를 처리하고 저장해주세요.", []

    collection = Collection(collection_name)
    collection.load()

    query_vector = embedding_executor.execute({"text": question})

    search_params = {"metric_type": "IP", "params": {"ef": 64}}
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=10,
        output_fields=["source", "text"]
    )

    reference = [{"distance": hit.distance, "source": hit.entity.get("source"), "text": hit.entity.get("text")} for hit in results[0]]

    preset_texts = [
        {"role": "system", "content": "- 너의 역할은 사용자의 질문에 reference를 바탕으로 답변하는거야. \n- 너가 가지고있는 지식은 모두 배제하고, 주어진 reference의 내용만을 바탕으로 답변해야해. \n- 답변의 출처가 되는 html의 내용인 'source'도 답변과 함께 {url:}의 형태로 제공해야해. \n- 만약 사용자의 질문이 reference와 관련이 없다면, {제가 가지고 있는 정보로는 답변할 수 없습니다.}라고만 반드시 말해야해."}
    ]
    for ref in reference:
        preset_texts.append({"role": "user", "content": f"reference: {ref['text']}, url: {ref['source']}"})
    
    preset_texts.append({"role": "user", "content": question})

    request_data = {
        'messages': preset_texts,
        'topP': 0.8, 'topK': 0, 'maxTokens': 1024,
        'temperature': 0.5, 'repetitionPenalty': 1.1,
        'stop': [], 'includeAiFilters': True, 'seed': 0
    }

    answer = completion_executor.execute(request_data)
    
    return answer, reference 