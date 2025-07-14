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

def hybrid_search(news_text, segmentation_executor, embedding_executor, topk=10, doc_weight=1.0, chunk_weight=0.7, doc_vector=None, chunk_vectors=None, filtered_full_text=None, filtered_chunks=None):
    collection_name = "NewsPickStock"
    if not utility.has_collection(collection_name):
        return f"'{collection_name}' 컬렉션이 존재하지 않습니다. 먼저 문서를 처리하고 저장해주세요.", []

    collection = Collection(collection_name)
    collection.load()

    # 전체 임베딩
    if doc_vector is None:
        if filtered_full_text is not None:
            doc_vector = embedding_executor.execute({"text": filtered_full_text})
        else:
            doc_vector = embedding_executor.execute({"text": news_text})

    # 청크 임베딩
    if chunk_vectors is None:
        if filtered_chunks is not None:
            chunk_vectors = [embedding_executor.execute({"text": chunk}) for chunk in filtered_chunks if len(chunk) > 10]
        else:
            segmented_chunks = segmentation_executor.execute({"text": news_text})
            chunk_vectors = [embedding_executor.execute({"text": chunk}) for chunk in segmented_chunks if isinstance(chunk, str) and len(chunk) > 10]

    search_params = {"metric_type": "IP", "params": {"ef": 64}}


    # doc 검색
    doc_results = collection.search(
        data=[doc_vector],
        anns_field="embedding",
        param=search_params,
        limit=topk,
        output_fields=["metadata", "type"],
        expr="type == 'doc'"
    )[0]

    # chunk 검색
    chunk_results = []
    for chunk_vector in chunk_vectors:
        chunk_results.extend(collection.search(
            data=[chunk_vector],
            anns_field="embedding",
            param=search_params,
            limit=topk,
            output_fields=["metadata", "type"],
            expr="type == 'chunk'"
        )[0])

    from collections import defaultdict
    stock_scores = defaultdict(float)
    seen = set()
    for hit in doc_results:
        meta = hit.entity.get("metadata")
        print("[hybrid_search] doc meta:", meta)
        stock = meta.get("company") if meta else None
        key = (stock, meta.get("date") if meta else None)
        if key not in seen:
            stock_scores[stock] += hit.distance * doc_weight
            seen.add(key)
    for hit in chunk_results:
        meta = hit.entity.get("metadata")
        print("[hybrid_search] chunk meta:", meta)
        stock = meta.get("company") if meta else None
        key = (stock, meta.get("date") if meta else None)
        if key not in seen:
            stock_scores[stock] += hit.distance * chunk_weight
            seen.add(key)
    ranked = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked 