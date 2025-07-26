from pymilvus import connections, Collection, utility
from concurrent.futures import ThreadPoolExecutor
import json
from pymilvus import Collection, utility
from datetime import datetime

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



def hybrid_search(
    news_text: str,
    segmentation_executor,
    embedding_executor,
    topk: int = 10,
    doc_weight: float = 1.0,
    chunk_weight: float = 0.7,
    doc_vector=None,
    chunk_vectors=None,
    filtered_full_text=None,
    filtered_chunks=None,
    date_window: int = 10
):
    collection_name = "NewsPickStock"
    if not utility.has_collection(collection_name):
        return f"'{collection_name}' 컬렉션이 존재하지 않습니다. 먼저 문서를 처리하고 저장해주세요.", []

    collection = Collection(collection_name)
    collection.load()

    # 전체 임베딩
    if doc_vector is None:
        doc_vector = embedding_executor.execute({"text": filtered_full_text or news_text})

    # 청크 임베딩
    if chunk_vectors is None:
        chunks = filtered_chunks or segmentation_executor.execute({"text": news_text})
        chunk_vectors = [embedding_executor.execute({"text": chunk}) for chunk in chunks if isinstance(chunk, str) and len(chunk) > 10]

    search_params = {"metric_type": "IP", "params": {"ef": 32}}

    # 기업 다양성 확보 (문서)
    doc_topk = topk
    while True:
        doc_results = collection.search(
            data=[doc_vector],
            anns_field="embedding",
            param=search_params,
            limit=doc_topk,
            output_fields=["metadata", "type"],
            expr="type == 'doc'"
        )[0]
        doc_companies = set((hit.entity.get("metadata") or {}).get("company") for hit in doc_results)
        if len(doc_companies) >= 3 or doc_topk >= 50:
            break
        doc_topk += 10

    # 기업 다양성 확보 (청크)
    chunk_topk = topk
    def search_chunk(chunk_vector):
        return collection.search(
            data=[chunk_vector],
            anns_field="embedding",
            param=search_params,
            limit=chunk_topk,
            output_fields=["metadata", "type"],
            expr="type == 'chunk'"
        )[0]

    with ThreadPoolExecutor() as executor:
        results = executor.map(search_chunk, chunk_vectors)
        chunk_results = [hit for result in results for hit in result]

    chunk_companies = set((hit.entity.get("metadata") or {}).get("company") for hit in chunk_results)
    while len(chunk_companies) < 3 and chunk_topk <= 50:
        chunk_topk += 10
        with ThreadPoolExecutor() as executor:
            results = executor.map(search_chunk, chunk_vectors)
            chunk_results = [hit for result in results for hit in result]
        chunk_companies = set((hit.entity.get("metadata") or {}).get("company") for hit in chunk_results)

    # 결과 통합 및 prices 구조 생성
    company_map = {}

    for hit in doc_results + chunk_results:
        entity = hit.entity
        meta = entity.metadata if hasattr(entity, "metadata") else {}
        company = meta.get("company")
        base_date_str = meta.get("base_date")
        date_str = meta.get("date")

        if not company or not base_date_str or not date_str:
            continue

        try:
            base_date = datetime.strptime(base_date_str, "%Y-%m-%d")
            date = datetime.strptime(date_str, "%Y-%m-%d")
        except Exception as e:
            print(f"[⚠️ 날짜 파싱 실패] meta: {meta} | 오류: {e}")
            continue

        delta_days = (date - base_date).days
        if abs(delta_days) > date_window:
            continue

        # ✅ company + base_date 를 고유 키로 사용
        key = f"{company}__{base_date_str}"
        score = hit.distance * (doc_weight if entity.get("type") == "doc" else chunk_weight)

        if key not in company_map:
            company_map[key] = {
                "company": company,
                "base_date": base_date_str,
                "prices": [],
                "score": 0.0
            }

        company_map[key]["prices"].append({
            "date": date_str,
            "open": meta.get("open"),
            "high": meta.get("high"),
            "low": meta.get("low"),
            "close": meta.get("close"),
            "volume": meta.get("volume"),
            "url": meta.get("url")
        })

        company_map[key]["score"] += score

    # 날짜순 정렬
    for comp in company_map.values():
        comp["prices"] = sorted(comp["prices"], key=lambda x: x["date"])

    ranked = sorted(company_map.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:3]