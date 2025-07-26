import os
from tqdm import tqdm
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import time
from utils.split_article_and_metadata import split_article_and_metadata
from utils.chunk_filter import is_irrelevant_chunk
import diskcache

def store_documents(segmentation_executor, embedding_executor, data_dir):
    chunked_documents = []
    doc_level_documents = []
    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    total_articles = 0

    segmentation_cache = diskcache.Cache('segmentation_cache.db')
    embedding_cache = diskcache.Cache('embedding_cache.db')

    for txt_file in txt_files:
        file_path = os.path.join(data_dir, txt_file)
        try:
            parsed_data = split_article_and_metadata(file_path)
        except Exception as e:
            print(f"[오류] {txt_file} 파일 파싱 실패: {e}")
            continue
        print(f"[{txt_file}]에서 {len(parsed_data)}개 기사 파싱 완료.")
        total_articles += len(parsed_data)
        for idx, (metadata, full_text) in enumerate(tqdm(parsed_data, desc=f"{txt_file} 뉴스 기사 분할 중")):
            request_data = {
    "alpha": -100,
    "segCnt": -1,
    "text": full_text,
    "postProcess": False
}
            filtered_chunk_texts = []
            # segmentation 캐시 적용
            if full_text in segmentation_cache:
                segmented_chunks = segmentation_cache[full_text]
            else:
                try:
                    segmented_chunks = segmentation_executor.execute(request_data)
                    segmentation_cache[full_text] = segmented_chunks
                except Exception as e:
                    print(f"  Segmentation Error: {e}")
                    segmented_chunks = 'Error'
            if segmented_chunks != 'Error':
                for chunk in segmented_chunks:
                    chunk_text = chunk if isinstance(chunk, str) else ' '.join(chunk)
                    if is_irrelevant_chunk(chunk_text):
                        continue
                    chunked_documents.append({
                        "text": chunk_text,
                        "metadata": metadata,
                        "type": "chunk"
                    })
                    filtered_chunk_texts.append(chunk_text)
            else:
                print(f"  문단 나누기 API 호출 실패")
            filtered_full_text = '\n'.join(filtered_chunk_texts)
            doc_level_documents.append({
                "text": filtered_full_text,
                "metadata": metadata,
                "type": "doc"
            })
    print(f"총 {total_articles}개의 뉴스 기사 파싱 완료.")

    # 전체 임베딩 캐시 적용
    for didx, doc in enumerate(tqdm(doc_level_documents, desc="전체 임베딩 생성 중")):
        text = doc['text']
        truncated_text = text[:8192] if len(text) > 8192 else text
        if truncated_text in embedding_cache:
            doc["embedding"] = embedding_cache[truncated_text]
        else:
            try:
                request_json = {"text": truncated_text}
                response_data = embedding_executor.execute(request_json)
                doc["embedding"] = response_data
                embedding_cache[truncated_text] = response_data
            except Exception as e:
                print(f"  전체 임베딩 오류: {e}")

    # 청크 임베딩 캐시 적용
    for cidx, chunked_document in enumerate(tqdm(chunked_documents, desc="청크 임베딩 생성 중")):
        text = chunked_document['text']
        if text in embedding_cache:
            chunked_document["embedding"] = embedding_cache[text]
        else:
            try:
                request_json = {"text": text}
                response_data = embedding_executor.execute(request_json)
                chunked_document["embedding"] = response_data
                embedding_cache[text] = response_data
            except Exception as e:
                print(f"  청크 임베딩 오류: {e}")

    collection_name = "NewsPickStock"
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"기존 컬렉션 삭제 완료.")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="metadata", dtype=DataType.JSON),
        FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=10)
    ]
    schema = CollectionSchema(fields, description="뉴스 기사 및 주식 정보")
    collection = Collection(name=collection_name, schema=schema)

    batch_size = 500
    # 전체 임베딩 batch insert
    batch_texts, batch_embeddings, batch_metadatas, batch_types = [], [], [], []
    for didx, doc in enumerate(tqdm(doc_level_documents, desc="전체 임베딩 DB 저장 중")):
        if "embedding" in doc:
            text = doc['text']
            batch_texts.append(doc['text'])
            batch_embeddings.append(doc['embedding'])
            batch_metadatas.append(doc['metadata'])
            batch_types.append(doc['type'])
            if len(batch_texts) == batch_size:
                entities = [batch_texts, batch_embeddings, batch_metadatas, batch_types]
                try:
                    collection.insert(entities)
                except Exception as e:
                    print(f"[Milvus insert 예외] {e}", flush=True)
                batch_texts, batch_embeddings, batch_metadatas, batch_types = [], [], [], []
    # 마지막 남은 것 insert
    if batch_texts:
        entities = [batch_texts, batch_embeddings, batch_metadatas, batch_types]
        try:
            collection.insert(entities)
        except Exception as e:
            print(f"[Milvus insert 예외] {e}", flush=True)
    print(f"전체 임베딩 저장 완료.")

    # 청크 임베딩 batch insert
    batch_texts, batch_embeddings, batch_metadatas, batch_types = [], [], [], []
    for didx, item in enumerate(tqdm(chunked_documents, desc="청크 임베딩 DB 저장 중")):
        if "embedding" in item:
            text = item['text']
            if len(text) > 9000:
                print(f"[경고] 청크 임베딩 본문 길이 초과({len(text)}자):\n앞500: {text[:500]}\n... [중략] ...\n뒤500: {text[-500:]}", flush=True)
                continue  # 저장하지 않음
            batch_texts.append(item['text'])
            batch_embeddings.append(item['embedding'])
            batch_metadatas.append(item['metadata'])
            batch_types.append(item['type'])
            if len(batch_texts) == batch_size:
                entities = [batch_texts, batch_embeddings, batch_metadatas, batch_types]
                try:
                    collection.insert(entities)
                except Exception as e:
                    print(f"[Milvus insert 예외] {e}", flush=True)
                batch_texts, batch_embeddings, batch_metadatas, batch_types = [], [], [], []
    # 마지막 남은 것 insert
    if batch_texts:
        entities = [batch_texts, batch_embeddings, batch_metadatas, batch_types]
        try:
            collection.insert(entities)
        except Exception as e:
            print(f"[Milvus insert 예외] {e}", flush=True)
    print(f"청크 임베딩 저장 완료.")

    print("데이터 저장 및 인덱스 생성 시작...")
    index_params = {
        "metric_type": "IP", "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 200}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    utility.index_building_progress(collection_name)
    print("인덱스 생성 완료 및 컬렉션 메모리 로드 완료.")
    collection.load() 