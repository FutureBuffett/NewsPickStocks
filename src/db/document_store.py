import os
from tqdm import tqdm
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

def store_documents(segmentation_executor, embedding_executor):
    """문서를 처리하고 Milvus에 저장합니다."""
    data_dir = 'data'
    chunked_documents = []

    try:
        file_list = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
        if not file_list:
            print(f"'{data_dir}' 디렉토리에서 .txt 파일을 찾을 수 없습니다.")
            return

        for filename in tqdm(file_list, desc="텍스트 파일 처리 중"):
            txt_file_path = os.path.join(data_dir, filename)
            try:
                with open(txt_file_path, 'r', encoding='utf-8') as f:
                    full_text = f.read()
            except FileNotFoundError:
                print(f"오류: {txt_file_path} 파일을 찾을 수 없습니다.")
                continue

            if full_text:
                try:
                    request_data = {
                        "postProcessMaxSize": 1000, "alpha": 0.0, "segCnt": -1,
                        "postProcessMinSize": 100, "text": full_text, "postProcess": True
                    }
                    segmented_paragraphs = segmentation_executor.execute(request_data)

                    if segmented_paragraphs != 'Error':
                        for paragraph in segmented_paragraphs:
                            chunked_documents.append({
                                "source": txt_file_path,
                                "text": ' '.join(paragraph)
                            })
                    else:
                        print(f"'{txt_file_path}'에 대한 문단 나누기 API 호출에 실패했습니다.")
                except Exception as e:
                    print(f"'{txt_file_path}' 처리 중 오류 발생: {e}")
    except FileNotFoundError:
        print(f"오류: '{data_dir}' 디렉토리를 찾을 수 없습니다.")
        return
    
    if not chunked_documents:
        print("처리할 문서가 없습니다. 임베딩 및 저장을 건너뜁니다.")
        return

    for chunked_document in tqdm(chunked_documents, desc="문서 임베딩 중"):
        try:
            request_json = {"text": chunked_document['text']}
            response_data = embedding_executor.execute(request_json)
            chunked_document["embedding"] = response_data
        except ValueError as e:
            print(f"임베딩 API 오류: {e}")
        except Exception as e:
            print(f"예상치 못한 오류: {e}")

    collection_name = "NewsPickStock"
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"기존 컬렉션 '{collection_name}'을(를) 삭제했습니다.")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=3000),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=9000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)
    ]
    schema = CollectionSchema(fields, description="뉴스 기사 및 주식 정보")
    collection = Collection(name=collection_name, schema=schema)

    for item in tqdm(chunked_documents, desc="Milvus에 데이터 저장 중"):
        if "embedding" in item:
            entities = [ [item['source']], [item['text']], [item['embedding']] ]
            collection.insert(entities)

    print("데이터 저장이 완료되었습니다.")

    index_params = {
        "metric_type": "IP", "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 200}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    utility.index_building_progress(collection_name)
    print("인덱스 생성이 완료되었습니다.")
    collection.load()
    print("컬렉션을 메모리에 로드했습니다.") 