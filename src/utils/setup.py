import os
from dotenv import load_dotenv

from executors.segmentation_executor import SegmentationExecutor
from executors.embedding_executor import EmbeddingExecutor
from executors.completion_executor import CompletionExecutor

load_dotenv()

def setup_executors():
    """API 실행자들을 초기화하고 반환합니다."""
    api_key = os.getenv('CLOVA_API_KEY')
    if not api_key:
        raise ValueError("CLOVA_API_KEY 환경 변수가 설정되지 않았습니다.")

    segmentation_executor = SegmentationExecutor(
        host='clovastudio.stream.ntruss.com',
        api_key=api_key,
        request_id='c2a7bd64bf46430faaa11edda36c63a9'
    )
    embedding_executor = EmbeddingExecutor(
        host='clovastudio.stream.ntruss.com',
        api_key=api_key,
        request_id='93c7a569f2914cc9859f4cb01780e70d'
    )
    completion_executor = CompletionExecutor(
        host='https://clovastudio.stream.ntruss.com',
        api_key=api_key,
        request_id='bde424d9851d426ab52096633744b993'
    )
    return segmentation_executor, embedding_executor, completion_executor 