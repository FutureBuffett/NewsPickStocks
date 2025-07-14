# 불필요 청크 키워드 리스트 + 향후 추가 필요
IRRELEVANT_KEYWORDS = [
    "후원", "광고", "구독", "좋아요", "문의", "도와주세요", "저작권", "Copyright", "모금", "뉴스",
]

def is_irrelevant_chunk(chunk, min_length=10):
    for keyword in IRRELEVANT_KEYWORDS:
        if keyword in chunk:
            return True
    if len(chunk.strip()) < min_length:
        return True
    return False 