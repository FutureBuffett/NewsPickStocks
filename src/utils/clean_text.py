import re

def clean_text(text):
    # 한글, 영문, 숫자, 공백, 마침표, 퍼센트, 줄바꿈만 남김
    text = re.sub(r'[^\w가-힣\n %.]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip() 