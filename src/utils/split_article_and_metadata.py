import os

def split_article_and_metadata(file_path):
    keys = [
        "company", "ticker", "sector", "category", "subcategory",
        "date", "open", "high", "low", "close", "volume", "url"
    ]
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        parts = line.split('\t')
        if len(parts) < 13:
            i += 1
            continue
        metadata = dict(zip(keys, parts[:12]))
        text = parts[12]
        # 본문이 큰따옴표로 시작하지만 끝나지 않은 경우
        if text.startswith('"') and not text.endswith('"'):
            text_lines = [text.lstrip('"')]
            i += 1
            while i < len(lines):
                next_line = lines[i].rstrip('\n')
                # 본문이 큰따옴표로 끝나는 줄을 만날 때까지 추가
                if next_line.endswith('"'):
                    text_lines.append(next_line.rstrip('"'))
                    break
                else:
                    text_lines.append(next_line)
                i += 1
            text = '\n'.join(text_lines)
        else:
            text = text.strip('"')
        results.append((metadata, text))
        i += 1
    return results

if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), '../../example.txt')
    file_path = os.path.abspath(file_path)
    data = split_article_and_metadata(file_path)
    for i, (meta, text) in enumerate(data):
        print(f"==== {i+1}번째 데이터 ====")
        print("[메타데이터]")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        print("[본문]")
        print(text)
        print("====================\n") 