[33mcommit e2d87e9139a7ece8bcba8d89febd5a8ffcf2becd[m[33m ([m[1;36mHEAD[m[33m -> [m[1;32mdev[m[33m, [m[1;31morigin/dev[m[33m)[m
Author: basisp <144201139+basisp@users.noreply.github.com>
Date:   Sun Jul 20 22:42:55 2025 +0900

    fix: RAG  속도 향상

[33mcommit 989ffbaad13ed77f681a1bff0ade173bb441b4db[m
Author: basisp <144201139+basisp@users.noreply.github.com>
Date:   Sat Jul 19 12:23:00 2025 +0900

    fix: utf-8 에서 utf-16으로 변경

[33mcommit 975956187e31621fa322b31789d74763579016b1[m
Author: basisp <144201139+basisp@users.noreply.github.com>
Date:   Thu Jul 17 18:29:48 2025 +0900

    Feat: 사용자 뉴스에 정규화를 적용합니다
    
    * feat: 사용자 입력 뉴스 정규화 적용
    
    * fix: milvus doc['text'] 필드 최대 길이 9000 -> 65535 로 수정

[33mcommit 67e4e90d1fb7364b24e1552549a2e009f4722842[m
Author: basisp <144201139+basisp@users.noreply.github.com>
Date:   Wed Jul 16 20:11:03 2025 +0900

    Feat: 벡터DB 파이프라인 최적화 및 기능 개선
    
    * feat: poetry에 diskcache 추가
    
    * fix: data 변경
    
    * feat: 문단 나누기 테스트앱에서 서비스 앱으로 변경
    
    * feat: hybrid_search 종목 중복시 유사도 제외, 최소 3개 이상의 종목이 나오게 수정
    
    * feat: 로컬 캐시에서 디스크 캐시로 변경, 벡터 DB 저장시 배치 전략 적용
    
    * fix: 메타데이터 key값 추가

[33mcommit 0b1d43e43a0ac050b06b393d0356a8d760c4a8ae[m
Author: basisp <144201139+basisp@users.noreply.github.com>
Date:   Mon Jul 14 20:09:24 2025 +0900

    Feat: RAG 고도화
    
    * feat: data 형식 변경
    
    * feat: 임베딩 모델 테스트 앱에서 서비스 앱으로 변경
    
    * feat: 10토큰 이하 || 특정 단어 포함 청크 삭제 구현
    
    * feat: data 를 메타데이터와 본문 으로 분리
    
    * feat: 전체 임베딩 + 부분 임베딩 저장 & 검색 구현

[33mcommit 3e527f84421db8f59e88e4c4dff77a6ed1433254[m
Author: basisp <144201139+basisp@users.noreply.github.com>
Date:   Wed Jul 9 17:16:02 2025 +0900

    fix: 임베딩 모델 v1에서 v2로 변경

[33mcommit 8e030ac2f015b2dd5c3e7878d82ed76eae3e531f[m
Author: basisp <144201139+basisp@users.noreply.github.com>
Date:   Sat Jul 5 19:23:30 2025 +0900

    Feat: 기본 RAG 시스템 구현
    
    * feat: 초기 인프라 설정
    
    * feat: 외부 API(Clova Studio)와 통신하는 Executor 클래스 추가
    
    * feat: poetry 설정 추가
    
    * feat: 데이터 처리 파이프라인 구현
    
    * feat: RAG 답변 구현
    
    * feat: Streamlit 앱 구현
    
    * feat: gitignore 수정

[33mcommit 5ad163bb680c421879f411d9f9063c0c1a643e5e[m[33m ([m[1;31morigin/main[m[33m, [m[1;31morigin/HEAD[m[33m, [m[1;32mmain[m[33m)[m
Author: basisp <144201139+basisp@users.noreply.github.com>
Date:   Sat Jul 5 12:13:22 2025 +0900

    Initial commit
