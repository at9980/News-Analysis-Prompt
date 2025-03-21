# 투자 어시스턴트 프로젝트

이 프로젝트는 **투자 어시스턴트** 애플리케이션으로, 뉴스 데이터 기반의 질의 응답 시스템을 구축하기 위해 만들어졌습니다. Langchain 커뮤니티 라이브러리, Transformers, Torch, 그리고 Streamlit을 활용하여, 사용자의 질문에 대해 관련 뉴스 요약을 검색하고, 이를 기반으로 텍스트 생성 모델이 답변을 제공합니다.

## 주요 기능

- **뉴스 데이터 관리**: `data.py` 파일에 정의된 뉴스 데이터를 활용합니다.
- **문서 검색 (Retriever)**: 
  - BM25 및 FAISS 기반의 검색기를 사용하여 뉴스 요약을 검색합니다.
  - 두 검색기의 결과를 앙상블(Ensemble) 방식으로 결합하여 더 정교한 검색 결과를 도출합니다.
- **텍스트 생성 (Generation)**:
  - Transformers 기반의 텍스트 생성 파이프라인을 사용해 사용자의 질문에 대한 답변을 생성합니다.
  - 검색된 뉴스 요약을 참고하여 프롬프트를 구성하고, 모델로부터 답변을 도출합니다.
- **인터랙티브 UI**: Streamlit을 통해 사용자와의 대화형 채팅 인터페이스를 제공합니다.

## 파일 구조

your_project/ ├── app.py # Streamlit을 통한 사용자 인터페이스 및 메인 애플리케이션 ├── data.py # 뉴스 데이터(문서)를 정의한 파일 ├── generation.py # 텍스트 생성 모델 초기화 및 프롬프트 생성 관련 함수들 └── retriever.py # BM25, FAISS, Ensemble Retriever를 초기화하고 검색 함수 제공

shell
## 설치 및 실행

### 1. 의존성 패키지 설치

다음 패키지들을 설치합니다:

```bash
pip install streamlit langchain_community transformers torch
2. 애플리케이션 실행
터미널에서 아래 명령어를 실행하여 Streamlit 애플리케이션을 시작합니다.

bash
streamlit run app.py
웹 브라우저가 자동으로 열리며, 채팅 인터페이스를 통해 질문을 입력하면 해당 질문에 대한 답변이 생성됩니다.

코드 설명
data.py
역할: 뉴스 데이터(문서)를 JSON 형태로 저장합니다.
구성: 각 뉴스 항목은 기업명, 날짜, 문서 카테고리, 요약, 그리고 주요 이벤트 정보를 포함합니다.
retriever.py
역할: 뉴스 데이터의 요약을 기반으로 BM25와 FAISS 검색기를 초기화하고, 두 검색기를 앙상블 방식으로 결합합니다.
주요 함수:
search(query: str): 주어진 쿼리에 대해 관련 뉴스 요약 문서를 검색하여 반환합니다.
generation.py
역할: 텍스트 생성 파이프라인을 설정하고, 검색된 뉴스 요약을 활용해 최종 프롬프트를 구성한 후 답변을 생성합니다.
주요 함수:
sllm_generate(query: str) -> str: 주어진 프롬프트를 기반으로 텍스트를 생성합니다.
prompt_and_generate(query: str, search_func) -> str: 검색된 뉴스 요약을 포함하여 프롬프트를 구성하고 최종 답변을 생성합니다.
app.py
역할: Streamlit을 통해 사용자와 상호작용하는 채팅 인터페이스를 제공합니다.
주요 기능:
세션 상태를 통해 대화 내역을 관리합니다.
사용자의 질문 입력을 받고, prompt_and_generate 함수를 통해 답변을 생성합니다.
생성된 답변을 채팅 메시지로 출력합니다.
기여 방법
이슈 제기: 프로젝트 사용 중 발견한 문제점이나 개선 사항을 이슈로 남겨주세요.
Pull Request: 개선된 코드를 작성하신 후 Pull Request를 보내주시면 검토 후 반영하겠습니다.
