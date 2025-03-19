# generation.py

import transformers
import torch

# 텍스트 생성 모델 및 파이프라인 초기화
model_id = "42dot/42dot_LLM-SFT-1.3B"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16}
)
pipeline.model.eval()

def sllm_generate(query: str) -> str:
    """
    주어진 프롬프트를 바탕으로 텍스트를 생성합니다.
    """
    answer = pipeline(
        query,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.2,
    )
    return answer[0]['generated_text'][len(query):]

def prompt_and_generate(query: str, search_func) -> str:
    """
    검색된 문서를 참고하여 최종 프롬프트를 구성하고 응답을 생성합니다.
    
    Parameters:
        query (str): 사용자 질문.
        search_func (function): 문서 검색 함수.
    
    Returns:
        str: 생성된 답변.
    """
    docs = [doc for doc in search_func(query)]
    prompt = f"""아래 질문을 기반으로 검색된 뉴스를 참고하여 질문에 대한 답변을 생성하시오.

질문: {query}
"""
    for i, doc in enumerate(docs):
        prompt += f"뉴스{i+1}\n"
        prompt += f"요약: {doc.page_content}\n\n"
    prompt += "답변: "
    
    answer = sllm_generate(prompt)
    return answer
