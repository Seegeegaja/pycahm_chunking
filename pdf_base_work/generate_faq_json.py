import json
import tiktoken
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import PydanticOutputParser

from common.prompt_template import prompt_pdf_base
from openai import Client
from common.base_calss import Output,QA
from pdf_base_work.pdf_chunking_save import attention_policy_chunking
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# embedding 문장을 토큰단위로 나누는 작업
embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
client = Client(api_key=OPENAI_API_KEY)

# json변환
output_parser = PydanticOutputParser(pydantic_object=Output)

def generate_faq_json(info_detail):
    prompt = prompt_pdf_base.format(
        format_instruction = output_parser.get_format_instructions(),
        info_detail = info_detail
    )
    # json타입으로 얻어오기
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are a helpful assistant"},
            {"role": "user" , "content":prompt}
        ],
        temperature=0,
        response_format={"type":"json_object"}
    )
    output= response.choices[0].message.content
    # json변환
    output_json = json.loads(output)
    return output_json
# vector DB에 저장 하고 검색 해 보기
def vector_save_and_search_test():
    raw_source = attention_policy_chunking()

    # VectorDB FAISS 에 저장하기
    db = FAISS.from_documents(raw_source, embedding_model)
    # 저장 후에 유사도의 따른 검색 결과 리턴할 리트리버를 정의
    retriever = db.as_retriever()
    # 질문
    question_1 = retriever.invoke("50%미만 출석은?")
    print(f"50% 미만 출석 : {question_1[0]}")
    question_2 = retriever.invoke("결혼은 몇일이 인정?")
    print('==============================================')
    print(f"결혼은 몇일이 인정 : {question_2[0]}")

if __name__ == '__main__':
    vector_save_and_search_test()