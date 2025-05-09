from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from common.system_prompt import system_message_1, system_message_2,system_message_3
from openai import Client
import os
from dotenv import load_dotenv

import common.prompt_template

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = Client(api_key=OPENAI_API_KEY)

def search(question):
#     파일의 위치 기준으로 절대경로로 불러오기
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INDEX_PATH = os.path.join(BASE_DIR , "src" , "db" , "qas.index")

# DB지정
    db = FAISS.load_local(
        INDEX_PATH,
        OpenAIEmbeddings(api_key=OPENAI_API_KEY),
        allow_dangerous_deserialization=True
    )
    # 유사도로 검색(코사인 유사도 검색 - 방향에 따른 검색)
    # 동일 방향 : 비슷한거
    # 직각 방향 : 상관 없는거
    # 반대 방향 : 반대 내용
    result = db.search(question , search_type="similarity")
    return result[0].metadata
# DB검색 결과와 질문을 바탕으로 GPT에게 답변을 생성
# context : FAISS DB에서 얻은 결과
# question : 고객이 질문한 내용
def generate_answer(context , question) :
    context_join = f"""
    Q:{context['question']}
    A:{context['answer']}
    """
    prompt = common.prompt_template.prompt_qa.format(
        context = context_join,question= question
    )
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages=[
            {"role":"system", "content" : system_message_1 },
            {"role":"user" , "content" : prompt}
        ],
        temperature=0
    )
    output= response.choices[0].message.content
    return output
if __name__ == '__main__':
    question_1 = "RAG는 무엇인가요?"
    qa_1 = search(question_1)
    print(qa_1['question'])
    print(qa_1['answer'])
    print(generate_answer(qa_1,question_1))
    print("=============================================")
    question_2 = "네이버 영화 리뷰  분석 서비스는 뭔가요"
    qa_2 = search(question_2)
    print(qa_2['question'])
    print(qa_2['answer'])
    print(generate_answer(qa_2, question_2))
    print("=============================================")
    question_3 = "남은 휴가는 어떻게 되나요?"
    qa_3 = search(question_3)
    print(qa_3['question'])
    print(qa_3['answer'])
    print(generate_answer(qa_3, question_3))
    print("=============================================")
