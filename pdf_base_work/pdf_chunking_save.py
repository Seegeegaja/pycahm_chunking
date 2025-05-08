from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

# PDF 변환
def attention_policy_chunking():
    # 훈련규정 읽어오기
    doc_policy = PyPDFLoader("../src/attendance_policy.pdf")
    # 전체 페이지를 통째로 split
    doc_policy_load = doc_policy.load()
    # print(doc_policy_load)
#     chunking 사이즈 규정
    policy_splitter = CharacterTextSplitter(
        separator=".\n",
        chunk_size=100,
        chunk_overlap=50,
        length_function=len
    )
#     규정 PDF 를 chunking하기
    att_chunk_docs= policy_splitter.split_documents(doc_policy_load)
#     규정 확인 해 보기
#     print(len(att_chunk_docs))
#     print(att_chunk_docs[1].page_content)
#     text 형식으로 되어있는 출렷인정 일수 청킹하기
    with open("../src/attention_condition.txt","r", encoding="UTF-8") as f:
        att_chunk_txt = f.read()
    att_text_doc = [Document(page_content=att_chunk_txt)]
    # print(att_text_doc)
#     chunking 사이즈 규정
    policy_splitter = RecursiveCharacterTextSplitter(
        separators=["\nㆍ","ㆍ"],
        chunk_size=100,
        chunk_overlap=30,
        length_function=len
    )
#     출석 인정일수 chunking
    attention_table = policy_splitter.split_documents(att_text_doc)
    # 결과 확인
    # print(attention_table)
# 두개 문서 청킹결과를 하나로 합치기
    raw_source = att_chunk_docs + attention_table
    return raw_source
if __name__ == '__main__':
    attention_policy_chunking()