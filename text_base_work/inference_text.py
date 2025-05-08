import json
import os
from typing import List

from dotenv import load_dotenv
from langchain.chains.question_answering.map_rerank_prompt import output_parser
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from common.prompt_template import prompt_text_base
from download_text_data import get_text_data
from common.base_calss import Output,QA

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#     Json 변화 파서 설정
output_parser = PydanticOutputParser(pydantic_object=Output)

def inference_json(info_data):
    prompt = PromptTemplate(
        template=prompt_text_base,
        input_variables=['infor_text_data'],
        partial_variables={"format_instruction": output_parser.get_format_instructions()}
    )
    model = ChatOpenAI(
        temperature=0,
        api_key=OPENAI_API_KEY,
        model_name="gpt-4o-mini"
    )
    chain = (prompt | model | output_parser)
    output = chain.invoke({"infor_text_data":info_data})
    return output

if __name__ == '__main__':
    info_data = get_text_data()
    result = inference_json(info_data)
    print(json.dumps(result.dict(), indent=2,ensure_ascii=False))
