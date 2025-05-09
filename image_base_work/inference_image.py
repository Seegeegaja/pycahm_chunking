import json

from langchain_core.output_parsers import PydanticOutputParser
from openai import Client
import os
from dotenv import load_dotenv

from common.prompt_template import prompt_image_base
from image_base_work.download_image_data import get_urls
from common.base_calss import Output,QA
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = Client(api_key=OPENAI_API_KEY)
#     Json 변화 파서 설정
output_parser = PydanticOutputParser(pydantic_object=Output)

def inference_image_text(url_list):
    prompt = prompt_image_base.format(format_instruction = output_parser.get_format_instructions())
    content = [
        {"type":"text","text": prompt},
    ]
    # 이미지를 보낼때는 이런식으로 해야한다
    for url in url_list[:5]:
        content.append({
            "type":"image_url",
            "image_url":{
                "url":url
            }
        })
    respons = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role":"system", "content":"You are a helpful assistant"},
            {"role":"user","content":content}
        ],
        max_tokens=1000,
        response_format={"type":"json_object"}
    )
    output= respons.choices[0].message.content
    output_json = json.loads(output)
    return output_json

if __name__ == '__main__':
    url_list = get_urls()
    result = inference_image_text(url_list)
    print(json.dumps(result , indent=2, ensure_ascii=False))