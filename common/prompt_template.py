prompt_text_base="""
다음 내용을 읽고 FAQ를 만들어 주세요
FAQ는 20개를 생성해주세요
{format_instruction}
...
{info_text_data}
"""
prompt_image_base="""
다음 내용을 읽고 FAQ를 만들어 주세요
FAQ는 20개를 생성해주세요
{format_instruction}
...
"""
prompt_pdf_base="""
다음 내용을 읽고 FAQ를 만들어 주세요
FAQ는 50개를 생성해주세요
{format_instruction}
...
{info_detail}
"""

prompt_qa = """
Answer the question based only on the following context:
{context}
Question: {question}

Answer in the following language: korean
"""
