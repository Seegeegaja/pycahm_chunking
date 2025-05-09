from fastapi import FastAPI, Body
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from service.search_and_answer import generate_answer,search

app = FastAPI()

# CORS 설정
origins = {
    "http://localhost:3000"
}
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials=True,
    allow_methods={"*"},
    allow_headers={"*"}
)

class RequestBody(BaseModel):
    question: str
@app.post("/chatbot")
async def answer(body : RequestBody = Body()):
    qa = search(body.question)
    return generate_answer(qa, body.question)