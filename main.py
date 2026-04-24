# This file contains the fastapi end point setup



## JSON VALIDATION CLASSES ##
from pydantic import BaseModel, Field
from typing import Literal, List


class ChatHistory(BaseModel):
    '''
        role is used to specify whose message it is, wheather it was sent by ai or user
    '''
    role : Literal["user", "ai"]
    content : str = Field(..., description = "the actual message sent by user or agent")


class ChatRequest(BaseModel):
    '''
        setting next_ask_clarity_question help ai to ask clarity type question next
        clarity_summary is the summary content created about user intention, used to get clarity in normal chat
        history is a list containing chat history so far
    '''
    ask_clarity_question_next: bool 
    clarity_summary : str | None = Field(default = None, description = "Clarified summary about user intention") 
    history : List[ChatHistory]




## FASTAPI INSTANCE CREATION ##
from fastapi import FastAPI
app = FastAPI()



## FASTAPI ENDPOINT CREATIONS ##
@app.get("/")
async def root() -> dict:
    return { "message" : "hello world" }


#getting the chat model from llm.py
from llm import get_chat_model 
try:
    model = get_chat_model()
except:
    print("> Failed to get chat model")

@app.post("/chat")
async def chat(user_request: ChatRequest) -> str:
    '''
        this endpoint is used to accept chat request from user and is validated against ChatRequest type
        this endpoint also return reply to the frontend
    '''
    print("> Received chat request")
    print("> Invoking model")

    # TESTING
    input_message = user_request.history[0].content

    result = model.invoke(input_message)
    return result.text


