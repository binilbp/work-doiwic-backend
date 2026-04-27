# This file contains the fastapi end point setup




# getting the chat model from llm.py
from llm import get_chat_model, get_summary_model 
summary_model = get_summary_model()
chat_model = get_chat_model()




CLARITY_SYSTEM_PROMPT = '''
        You are a polite AI assistant focused on problem-solving. 
        Your sole task is to ask clarifying questions to fully understand the user's intent and gather missing context before attempting to answer their original query.
        Ask only maximum of two questions
        Do not answer their original query 
    '''

SUMMARY_SYSTEM_PROMPT = '''
        Analyze the chat history and extract the user's core requirements.
        Do not ask questions only output strictly in this format:
        Goal: [One sentence stating exactly what the user is trying to achieve]
        Specific Requirements: [A short, comma-separated list of any features, filters, or details they asked for]
    '''


from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Literal, List


#TypedDict is used to maintain dict type
#ChatHistory as a dict can easily be passed to model.invoke() 
class ChatHistory(TypedDict): 
    role : Literal["user", "assistant"]#using OpenAi messages scheme
    content : Annotated[str, Field(..., description = "the actual message sent by user or agent")]


class ChatRequest(BaseModel):
    clarity_chat: bool = Field(default = False, description = "Set True if the current chat is clarity chat")
    clarity_summary : str | None = Field(default = None, description = "Clarified summary about user intention") 
    history : List[ChatHistory]




#creating fastapi instance
from fastapi import FastAPI
app = FastAPI()




@app.post("/chat")
async def chat(user_request: ChatRequest) -> str:

    print("> Received chat request")
    messages = user_request.history

    if user_request.clarity_chat:
        print("> Identified as clarity chat request")
        # add clarity system prompt to the messages history
        messages.insert(0, {"role" : "system", "content" : f"{CLARITY_SYSTEM_PROMPT}"})

    print("> Invoking model for clarity")
    try:
        result = chat_model.invoke(messages)
        print(f"> Sending chat reply : {result.text} ")
        return result.text
    except Exception as e:
        print(f"!! Failed to invoke model\n{e}")




@app.post("/chat-summary")
async def summary_chat(history: List[ChatHistory]) -> str:
    print("> Recieved chat summary request")
    messages = history
    # add system prompt to the messages history
    messages.insert(0, {"role" : "system", "content" : f"{SUMMARY_SYSTEM_PROMPT}"})

    print("> Invoking model for summary")
    try:
        result = summary_model.invoke(messages)
        print(f"> Sending summary : {result.text} ")
        return result.text
    except Exception as e:
        print(f"!! Failed to invoke model\n{e}")
   


