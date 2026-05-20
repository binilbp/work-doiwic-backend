
from llm import get_chat_model
chat_model = get_chat_model()



ROLE_SYSTEM_PROMPT = """
You are clarity AI, your task is to read user input and explain his role he is trying to explain in one word

IMPORTANT: you must not reply back with normal unneccesary talk, only give back the output format, if there is no enough information,
set insufficient_info to true, also state what you want in the reply_text 

OUTPUT FORMAT:
You must output strictly valid JSON matching this schema, with no markdown formatting or extra text:
{
  "reply_text": "Your conversational response here",
  "insufficient_info": bool,
  "role" : "you summary of the role",
}

"""



from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Literal, List, Dict


class RoleRequest(BaseModel):
    input: Annotated[str, Field(..., description = "the input message sent by user")]

class RoleResponse(BaseModel):
    reply_text: str
    insufficient_info: bool
    role: str




def clean_json_string(raw_str: str) -> str:
    """Extracts the JSON block even if the LLM adds extra conversational text."""
    start_idx = raw_str.find('{')
    end_idx = raw_str.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        cleaned = raw_str[start_idx:end_idx + 1]
        return cleaned
        
    return raw_str.strip()




from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



import json

@app.post("/role")
async def summarize_role(request: RoleRequest) -> dict:
    print("> Received role summary request")
    user_input = request.input
    
    system_instruction = f"{ROLE_SYSTEM_PROMPT}\n\n USER INPUT: {user_input}"
    
    messages = [{"role": "system", "content": system_instruction}]

    print("> Invoking model...")
    try:
        result = chat_model.invoke(messages)
        
        raw_output = clean_json_string(result.text)
        parsed_json = json.loads(raw_output)
        
        validated_response = RoleResponse(**parsed_json)
        
        print(f"> Sending respone: {validated_response}")
        return validated_response.model_dump()

    except json.JSONDecodeError as e:
        print(f"!! Failed to parse AI JSON: {result.text}")
        raise HTTPException(status_code=500, detail="AI returned invalid JSON structure.")

    except Exception as e:
        print(f"!! Failed to invoke model: {e}")
        raise HTTPException(status_code=500, detail=str(e))













# class ChatHistory(TypedDict): 
#
#     role : Literal["user", "assistant", "system"]
#     content : Annotated[str, Field(..., description = "the actual message sent by user or agent")]
#
# class ChatRequest(BaseModel):
#     history: List[ChatHistory]
#     current_context: Dict[str, str] = Field(default_factory=dict, description="Current known facts about the user and user request")
#
# class PlanRequest(BaseModel):
#     context: Dict[str, str] = Field(default_factory=dict, description="Known facts about the user and user request")
#
# class AIResponse(BaseModel):
#     reply_text: str
#     extracted_context: Dict[str, str]
#     suggestion_bubbles: List[str]
#
# @app.post("/execution_plan")
# async def query_chat(request: ChatRequest) -> dict:
#     print("> Received plan request")
#     messages = request.history
#
#     context_str = json.dumps(request.current_context)
#     system_instruction = f"{QUERY_SYSTEM_PROMPT}\n\nCONTEXT GATHERED UNTIL NOW: {context_str}"
#
#     messages.insert(0, {"role": "system", "content": system_instruction})
#
#     print("> Invoking model...")
#     try:
#         result = chat_model.invoke(messages)
#
#         raw_output = clean_json_string(result.text)
#         parsed_json = json.loads(raw_output)
#
#         validated_response = AIResponse(**parsed_json)
#
#         # print(f"> Sending structured response plan {}.")
#
#         return validated_response.model_dump()
#
#     except json.JSONDecodeError as e:
#         print(f"!! Failed to parse AI JSON: {result.text}")
#         raise HTTPException(status_code=500, detail="AI returned invalid JSON structure.")
#
#     except Exception as e:
#         print(f"!! Failed to invoke model: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
#

