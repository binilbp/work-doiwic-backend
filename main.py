# This file contains the fastapi end point setup




# getting the chat model from llm.py
from llm import get_chat_model
chat_model = get_chat_model()




SYSTEM_PROMPT = """
You are a highly analytical and helpful AI assistant. Your goal is to solve the user's problem, but you MUST fully understand their context first.

RULES:
1. Review the chat history and the current known context.
2. If you DO NOT have enough information to give a perfect, tailored answer, ask ONE clarifying question.
3. If you ask a clarifying question, provide exactly 3 likely answers as `suggestion_bubbles`.
4. If you DO have enough information, give your final answer and leave `suggestion_bubbles` as an empty list [].
5. Extract any confirmed facts about the user's situation (e.g., Role, Topic, Budget, Error Code) into `extracted_context` as key-value pairs. 
6. Always carry over existing context unless the user explicitly changes it.

OUTPUT FORMAT:
You must output strictly valid JSON matching this schema, with no markdown formatting or extra text:
{
  "reply_text": "Your conversational response here",
  "extracted_context": {"Key": "Value"},
  "suggestion_bubbles": ["Option 1", "Option 2", "Option 3"]
}
You must output STRICTLY valid JSON. Do not include ANY conversational text before or after the JSON block.
"""


from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Literal, List, Dict


class ChatHistory(TypedDict): 
    role : Literal["user", "assistant", "system"]#using OpenAi messages scheme
    content : Annotated[str, Field(..., description = "the actual message sent by user or agent")]

class ChatRequest(BaseModel):
    history: List[ChatHistory]
    current_context: Dict[str, str] = Field(default_factory=dict, description="Current known facts about the user")

class AIResponse(BaseModel):
    reply_text: str
    extracted_context: Dict[str, str]
    suggestion_bubbles: List[str]




def clean_json_string(raw_str: str) -> str:
    """Extracts the JSON block even if the LLM adds extra conversational text."""
    start_idx = raw_str.find('{')
    end_idx = raw_str.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        cleaned = raw_str[start_idx:end_idx + 1]
        return cleaned
        
    return raw_str.strip()


#creating fastapi instance
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

@app.post("/chat")
async def chat(request: ChatRequest) -> dict:
    print("> Received chat request")
    messages = request.history
    
    context_str = json.dumps(request.current_context)
    system_instruction = f"{SYSTEM_PROMPT}\n\nCURRENT CONTEXT: {context_str}"
    
    messages.insert(0, {"role": "system", "content": system_instruction})

    print("> Invoking model...")
    try:
        result = chat_model.invoke(messages)
        
        raw_output = clean_json_string(result.text)
        parsed_json = json.loads(raw_output)
        
        validated_response = AIResponse(**parsed_json)
        
        print(f"> Sending structured response with {len(validated_response.suggestion_bubbles)} bubbles.")
        
        return validated_response.model_dump()

    except json.JSONDecodeError as e:
        print(f"!! Failed to parse AI JSON: {result.text}")
        raise HTTPException(status_code=500, detail="AI returned invalid JSON structure.")

    except Exception as e:
        print(f"!! Failed to invoke model: {e}")
        raise HTTPException(status_code=500, detail=str(e))



