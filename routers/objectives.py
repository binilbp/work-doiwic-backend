import json
from fastapi import APIRouter, HTTPException
from libllm import clean_json_string, get_chat_model
from pydantic import BaseModel, Field
from typing import Annotated, Dict




chat_model = get_chat_model()
router = APIRouter()




OBJECTIVE_SYSTEM_PROMPT = """
You are Clarity AI. Your primary task is to read the user's input and extract their main objective (their ultimate goal or the problem they want to solve).

CRITICAL RULES:
1. Focus ONLY on extracting the objective. No need to get too much info about context
2. If the input is too much obscure, vague, or lacks any context: 
   - Set "insufficient_info" to true.
   - Set "objectives" to null (strictly use JSON null).
   - Explain what additional information you need in "reply_text".
3. If you CAN extract the objective:
   - Set "insufficient_info" to false.
   - Summarize the user's objective clearly.
   - Provide a conversational acknowledgment in "reply_text".
4. Output ONLY raw JSON. No markdown formatting, no code blocks (
```json).

OUTPUT FORMAT:
{
  "reply_text": "Your conversational confirmation or follow-up question here",
  "insufficient_info": bool,
  "objectives": "A brief summary of the user's objective, or None if insufficient_info is true"
}
"""

class ObjectiveRequest(BaseModel):
    input: Annotated[str, Field(..., description="the input message sent by user")]
    user_info: Dict[str, str] = Field(default_factory=dict, description="Known facts about the user")

class ObjectiveResponse(BaseModel):
    reply_text: str
    insufficient_info: bool
    objectives: str | None

@router.post("/objectives")
async def extract_objective(request: ObjectiveRequest) -> dict:
    print("> Received objective extraction request")
    user_input = request.input
    user_info = request.user_info
    
    system_instruction = f"{OBJECTIVE_SYSTEM_PROMPT}\n\n USER INPUT: {user_input} INFO COLLECTED SO FAR:{user_info}"
    messages = [{"role": "system", "content": system_instruction}]
    print("> Invoking model...")

    try:
        result = chat_model.invoke(messages)
        
        raw_output = clean_json_string(result.text)
        parsed_json = json.loads(raw_output)
        
        print(f"> Recieved respone: {parsed_json}")
        validated_response = ObjectiveResponse(**parsed_json)
        return validated_response.model_dump()

    except json.JSONDecodeError as e:
        print(f"!! Failed to parse AI JSON: {result.text}")
        raise HTTPException(status_code=500, detail="AI returned invalid JSON structure.")

    except Exception as e:
        print(f"!! Failed to invoke model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


