import json
from fastapi import APIRouter, HTTPException
from libllm import clean_json_string, get_chat_model
from pydantic import BaseModel, Field
from typing import Annotated, Dict




chat_model = get_chat_model()
router = APIRouter()




CURRENT_STATE_SYSTEM_PROMPT = """
You are Clarity AI. Your task is to extract the user's current state only (what they have already done, or their starting point etc).
NOTE: do not ask any questions back to help the user.

CRITICAL RULES:
1. If the input is too obscure, vague, or lacks context about their current state:
   - Set "insufficient_info" to true.
   - Set "current_state" to null (strictly use JSON null).
   - Use "reply_text" to ask them to explain what their current position is.
2. If you CAN extract the current state:
   - Set "insufficient_info" to false.
   - Summarize their current state clearly dont mention their profile or objective or any other unneccessary details only mention the current state.
   - Provide a conversational acknowledgment in "reply_text".
3. Output ONLY raw JSON. No markdown formatting, no code blocks (```json).

OUTPUT FORMAT:
{
  "reply_text": "Your conversational reply ",
  "insufficient_info": bool,
  "current_state": "Summary of current state, or None if insufficient_info is true"
}
"""

class CurrentStateRequest(BaseModel):
    input: Annotated[str, Field(..., description="the input message sent by user")]
    user_info: Dict[str, str] = Field(default_factory=dict, description="Known facts about the user")

class CurrentStateResponse(BaseModel):
    reply_text: str
    insufficient_info: bool
    current_state: str | None

@router.post("/current_state")
async def extract_current_state(request: CurrentStateRequest) -> dict:
    print("> Received current_state extraction request")
    user_input = request.input
    user_info = request.user_info
    
    system_instruction = f"{CURRENT_STATE_SYSTEM_PROMPT}\n\n USER INPUT: {user_input} INFO COLLECTED SO FAR:{user_info}"
    messages = [{"role": "system", "content": system_instruction}]
    print("> Invoking model...")

    try:
        result = chat_model.invoke(messages)
        
        raw_output = clean_json_string(result.text)
        parsed_json = json.loads(raw_output)
        
        print(f"> Recieved respone: {parsed_json}")
        validated_response = CurrentStateResponse(**parsed_json)
        return validated_response.model_dump()

    except json.JSONDecodeError as e:
        print(f"!! Failed to parse AI JSON: {result.text}")
        raise HTTPException(status_code=500, detail="AI returned invalid JSON structure.")

    except Exception as e:
        print(f"!! Failed to invoke model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


