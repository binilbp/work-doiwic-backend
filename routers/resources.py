import json
from fastapi import APIRouter, HTTPException
from libllm import clean_json_string, get_chat_model
from pydantic import BaseModel, Field
from typing import Annotated, Dict




chat_model = get_chat_model()
router = APIRouter()




RESOURCES_SYSTEM_PROMPT = """
You are Clarity AI. Your task is to extract the user's available resources (user tools, budget, software, assets, team etc).
NOTE: user tools here doesnt refer to AI tool calling, its just tools physically usable by the user

CRITICAL RULES:
1. If the input is too obscure, vague, or lacks context about resources:
   - Set "insufficient_info" to true.
   - Set "resources" to null (strictly use JSON null).
   - Use "reply_text" to ask what tools, budget, or help they have.
2. If you CAN extract resources:
   - Set "insufficient_info" to false.
   - Summarize the available resources clearly.
   - Provide a conversational acknowledgment in "reply_text".
3. Output ONLY raw JSON. No markdown formatting, no code blocks (
```json).

OUTPUT FORMAT:
{
  "reply_text": "Your conversational confirmation or follow-up question here",
  "insufficient_info": bool,
  "resources": "Summary of resources, or null if insufficient_info is true"
}
"""

class ResourcesRequest(BaseModel):
    input: Annotated[str, Field(..., description="the input message sent by user")]
    user_info: Dict[str, str] = Field(default_factory=dict, description="Known facts about the user")

class ResourcesResponse(BaseModel):
    reply_text: str
    insufficient_info: bool
    resources: str | None

@router.post("/resources")
async def extract_resources(request: ResourcesRequest) -> dict:
    print("> Received resources extraction request")
    user_input = request.input
    user_info = request.user_info
    
    system_instruction = f"{RESOURCES_SYSTEM_PROMPT}\n\n USER INPUT: {user_input} INFO COLLECTED SO FAR:{user_info}"
    messages = [{"role": "system", "content": system_instruction}]
    print("> Invoking model...")

    try:
        result = chat_model.invoke(messages)
        
        raw_output = clean_json_string(result.text)
        parsed_json = json.loads(raw_output)
        
        print(f"> Recieved respone: {parsed_json}")
        validated_response = ResourcesResponse(**parsed_json)
        return validated_response.model_dump()

    except json.JSONDecodeError as e:
        print(f"!! Failed to parse AI JSON: {result.text}")
        raise HTTPException(status_code=500, detail="AI returned invalid JSON structure.")

    except Exception as e:
        print(f"!! Failed to invoke model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


