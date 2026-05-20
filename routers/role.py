import json
from fastapi import APIRouter, HTTPException
from libllm import clean_json_string, get_chat_model
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Literal, List, Dict




chat_model = get_chat_model()
router = APIRouter()




ROLE_SYSTEM_PROMPT = """
You are Clarity AI. Your primary task is to read the user's input and extract the specific role or profession they are describing.

CRITICAL RULES:
    The extracted role MUST be exactly ONE WORD (e.g., "Developer", "Teacher", "Manager").
    If the input is too obscure, vague, or lacks context to determine a single-word role, set "insufficient_info" to true. In this case, set "role" to null, and use "reply_text" to ask the user a clarifying question.
    If you successfully identify the role, set "insufficient_info" to false and provide a brief, friendly confirmation in "reply_text".
    You must NOT output any conversational filler or text outside the JSON object.
    Do NOT wrap the JSON in markdown formatting or code blocks (do not use ```json). Output raw text only.

OUTPUT FORMAT:
You must output strictly valid JSON matching this exact schema:
{
"reply_text": "Your conversational confirmation or follow-up question here",
"insufficient_info": true or false,
"role": "The one-word role, or "" if insufficient info"
}
"""




class RoleRequest(BaseModel):
    input: Annotated[
            str, 
            Field(..., description = "the input message sent by user")
        ]

class RoleResponse(BaseModel):
    reply_text: str
    insufficient_info: bool
    role: str | None





@router.post("/role")
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
        
        print(f"> Recieved respone: {parsed_json}")
        validated_response = RoleResponse(**parsed_json)
        return validated_response.model_dump()

    except json.JSONDecodeError as e:
        print(f"!! Failed to parse AI JSON: {result.text}")
        raise HTTPException(status_code=500, detail="AI returned invalid JSON structure.")

    except Exception as e:
        print(f"!! Failed to invoke model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


