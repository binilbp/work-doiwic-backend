import json
from fastapi import APIRouter, HTTPException
from libllm import clean_json_string, get_chat_model
from pydantic import BaseModel, Field
from typing import Annotated 




chat_model = get_chat_model()
router = APIRouter()




PROFILE_SYSTEM_PROMPT = """
You are Clarity AI. Your primary task is to read the user's input and extract their specific profile (who they are, their role, or their skill level).

CRITICAL RULES:
1. Focus ONLY on extracting the profile (e.g., "Junior Developer", "Exhausted Parent", "Marketing Manager").
2. If the input is too obscure, vague, or lacks context to determine a profile: 
   - Set "insufficient_info" to true.
   - Set "profile" to null.
   - Use "reply_text" to ask the user a clarifying question.
3. If you successfully identify the profile:
   - Set "insufficient_info" to false.
   - Set "profile" to the extracted profile.
   - Provide a brief, friendly confirmation in "reply_text".
4. You must NOT output any conversational filler or text outside the JSON object.
5. Do NOT wrap the JSON in markdown formatting or code blocks (do not use ```json). Output raw text only.

OUTPUT FORMAT:
You must output strictly valid JSON matching this exact schema:
{
  "reply_text": "Your conversational confirmation or follow-up question here",
  "insufficient_info": bool,
  "profile": "The extracted profile, or None if insufficient info"
}
"""

class ProfileRequest(BaseModel):
    input: Annotated[str, Field(..., description="the input message sent by user")]

class ProfileResponse(BaseModel):
    reply_text: str
    insufficient_info: bool
    profile: str | None

@router.post("/profile")
async def extract_profile(request: ProfileRequest) -> dict:
    print("> Received profile extraction request")
    user_input = request.input
    
    system_instruction = f"{PROFILE_SYSTEM_PROMPT}\n\n USER INPUT: {user_input}"
    messages = [{"role": "system", "content": system_instruction}]    

    print("> Invoking model...")
    try:
        result = chat_model.invoke(messages)
        
        raw_output = clean_json_string(result.text)
        parsed_json = json.loads(raw_output)
        
        print(f"> Recieved respone: {parsed_json}")
        validated_response = ProfileResponse(**parsed_json)
        return validated_response.model_dump()

    except json.JSONDecodeError as e:
        print(f"!! Failed to parse AI JSON: {result.text}")
        raise HTTPException(status_code=500, detail="AI returned invalid JSON structure.")

    except Exception as e:
        print(f"!! Failed to invoke model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


