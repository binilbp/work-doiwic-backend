import json
from fastapi import APIRouter, HTTPException
from libllm import clean_json_string, get_chat_model
from pydantic import BaseModel, Field
from typing import Annotated, Dict




chat_model = get_chat_model()
router = APIRouter()




# CONTEXT_SYSTEM_PROMPT = """
# You are Clarity AI. Your primary task is to read the user's input and understand the context of the user.
#
# CRITICAL RULES:
#     the goal, constraints, resources,friction would be given later. your task is just to understand the context.
#     If the input is too obscure, vague, or lacks context then set "insufficient_info" to true, "context" to none and explain what more u need in "reply_text" .If you are able to extract the context then summarize the context and set it as the value for the key "context"
#
#     You must NOT output any conversational filler or text outside the JSON object.
#     Do NOT wrap the JSON in markdown formatting or code blocks (do not use ```json). Output raw text only.
#
# OUTPUT FORMAT:
# You must output strictly valid JSON matching this exact schema:
# {
# "reply_text": "Your conversational confirmation or follow-up question here",
# "insufficient_info": true or false,
# "context": "The one-word role, or "" if insufficient info"
# }
# """

CONTEXT_SYSTEM_PROMPT = """
You are Clarity AI. Your primary task is to read the user's input and extract their situational context.

CRITICAL RULES:
1. Focus ONLY on simple situation. The goal, constraints, resources, and friction will be handled later.
2. If the input is too obscure, vague, or lacks context: 
   - Set "insufficient_info" to true
   - Set "context" to null (strictly use JSON null, not a string)
   - Explain what additional information you need from the user in "reply_text".
3. If you CAN extract the context:
   - Summarize the user's situational context clearly and concisely.
   - Set "insufficient_info" to false.
   - Provide a conversational acknowledgment in "reply_text".
4. Output ONLY raw JSON. No markdown formatting, no conversational filler, no code blocks (```json).

OUTPUT FORMAT:
{
  "reply_text": "Your conversational confirmation or follow-up question here",
  "insufficient_info": true,
  "context": "A brief summary of the user's context, or null if insufficient_info is true"
}
"""




class ContextRequest(BaseModel):
    input: Annotated[
            str, 
            Field(..., description = "the input message sent by user")
        ]
    user_info: Dict[str, str] = Field(default_factory=dict, description="Known facts about the user and user request")

class ContextResponse(BaseModel):
    reply_text: str
    insufficient_info: bool
    context: str | None





@router.post("/context")
async def summarize_role(request: ContextRequest) -> dict:
    print("> Received context summary request")
    user_input = request.input
    user_info = request.user_info
    
    system_instruction = f"{CONTEXT_SYSTEM_PROMPT}\n\n USER INPUT: {user_input} INFO COLLECTED SO FAR:{user_info}"
    
    messages = [{"role": "system", "content": system_instruction}]

    print("> Invoking model...")
    try:
        result = chat_model.invoke(messages)
        
        raw_output = clean_json_string(result.text)
        parsed_json = json.loads(raw_output)
        
        print(f"> Recieved respone: {parsed_json}")
        validated_response = ContextResponse(**parsed_json)
        return validated_response.model_dump()

    except json.JSONDecodeError as e:
        print(f"!! Failed to parse AI JSON: {result.text}")
        raise HTTPException(status_code=500, detail="AI returned invalid JSON structure.")

    except Exception as e:
        print(f"!! Failed to invoke model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


