import json
from fastapi import APIRouter, HTTPException
from libllm import clean_json_string, get_chat_model
from pydantic import BaseModel, Field
from typing import Annotated, Dict




chat_model = get_chat_model()
router = APIRouter()




EXECUTION_SYSTEM_PROMPT = """
You are Clarity AI, an expert project manager and strategic planner. Your task is to generate a comprehensive, highly actionable execution plan to achieve the user's objective.

CRITICAL RULES:
1. Thoroughly analyze the INFO COLLECTED SO FAR. 
   - Tailor the complexity and tone to the user's "profile".
   - Start exactly from their "current_state" (do not include steps they have already completed).
   - Exclusively suggest tools/methods that align with their "resources".
   - Ensure the plan strictly navigates around their "constraints".
2. The plan must be step-by-step, practical, and directly solve the "objective".
3. Format the "execution_plan" value as a clean, highly readable Markdown string (using ## headers, bullet points, or numbered lists). Ensure newlines are correctly escaped in the JSON (using \\n).
4. Provide a brief, encouraging conversational wrap-up in "reply_text".
5. Output ONLY raw JSON. No markdown formatting outside the JSON values, no code blocks (```json).

OUTPUT FORMAT:
You must output strictly valid JSON matching this exact schema:
{
  "reply_text": "Your conversational wrap-up or encouragement here",
  "execution_plan": "The full step-by-step plan formatted in Markdown"
}
"""

class ExecutionRequest(BaseModel):
    user_info: Dict[str, str] = Field(default_factory=dict, description="The fully populated context object (profile, objective, current_state, resources, constraints)")

class ExecutionResponse(BaseModel):
    reply_text: str
    execution_plan: str

@router.post("/execution")
async def generate_execution_plan(request: ExecutionRequest) -> dict:
    print("> Received execution plan request")
    user_info = request.user_info
    
    system_instruction = f"{EXECUTION_SYSTEM_PROMPT}\n\n INFO COLLECTED SO FAR:{user_info}"
    messages = [{"role": "system", "content": system_instruction}]

    print("> Invoking model for plan generation...")
    try:
        result = chat_model.invoke(messages)
        
        raw_output = clean_json_string(result.text)
        parsed_json = json.loads(raw_output)
        
        print(f"> Received execution response{parsed_json}")
        validated_response = ExecutionResponse(**parsed_json)
        return validated_response.model_dump()

    except json.JSONDecodeError as e:
        print(f"!! Failed to parse AI JSON: {result.text}")
        raise HTTPException(status_code=500, detail="AI returned invalid JSON structure.")

    except Exception as e:
        print(f"!! Failed to invoke model: {e}")
        raise HTTPException(status_code=500, detail=str(e))
