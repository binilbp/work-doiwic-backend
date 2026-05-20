
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware

from routers import role

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(role.router)















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

