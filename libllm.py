#llm helper functions


## ENV FILE LOADING
# for api keys
from dotenv import load_dotenv
load_dotenv()


# LLM 
from langchain.chat_models import init_chat_model
def get_chat_model():
    model = init_chat_model("groq:llama-3.3-70b-versatile")
    return model



# clean LLM output
def clean_json_string(raw_str: str) -> str:
    """Extracts the JSON block even if the LLM adds extra conversational text."""
    start_idx = raw_str.find('{')
    end_idx = raw_str.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        cleaned = raw_str[start_idx:end_idx + 1]
        return cleaned
        
    return raw_str.strip()

