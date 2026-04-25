# This file contains the python code to setup and create llm functions



## ENV FILE LOADING
# for api keys
from dotenv import load_dotenv
load_dotenv()


# LLM 
from langchain.chat_models import init_chat_model
def get_chat_model():
    try:
        model = init_chat_model("google_genai:gemini-2.5-flash-lite")
        return model
    except:
        print("!! Failed to load model")


