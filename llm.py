# This file contains the python code to setup and create llm functions



## ENV FILE LOADING
# for api keys
from dotenv import load_dotenv
load_dotenv()


# LLM 
from langchain.chat_models import init_chat_model
def get_chat_model():
    model = init_chat_model("groq:llama-3.3-70b-versatile")
    return model


