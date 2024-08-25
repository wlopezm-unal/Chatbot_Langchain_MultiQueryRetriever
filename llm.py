import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse.decorators import observe
from langfuse.openai import openai
import os

#Cargar las variables de entorno | Load environment variables
load_dotenv()
# Configurar las credenciales de Langfuse
os.environ["LANGFUSE_PUBLIC_KEY"] = "LANGFUSE_PUBLIC_KEY"
os.environ["LANGFUSE_SECRET_KEY"] = "LANGFUSE_SECRET_KEY"
class LLM:
    def __init__(self):
        self.model_name = "gemini-1.5-pro" #"mixtral-8x7b-32768"#"gpt-4o"
        self.temperature = 0
        self.max_tokens = None
        self.llm = self.init_llm()

    #@observe()
    def init_llm(self):
        
        llm=ChatGoogleGenerativeAI(
        model=self.model_name,
        temperature=self.temperature,
        max_tokens=self.max_tokens
        )
        
        return llm
    
    def response_llm(self):
        llm=ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=self.temperature,
            max_tokens=self.max_tokens)
        return llm
 


"""
https://www.kaggle.com/code/aritrase/langchaincrashcourse-multi-query-retriever-part10/notebook
https://python.langchain.com/v0.2/docs/integrations/vectorstores/faiss/
"""