from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.globals import set_verbose, set_debug
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from ingest_data import IngestData
from consult_db import ConsultDB
from retriever import Retriever_QA
from llm import LLM
from prompt import Prompt
import google.generativeai as genai
import logging
import os

import phoenix as px
from phoenix.trace.langchain import LangChainInstrumentor

#Cargar las variables de entorno | Load environment variables
load_dotenv()

#Cargar el api_key de google gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
genai.configure(api_key=os.getenv("GROQ_API_KEY"))



class Chatbot(Retriever_QA, ConsultDB, LLM, Prompt):
    """
    A class used to create a chatbot that processes a user's question, generates related questions, retrieves documents from a database, and provides an answer.

    Attributes
    ----------
    question : str
        The user's question to be processed.
    retriever : Retriever_QA
        An instance of the Retriever_QA class to generate related questions.
    db_consultant : ConsultDB
        An instance of the ConsultDB class to retrieve documents from the database.
    text_splitter : CharacterTextSplitter
        An instance of the CharacterTextSplitter class to split text into chunks.

    Methods
    -------
    input() -> dict
        Processes the user's question through a pipeline to generate related questions, retrieve documents, and provide an answer.
    """
    
    def __init__(self, question):
        """
        Constructs all the necessary attributes for the Chatbot object.

        Parameters
        ----------
        question : str
            The user's question to be processed.
        """
        self.question = question
        self.retriever = Retriever_QA(question)
        self.db_consultant = ConsultDB([question])
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


    def input(self,):
        """
        Processes the user's question through a pipeline to generate related questions, retrieve documents, and provide an answer.

        Returns
        -------
        dict
            A dictionary containing the context and the answer to the user's question.
        """
        #get information from the pipeline
        set_verbose(True)
                #activate mode debug, use to it can identy problems
        set_debug(True)

        ##Permite ver información detallada sobre el funcionamiento del recuperador multiquery
        #Ver mensajes informativos para identificar problemas
        #ayuda a optimizar el rendimiento del Rag Avanzado
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        #1.Get different questions using as templete the user's question 
        retriever_qa = Retriever_QA(question=self.question)
        questions_generated = retriever_qa.generate_questions()

        #2.Get the documents in the database
        retrieve=ConsultDB(questions=questions_generated)
        documents = retrieve.get_all_document()
        # Split the combined documents into chunks        


        #3.load the llm model to Groq
        llm=LLM().init_llm()
        
        logger.info("Starting the pipeline to get the answer of the questions of the user")
        #4.Run the process chain to can get the answer the user's question
        final_rag_chain=(
            RunnablePassthrough(lambda x: {"context": documents, "question": x}) #get object runnable to be able to join the runnableSecuence of the pipeline final rag_chain
            |Prompt(questions=None).prompt_answer(context=documents, question=self.question) #call prompt QA using the questions generated
            |llm #load the llm model to Groq
            |StrOutputParser())
        
        # Create a dictionary with context and question
        input_dict = {
            "context": documents,
            "question": self.question
    }
        logger.info("Finishing the pipeline to get the answer of the questions of the user") 
        return final_rag_chain.invoke(input_dict)  

if 'phoenix_session' not in st.session_state:
    st.session_state.phoenix_session = px.launch_app()
    LangChainInstrumentor().instrument()           

def main():
    """
    The main function to run the Streamlit app for the chatbot.
    """
    st.set_page_config(page_title="Chatbot de preguntas y respuestas con tus PDFs", page_icon=":robot:")
    st.title("Chatbot") #App title
    st.header("Chatbot using Langchain's model for chatting with your PDF's ")

    #Usar la sesión existente
    session = st.session_state.phoenix_session

    # Inicializar la sesión de Phoenix
    """
    The model uses your original question to create two derived questions to retrieve information from the database to provide you with a more complete answer. 
    """
    user_question = st.text_input("Enter your question: ") #Space where the user can enter the question

    if st.button("Send"):  # Agregar un botón para enviar la pregunta
        if user_question:  # Verificar si se ha ingresado una pregunta
            chatbot=Chatbot(question=user_question)
            # Obtener la respuesta a la pregunta
            response=chatbot.input() # Obtener la respuesta del chatbot
            st.write("Answer: ", response)  # Mostrar la respuesta en la interfaz
        else:
            st.warning("Por favor ingresa una pregunta antes de enviar")  # Mostrar un mensaje de advertencia si no se ha ingresado una pregunta
    
    with st.sidebar:
        st.write("Menu")
        pdfs = st.file_uploader("Load your Pdf files", type="pdf", accept_multiple_files=True)

        if st.button("Submit"):
            with st.spinner("Cargando PDFs..."):
                ingest_data=IngestData(pdf_paths=pdfs)#here it will start  the process of ingest data the different Qdrant collections
                ingest_data.load_data_to_db()
                st.success("Carga de PDFs exitosa")
                
if __name__ == "__main__":
    main()
    
