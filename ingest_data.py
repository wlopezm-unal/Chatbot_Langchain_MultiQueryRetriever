from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from llm import LLM
from db import VectorDB
import tempfile
import os

class IngestData(VectorDB, LLM):
    """
    A class used to ingest data from PDF files, summarize the content, split the text into chunks, 
    and load the data into a database.

    Attributes
    ----------
    pdf_paths : list
        A list of paths to the PDF files to be ingested.
    text : str, optional
        Text content to be processed (default is None).
    chunks_size : int
        The size of the chunks to split the text into (default is 10000).
    overlap_size : int
        The size of the overlap between chunks (default is 900).

    Methods
    -------
    summary(pdf_content)
        Summarizes the content of the PDF files.
    splittext(pdf_content)
        Splits the text content of the PDF files into chunks.
    load_data_to_db()
        Loads the data from the PDF files into the database.
    """

    def __init__(self, pdf_paths, text=None):
        
        self.pdf_paths = pdf_paths
        self.text = text
        self.chunks_size=500
        self.overlap_size=50
        
        super().__init__(text)

    def summary(self, pdf_content):

        """
        Summarizes the content of the PDF files.

        Parameters
        ----------
        pdf_content : list
            A list of PDF content to be summarized.

        Returns
        -------
        list
            A list of summarized content.
        """
        
        documents = [Document(page_content=data.page_content) for data in pdf_content]
                   
        
        llm=LLM().init_llm()
        
        #ChatOpenAI(model_name="gpt-4o", max_retries=0)
        chain= load_summarize_chain(
                  llm, chain_type="stuff", 
                  verbose=False
        )             
        
        summaries = chain.invoke( documents)
        return summaries

    def splittext(self, pdf_content):

        """
        Splits the text content of the PDF files into chunks.

        Parameters
        ----------
        pdf_content : list
            A list of PDF content to be split into chunks.

        Returns
        -------
        list
            A list of text chunks.
        """
       
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunks_size,
            chunk_overlap=self.overlap_size)
        chunks = splitter.split_documents(pdf_content)
        return  chunks
    
    def load_data_to_db(self):
       
       """
        Loads the data from the PDF files into the database.

        This method processes each document, creates a temporary file to store the content,
        loads the PDF, summarizes the content, splits the text, and stores the data in the database.
        """
        
        #Verify if exist pdf_paths
       if self.pdf_paths is not None:

        # Processes each document
        for document in self.pdf_paths:

            #creating a temporaly file to stora the content of the load file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(document.read())  # Escribimos el contenido del archivo subido
                doc = temp_file.name      
                
            try:
                #Load the pdf
                pdf_loader = PyPDFLoader(doc)
                #Read the pdf
                pdf_read=pdf_loader.load()
                
                #Get the summary and store it
                summary_result = self.summary(pdf_read)
                VectorDB(text=summary_result["input_documents"][0].page_content, type_collection="Summary").create_and_store_embedding()
                
                #Split text and store it
                split_result = self.splittext(pdf_read)
                VectorDB(text=split_result, type_collection="Splited_text").create_and_store_embedding()            
                
                #Document full store it
                VectorDB(text=pdf_read, type_collection="Documents").create_and_store_embedding()

            finally:
            # Asegurarse de eliminar el archivo temporal
                os.unlink(doc)

