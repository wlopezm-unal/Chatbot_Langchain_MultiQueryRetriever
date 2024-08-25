from langchain.schema.runnable import RunnableParallel
import google.generativeai as gemini_client
from langchain.schema import Document
from db import VectorDB
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv
import os

#Cargar las variables de entorno | Load environment variables
load_dotenv()

#Cargar el api_key de google gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


class ConsultDB( VectorDB):
    """
    A class used to consult a database of documents using vector embeddings.

    Attributes
    ----------
    questions : list of str
        A list of questions to query the database.
    model : str
        The model used for generating embeddings.

    Methods
    -------
    _search_collection(collection_name: str, question: str) -> List[Document]
        Searches for documents in a specified collection related to a given question.
    query_parallel(input) -> Dict[str, List[Document]]
        Searches for documents in parallel across multiple collections related to the input question.
    get_all_document() -> List[Dict[str, List[Document]]]
        Retrieves all documents related to the list of questions.
    """

    def __init__(self, questions:list[str]):
        """
        Constructs all the necessary attributes for the ConsultDB object.

        Parameters
        ----------
        questions : list of str
            A list of questions to query the database.
        """

        self.questions = questions
        self.model="models/embedding-001" # "sentence-transformers/all-MiniLM-L6-v2" #"models/embedding-001"            
        super().__init__(text=None)    

    def _search_collection(self, collection_name: str, question:str) -> List[Document]:
        """
        Searches for documents in a specified collection related to a given question.

        Parameters
        ----------
        collection_name : str
            The name of the collection to search in.
        question : str
            The question to query the collection with.

        Returns
        -------
        List[Document]
            A list of documents related to the question.
        """

        #load Qdrant client
        client = self.check_connection_qdrant()

        #Configure the api_key of google gemini
        gemini_client.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        #Searching for document in Qdrant
        results=client.search(
            collection_name=collection_name,
            query_vector=gemini_client.embed_content(
                model=self.model,
                content=question,
                task_type="retrieval_query",  #this is it can recover this information
                )["embedding"],
        )
        return [Document(page_content=result.payload['page_content'], metadata={"score": result.score, "collection": collection_name}) 
                for result in results]
    
    def query_parallel(self, input) -> Dict[str, List[Document]]:
        """
        Searches for documents in parallel across multiple collections related to the input question.

        Parameters
        ----------
        input : str
            The input question to query the collections with.

        Returns
        -------
        Dict[str, List[Document]]
            A dictionary with collection names as keys and lists of related documents as values.
        """

        #Search parallel in the collections the documents related to the question's user 
        parallel_search = RunnableParallel(
            #original=lambda x: self._search_collection('Documents',  question=x),
            summaries=lambda x: self._search_collection('Summary',  question=x),
            splits=lambda x: self._search_collection('Splited_text',  question=x)
        )
        
        return parallel_search.invoke(input)

    def get_all_document(self):
        """
        Retrieves all documents related to the list of questions.

        Returns
        -------
        List[Dict[str, List[Document]]]
            A list of dictionaries with collection names as keys and lists of related documents as values.
        """
        
        #Create a list where it gonna store all the documents
        list_documents=[]

        #interate over the list of questions derivated of the user's question, so get documents of the Qdrant database
        for question in self.questions:
            #call the method query_parallel
            documents=self.query_parallel(input=question)
            list_documents.append(documents)
        return list_documents