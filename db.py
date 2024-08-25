#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse
import google.generativeai as genai
from langchain.schema import Document
import uuid
import logging
import os

"""
Importaciones y Configuración de Logging
Configura el módulo de logging para registrar información importante durante la ejecución del código.
Crea un logger para la clase, permitiendo registrar mensajes específicos de esta.
"""
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDB:
    """
    Inicializa una instancia de VectorDB con varios parámetros, incluyendo:
    
    Attributes
    ----------
    text : str
        Texto a procesar.
    url : str
        URL del servidor de Qdrant.
    host : str
        Host del servidor de Qdrant.
    port : int
        Puerto del servidor de Qdrant.
    collection_name : list
        Lista de nombres de las diferentes colecciones.
    type_collection : str
        Nombre específico de la colección.
    model : str
        Modelo de embeddings a utilizar para generar los embeddings.

    Methods
    -------
    check_connection_qdrant()
        Verifica si hay conexión con el servidor de Qdrant.
    create_vectordb()
        Crea las colecciones en la base de datos de vectores.
    check_colecction()
        Verifica si la colección especificada existe en la base de datos.
    create_and_store_embedding()
        Crea y almacena los embeddings en la base de datos de vectores.
    """

    def __init__(self, text, type_collection=None):
        """
        Construye todos los atributos necesarios para el objeto VectorDB.

        Parameters
        ----------
        text : str
            Texto a procesar.
        type_collection : str, optional
            Nombre específico de la colección (default es None).
        """
        
        self.text = text
        self.url = "http://localhost:6333"
        self.port = 6333
        self.collection_name = ['Documents', 'Summary', 'Splited_text']
        self.type_collection=type_collection
        self.host = "localhost"
        self.model = "models/embedding-001"

    def check_connection_qdrant(self):
        """
        Verifica si hay conexión con el servidor de Qdrant.

        Returns
        -------
        QdrantClient
            Cliente de Qdrant si la conexión es exitosa.

        Raises
        ------
        Exception
            Si hay un error al conectar con el servidor de Qdrant.
        """

        #Check if there connection with Qdrant
        try:
            client = QdrantClient(host=self.host, port=self.port, timeout=30.0)
            logger.info("succesful connection with Qdrant")
            return client
        except Exception as e:            
            raise logger.error(f"connect error with  Qdrant: {e}")

    def create_vectordb(self):
        #verificar el cliente de Qdrant
        client=self.check_connection_qdrant()
        
        #crear la coleccion
        try:
            for collection in self.collection_name:
                embedding=GoogleGenerativeAIEmbeddings(model=self.model)
                # Realizar un embedding de prueba para obtener la dimensionalidad
                test_embedding = embedding.embed_query("test")
                #obtener la dimensionadlidad de los embeddings
                embedding_dim = len(test_embedding)

                client.recreate_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
                )
                logger.info(f"Create colecction: {collection}, succesfully")
        except Exception as e:
            logger.error(f"Error creating collections: {e}")

    def check_colecction(self):

        """
        Verifica si la colección especificada existe en la base de datos.

        Si la colección no existe, la crea utilizando el método `create_vectordb`.

        Raises
        ------
        Exception
            Si hay un error al verificar la colección.
        """

        
        #Configure client Qdrant. Connection point with Qdrant server
        client = self.check_connection_qdrant()

        #Check if the collection exists
        try:
            client.get_collection(collection_name=self.type_collection)
            logger.info(f"Collection {self.type_collection} exists")

        except UnexpectedResponse as e:
            
            if e.status_code == 404:
                logger.error(f"Collection {self.type_collection} does not exist")
                
                #create collection(s) using  create_vectordb funtion
                self.create_vectordb()
            
            #If is other error type, print the error
            else:
                logger.error(f"Error checking collection: {e}")
                raise

        except Exception as e:
            logger.error(f"Error checking collection: {e}")
            raise
            
    
    def create_and_store_embedding(self):

        """
        Crea y almacena los embeddings en la base de datos de vectores.

        Este método primero verifica si la colección especificada existe,
        luego crea los embeddings y los almacena en la base de datos de vectores.

        Raises
        ------
        ValueError
            Si el tipo de `self.text` no es soportado.
        Exception
            Si hay un error al crear y almacenar los embeddings.
        """

        try:
            #First check if the collection exists
            self.check_colecction()

            #Create embedding
            embedding=GoogleGenerativeAIEmbeddings(model=self.model)
            #create metadata with ID unique        
            # Asegúrate de que self.text sea una cadena
            if isinstance(self.text, str):
                documents = [Document(page_content=self.text, metadata={"id": str(uuid.uuid4())})]
            elif isinstance(self.text, list):
                documents = [Document(page_content=chunk.page_content, metadata={"id": str(uuid.uuid4())}, page=chunk.metadata['page']) for chunk in self.text] #if isinstance(chunk, str)]


            else:
                raise ValueError(f"Elemento de lista no soportado: {type(self.text)}")
            
              # Configure client Qdrant. Connection point with Qdrant server
            self.check_connection_qdrant()
            

            #insert data in qdrant
            # Insert data in qdrant
            Qdrant.from_documents(
                documents=documents,
                embedding=embedding,  # Add this line
                collection_name=self.type_collection,
                url=self.url,
                prefer_grpc=False
            )
            logger.info(f"{self.type_collection} saved with sucessfully")            

        except Exception as e:
            logger.error(f"Error creating vector_db: {e}")
            raise


