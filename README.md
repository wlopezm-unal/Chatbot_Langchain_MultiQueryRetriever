﻿# Chatbot using Langchain's model for chatting with your PDF's
 ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Chatbot where you can chat with your PDF. It consists of a multiretriever and multivector model. When you insert your PDF it will generate a split and a summary of your documents, where in a vectorial base Qdrant will save the complete document, the split and a summary of the document in different collections respectively. 
When the user inserts the query, the multiquery retriever will create a query adjacent to the original one, having two queries, where they will be used to search for the documentation in the Qdrant summary and split collections. Only two collections were selected for the search due to the limitation of the number of tokens to be passed to the LLM model. 
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#USE

1. pip install requirements.txt
2. Deploy Qdrant using Docker-Compose
   2.1 Run image qdrant
3. deployment streamlit : streamlit run main.py

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Deployment
1. Qdrant
   ![image](https://github.com/user-attachments/assets/1c45b660-1fa6-4a30-b867-7dfd9b38a0a0)
2. Streamlit
   ![image](https://github.com/user-attachments/assets/a515bc42-2e50-422a-bab4-18894d633c21)
3. Consult you question
   ![image](https://github.com/user-attachments/assets/7cc5de78-217b-451c-829a-49c57a00ed1e)





   
   
