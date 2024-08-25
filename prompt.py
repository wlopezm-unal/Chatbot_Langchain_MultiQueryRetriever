from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

class Prompt:
    def __init__(self, questions:list):
        self.prompt = self.prompt_template()
        self.questions = questions
    
    def prompt_template(self):
        return PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate two 
            different versions of the given user question to retrieve relevant documents from a vector 
            database. By generating multiple perspectives on the user question, your goal is to help
            the user overcome some of the limitations of the distance-based similarity search. 
            Provide these alternative questions in JSON format. Generate a list of questions in JSON format. 
            Each question must be on a separate line, including the original question:
        [
            “What is the capital city of France?”,
            “What is the capital city of Germany?”
        ].
            Original question: {question}"""
        )
    
    def prompt_answer(self, context, question):
        
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the questions. If you don't know the answer, say that you "
            "don't know. Use seventy sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "Context: {context}"
            
        )
        
        human_prompt = (
            "Please answer the following questions based on the given context:\n"
            "{context}\n\n"
            "User's next questions: {question}"
            )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", human_prompt),
            ]
        )
        return prompt
    
    