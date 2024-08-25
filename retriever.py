from langchain_core.output_parsers import CommaSeparatedListOutputParser
from llm import LLM
from prompt import Prompt

class Retriever_QA(LLM, Prompt):
    """
    A class used to generate questions using a language model and a prompt template.

    Attributes
    ----------
    question : str
        The question to be processed.

    Methods
    -------
    generate_questions() -> list
        Generates a list of questions based on the input question using a language model and a prompt template.
    """

    def __init__(self, question):
        """
        Constructs all the necessary attributes for the Retriever_QA object.

        Parameters
        ----------
        question : str
            The question to be processed.
        """
 
        self.question=question
        super().__init__()
        
    def generate_questions(self):
        """
        Generates a list of questions based on the input question using a language model and a prompt template.

        This method initializes the language model, sets up the prompt template, and processes the input question
        to generate a list of related questions.

        Returns
        -------
        list
            A list of generated questions.
        """
         
        llm = self.init_llm()
        QUERY_PROMPT = self.prompt_template()
        output_parser = CommaSeparatedListOutputParser() #LineListOutputParser()
        llm_chain = QUERY_PROMPT|llm| output_parser #StrOutputParser()  #
        result = llm_chain.invoke(self.question)
        return result
