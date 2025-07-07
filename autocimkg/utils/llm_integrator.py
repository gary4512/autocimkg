from langchain_core.exceptions import OutputParserException
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import time
import openai
from typing import Union
import numpy as np
import logging
import sys
from importlib import reload

class LLMIntegrator:
    """
    A parser designed for extracting and embedding information using Langchain and OpenAI APIs.
    """
    
    def __init__(self, llm_model, embeddings_model, sleep_time: int = 5, logger = None):
        """
        Initializes the LangchainLLMIntegrator with specified API key, models, and operational parameters.
        
        :param llm_model: Chat model
        :param embeddings_model: Embeddings model
        :param sleep_time: Time to wait (in seconds) when encountering rate limits or errors
        :param logger: Logger instance used for logging operations (defaults to None)
        """

        self.model = llm_model
        self.embeddings_model = embeddings_model
        self.sleep_time = sleep_time
        self.logger = logger

        # stdout logger
        if self.logger is None:
            reload(logging)
            self.logger = logging.getLogger("autocimkg")
            self.logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s: %(message)s',
                                               datefmt="%Y-%m-%d %H:%M:%S")
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(console_handler)


    def calculate_embeddings(self, text: Union[str, list[str]]) -> np.ndarray:
        """
        Calculates embeddings for the given text using the initialized embeddings model.
        NOTE: exceptions allowed to interrupt program flow, as embeddings are indispensable!
        
        :param text: Text or list of texts to embed
        :returns: Calculated embeddings as a NumPy array
        :raises TypeError: Input text is neither a string nor a list of strings
        """

        if isinstance(text, list):
            return np.array(self.embeddings_model.embed_documents(text))
        elif isinstance(text, str):
            return np.array(self.embeddings_model.embed_query(text))
        else:
            raise TypeError("Invalid text type, please provide a string or a list of strings.")

    def extract_information(self, output_data_structure, context: str, instructions: str, retry: int = 0):
        """
        Extracts information from a given context and format it as JSON using a specified structure.
        Note: Handles rate limit and bad request errors by waiting and retrying.
        
        :param output_data_structure: The data structure definition for formatting the JSON output
        :param context: The context from which to extract information
        :param instructions: The query to provide to the language model for extracting information
        :param retry: Retry count (default is 0, maximizes at 5 and leads to fallback return {})

        :returns: The structured JSON output based on the provided data structure and extracted information (or {})
        """

        # return "empty JSON"
        if retry == 5: return {}

        # Set up a parser and inject instructions into the prompt template.
        parser = JsonOutputParser(pydantic_object=output_data_structure)

        template = f"{{query}}\n\n" \
                  + "# FORMAT INSTRUCTIONS\n" \
                 + f"{{format_instructions}}\n\n" \
                 + f"{context}"

        prompt = PromptTemplate(
            template=template,
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | self.model | parser

        try:
            return chain.invoke({"query": instructions})
        except openai.BadRequestError:
            self.logger.exception("Too much requests, we are sleeping!")
            time.sleep(self.sleep_time)
            return self.extract_information(output_data_structure, context, instructions, retry + 1)
        except openai.RateLimitError:
            self.logger.exception("Too much requests exceeding rate limit, we are sleeping!")
            time.sleep(self.sleep_time)
            return self.extract_information(output_data_structure, context, instructions, retry + 1)
        except OutputParserException as e:
            self.logger.warning("Error in parsing the instance: %s", context)
            raise e
