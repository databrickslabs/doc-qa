from databricks.labs.doc_qa.llm_providers import openai_provider
from databricks.labs.doc_qa.llm_utils import PromptTemplate
import openai
from databricks.labs.doc_qa.chatbot.retriever import Document, BaseRetriever
import logging
import tiktoken
import dataclasses

logger = logging.getLogger(__name__)

class LlmProvider:

    def prompt(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError()


class OpenAILlmProvider(LlmProvider):

    def __init__(self, api_key: str, model, temperature, **kwargs):
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        openai.api_key = api_key
    
    def prompt(self, prompt: str, **kwargs) -> str:
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        response_message = openai_provider.request_openai(messages=messages, functions=[], model=self._model, temperature=self._temperature)
        return response_message['content']

class ChatResponse:
    content: str
    relevant_documents: list[Document]

    def __init__(self, content: str, relevant_documents: list[Document]):
        self.content = content
        self.relevant_documents = relevant_documents


class BaseChatBot:
    def __init__(self, llm_provider: str, retriever: BaseRetriever, 
                 whole_prompt_template: PromptTemplate, 
                 document_prompt_tempate: PromptTemplate, 
                 max_num_tokens_for_context: int = 3500, **kwargs):
        self._llm_provider = llm_provider
        self._retriever = retriever
        self._whole_prompt_template = whole_prompt_template
        self._document_prompt_tempate = document_prompt_tempate
        self._max_num_tokens_for_context = max_num_tokens_for_context
        self._enc = tiktoken.encoding_for_model(self._llm_provider.model)

    def chat(self, prompt: str, top_k=1, **kwargs) -> ChatResponse:
        """
        Chat with the chatbot.
        """
        relevant_documents = self._retriever.find_similar_docs(query=prompt, top_k=top_k)
        # First, format the prompt for each document
        document_str = ""
        total_num_tokens = 0
        for index, document in enumerate(relevant_documents):
            # use all attributes of document, except for created_at or vector, as the format parameter
            doc_formated_prompt = self._document_prompt_tempate.format(**{k: v for k, v in document.__dict__.items() if k not in ['created_at', 'vector']})
            num_tokens = len(self._enc.encode(doc_formated_prompt))
            if total_num_tokens + num_tokens > self._max_num_tokens_for_context:
                logger.warning(f"Exceeding max number of tokens for context: {self._max_num_tokens_for_context}, existing on {index}th document out of {len(relevant_documents)} documents")
                break
            total_num_tokens += num_tokens
            document_str += doc_formated_prompt + "\n"
        logger.debug(f"Document string: {document_str}")
        # Then, format the whole prompt
        whole_prompt = self._whole_prompt_template.format(context=document_str, prompt=prompt)
        logger.debug(f"Whole prompt: {whole_prompt}")
        response = self._llm_provider.prompt(whole_prompt)
        return ChatResponse(content=response, relevant_documents=relevant_documents)
    

