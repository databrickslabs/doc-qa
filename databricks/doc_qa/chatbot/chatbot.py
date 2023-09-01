from databricks.doc_qa.llm_providers import openai_provider
from databricks.doc_qa.llm_utils import PromptTemplate
import openai
from databricks.doc_qa.chatbot.retriever import Document, BaseRetriever
import logging
import tiktoken

logger = logging.getLogger(__name__)

class LlmProvider:

    def prompt(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError()


class OpenAILlmProvider(LlmProvider):

    def __init__(self, api_key: str, model, temperature, **kwargs):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        openai.api_key = api_key
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def prompt(self, prompt: str, **kwargs) -> str:
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        response_message = openai_provider.request_openai(messages=messages, functions=[], model=self.model, temperature=self.temperature)
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
        self.llm_provider = llm_provider
        self.retriever = retriever
        self.whole_prompt_template = whole_prompt_template
        self.document_prompt_tempate = document_prompt_tempate
        self.max_num_tokens_for_context = max_num_tokens_for_context
        self.enc = tiktoken.encoding_for_model(self.llm_provider.model)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def chat(self, prompt: str, top_k=1, **kwargs) -> ChatResponse:
        """
        Chat with the chatbot.
        """
        relevant_documents = self.retriever.find_similar_docs(query=prompt, top_k=top_k)
        # First, format the prompt for each document
        document_str = ""
        total_num_tokens = 0
        for index, document in enumerate(relevant_documents):
            # use all attributes of document, except for created_at or vector, as the format parameter
            doc_formated_prompt = self.document_prompt_tempate.format_prompt(**{k: v for k, v in document.__dict__.items() if k not in ['created_at', 'vector']})
            num_tokens = len(self.enc.encode(doc_formated_prompt))
            if total_num_tokens + num_tokens > self.max_num_tokens_for_context:
                logger.warning(f"Exceeding max number of tokens for context: {self.max_num_tokens_for_context}, existing on {index}th document out of {len(relevant_documents)} documents")
                break
            total_num_tokens += num_tokens
            document_str += doc_formated_prompt + "\n"
        logger.debug(f"Document string: {document_str}")
        # Then, format the whole prompt
        whole_prompt = self.whole_prompt_template.format_prompt(context=document_str, prompt=prompt)
        logger.debug(f"Whole prompt: {whole_prompt}")
        response = self.llm_provider.prompt(whole_prompt)
        return ChatResponse(content=response, relevant_documents=relevant_documents)
    

