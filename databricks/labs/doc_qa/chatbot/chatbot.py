from databricks.labs.doc_qa.llm_providers import openai_provider
from databricks.labs.doc_qa.llm_utils import PromptTemplate
import openai
from databricks.labs.doc_qa.chatbot.retriever import Document, BaseRetriever
from databricks.labs.doc_qa.logging_utils import logger
import tiktoken
from concurrent.futures import ThreadPoolExecutor


class LlmProvider:
    def prompt(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError()


class OpenAILlmProvider(LlmProvider):
    def __init__(self, api_key: str, model_name: str, temperature, **kwargs):
        self._api_key = api_key
        self.model_name = model_name
        self._temperature = temperature
        openai.api_key = api_key

    def prompt(self, prompt: str, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        response_message = openai_provider.request_openai(
            messages=messages,
            functions=[],
            model=self.model_name,
            temperature=self._temperature,
        )
        return response_message["content"]


class ChatResponse:
    query: str
    content: str
    relevant_documents: list[Document]

    def __init__(self, query: str, content: str, relevant_documents: list[Document]):
        self.query = query
        self.content = content
        self.relevant_documents = relevant_documents


class BaseChatBot:
    def __init__(
        self,
        llm_provider: str,
        retriever: BaseRetriever,
        whole_prompt_template: PromptTemplate,
        document_prompt_tempate: PromptTemplate,
        max_num_tokens_for_context: int = 3500,
        **kwargs,
    ):
        self._llm_provider = llm_provider
        self._retriever = retriever
        self._whole_prompt_template = whole_prompt_template
        self._document_prompt_tempate = document_prompt_tempate
        self._max_num_tokens_for_context = max_num_tokens_for_context
        self._enc = tiktoken.encoding_for_model(self._llm_provider.model_name)

    def chat_batch(
        self, queries: list[str], top_k=1, concurrency: int = 20, **kwargs
    ) -> list[ChatResponse]:
        logger.info(
            f"Start chatting with {len(queries)} queries using concurrency {concurrency} and top_k {top_k}"
        )
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            results = executor.map(
                lambda query: self.chat(query=query, top_k=top_k, **kwargs),
                queries,
            )
            return list(results)

    def chat(self, query: str, top_k=1, **kwargs) -> ChatResponse:
        """
        Chat with the chatbot.
        """
        relevant_documents = self._retriever.find_similar_docs(query=query, top_k=top_k)
        # First, format the prompt for each document
        document_str = ""
        total_num_tokens = 0
        included_documents = []
        for index, document in enumerate(relevant_documents):
            # use all attributes of document, except for created_at or vector, as the format parameter
            doc_formated_prompt = self._document_prompt_tempate.format(
                **{
                    k: v
                    for k, v in document.__dict__.items()
                    if k not in ["created_at", "vector"]
                }
            )
            num_tokens = len(self._enc.encode(doc_formated_prompt))
            if total_num_tokens + num_tokens > self._max_num_tokens_for_context:
                logger.warning(
                    f"Exceeding max number of tokens for context: {self._max_num_tokens_for_context}, existing on {index}th document out of {len(relevant_documents)} documents"
                )
                break
            total_num_tokens += num_tokens
            document_str += doc_formated_prompt + "\n"
            included_documents.append(document)
        logger.debug(f"Document string: {document_str}")
        # Then, format the whole prompt
        whole_prompt = self._whole_prompt_template.format(
            context=document_str, query=query
        )
        logger.debug(f"Whole prompt: {whole_prompt}")
        response = self._llm_provider.prompt(whole_prompt)
        return ChatResponse(
            query=query, content=response, relevant_documents=included_documents
        )


class TopDocRelevanceChatbot(BaseChatBot):
    def __init__(
        self,
        llm_provider: str,
        retriever: BaseRetriever,
        whole_prompt_template: PromptTemplate,
        document_prompt_tempate: PromptTemplate,
        max_num_tokens_for_context: int = 3500,
        **kwargs,
    ):
        super().__init__(
            llm_provider=llm_provider,
            retriever=retriever,
            whole_prompt_template=whole_prompt_template,
            document_prompt_tempate=document_prompt_tempate,
            max_num_tokens_for_context=max_num_tokens_for_context,
            **kwargs,
        )

    def get_doc_relevance_score(self, query: str, document: Document) -> float:
        prompt = f"""You are a helpful assistant good at rating the relevance of the document to a given question based on a set of requirements. 
    And your task is to provide a one line justification and a rating of between 0 to 2 for the relevance. 
    And you'll be provided with a function to submit your justification and relevance rating.

    Below are the Requirements:
    - If the document is complete irrelevant to the question, please rate 0.
    - If the document can be used to answer part of the question, please rate 1.
    - If the document can be directly used to answer the question, please rate 2.
    
    Below is the question:

    {query}

    Below is the document:

    {document.text}

    Please submit your rate of relevance (0-2) for the question above:
        """

        messages = [{"role": "user", "content": prompt}]

        submit_function = {
            "name": "submit_function",
            "description": "Call this function to submit the step by step reasoning and the relevance score between 0 to 2.",
            "parameters": {
                "type": "object",
                "properties": {
                    "step_by_step_reasoning": {
                        "type": "string",
                        "description": "The section for step by step reasoning.",
                    },
                    "relevance_score": {
                        "type": "number",
                        "description": "The relevance score between 0 to 2.",
                    },
                },
                "required": ["step_by_step_reasoning", "relevance_score"],
            },
        }

        response = openai_provider.request_openai(
            messages=messages,
            model="gpt-3.5-turbo",
            functions=[submit_function],
            temperature=0.0,
        )
        func_args = json.loads(response["function_call"]["arguments"])
        step_by_step_reasoning = func_args["step_by_step_reasoning"]
        relevance_score = func_args["relevance_score"]
        return relevance_score

    def chat(self, query: str, top_k=1, **kwargs) -> ChatResponse:
        relevant_documents = self._retriever.find_similar_docs(query=query, top_k=top_k)

        # get the relevance_score for each document in parallel, get a dictionary of document -> relevance_score
        with ThreadPoolExecutor(max_workers=20) as executor:
            results = executor.map(
                lambda document: (
                    document,
                    self.get_doc_relevance_score(query=query, document=document),
                ),
                relevant_documents,
            )
            document_to_relevance_score = dict(results)

        # sort the documents by relevance_score
        sorted_documents = sorted(
            document_to_relevance_score.items(), key=lambda x: x[1], reverse=True
        )
        for document, relevance_score in sorted_documents:
            logger.info(f"[Relevance={relevance_score}] for document {document.source}")

        # take the top 1 document for sorted_documents
        sorted_documents = sorted_documents[:1]

        # First, format the prompt for each document
        document_str = ""
        total_num_tokens = 0
        included_documents = []
        for index, document in enumerate(sorted_documents):
            # use all attributes of document, except for created_at or vector, as the format parameter
            doc_formated_prompt = self._document_prompt_tempate.format(
                **{
                    k: v
                    for k, v in document.__dict__.items()
                    if k not in ["created_at", "vector"]
                }
            )
            num_tokens = len(self._enc.encode(doc_formated_prompt))
            if total_num_tokens + num_tokens > self._max_num_tokens_for_context:
                logger.warning(
                    f"Exceeding max number of tokens for context: {self._max_num_tokens_for_context}, existing on {index}th document out of {len(relevant_documents)} documents"
                )
                break
            total_num_tokens += num_tokens
            document_str += doc_formated_prompt + "\n"
            included_documents.append(document)
        logger.debug(f"Document string: {document_str}")
        # Then, format the whole prompt
        whole_prompt = self._whole_prompt_template.format(
            context=document_str, query=query
        )
        logger.debug(f"Whole prompt: {whole_prompt}")
        response = self._llm_provider.prompt(whole_prompt)
        return ChatResponse(
            query=query, content=response, relevant_documents=included_documents
        )
