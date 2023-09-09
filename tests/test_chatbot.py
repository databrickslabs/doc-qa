import pytest
from unittest.mock import Mock, patch
from databricks.labs.doc_qa.chatbot.chatbot import OpenAILlmProvider, BaseChatBot, ChatResponse, LlmProvider
from databricks.labs.doc_qa.llm_providers import openai_provider
from databricks.labs.doc_qa.llm_utils import PromptTemplate
from databricks.labs.doc_qa.chatbot.retriever import BaseRetriever, Document


def test_openai_llm_provider():
    with patch.object(openai_provider, "request_openai") as mock_request:
        mock_request.return_value = {'content': 'Test response'}
        
        provider = OpenAILlmProvider(api_key='test_key', model='gpt-3', temperature=0.5)
        response = provider.prompt('Test prompt')
        
        assert response == 'Test response'
        mock_request.assert_called_once()

def test_base_chat_bot_creation():
    mock_llm_provider = Mock(spec=OpenAILlmProvider)
    mock_llm_provider.model = 'gpt-4'
    mock_retriever = Mock(spec=BaseRetriever)
    mock_whole_prompt_template = Mock(spec=PromptTemplate)
    mock_document_prompt_template = Mock(spec=PromptTemplate)

    bot = BaseChatBot(
        llm_provider=mock_llm_provider, 
        retriever=mock_retriever,
        whole_prompt_template=mock_whole_prompt_template,
        document_prompt_tempate=mock_document_prompt_template,
        max_num_tokens_for_context=3500
    )

    assert bot._llm_provider == mock_llm_provider
    assert bot._retriever == mock_retriever
    assert bot._whole_prompt_template == mock_whole_prompt_template
    assert bot._document_prompt_tempate == mock_document_prompt_template
    assert bot._max_num_tokens_for_context == 3500

def test_base_chat_bot():
    # create mock objects for dependencies
    mock_llm_provider = Mock(spec=OpenAILlmProvider)
    mock_llm_provider.prompt.return_value = 'Test response'
    mock_llm_provider.model = 'gpt-4'  # add this line
    mock_retriever = Mock(spec=BaseRetriever)
    mock_retriever.find_similar_docs.return_value = [
        Document(id='1', text='Doc 1', created_at=None, vector=None),
        Document(id='2', text='Doc 2', created_at=None, vector=None)
    ]
    mock_whole_prompt_template = Mock(spec=PromptTemplate)
    mock_document_prompt_template = Mock(spec=PromptTemplate)
    mock_whole_prompt_template.format_prompt.return_value = 'Formatted prompt'
    mock_document_prompt_template.format_prompt.return_value = 'Formatted document prompt'

    bot = BaseChatBot(llm_provider=mock_llm_provider, retriever=mock_retriever, 
                      whole_prompt_template=mock_whole_prompt_template, 
                      document_prompt_tempate=mock_document_prompt_template, 
                      max_num_tokens_for_context=3500)

    response = bot.chat('Test prompt')

    assert isinstance(response, ChatResponse)
    assert response.content == 'Test response'
    assert len(response.relevant_documents) == 2
    assert response.relevant_documents[0].text == 'Doc 1'
    assert response.relevant_documents[1].text == 'Doc 2'


def test_openai_llm_provider_exception():
    with patch.object(openai_provider, "request_openai") as mock_request:
        mock_request.side_effect = Exception('OpenAI request failed')
        
        provider = OpenAILlmProvider(api_key='test_key', model='gpt-3', temperature=0.5)
        with pytest.raises(Exception, match='OpenAI request failed'):
            provider.prompt('Test prompt')

        mock_request.assert_called_once()

def test_llm_provider():
    provider = LlmProvider()
    with pytest.raises(NotImplementedError):
        provider.prompt('Test prompt')

def test_chat_response():
    relevant_documents = [Document(id='1', text='Doc 1', created_at=None, vector=None)]
    response = ChatResponse(content='Test response', relevant_documents=relevant_documents)

    assert response.content == 'Test response'
    assert response.relevant_documents == relevant_documents


def test_openai_llm_provider_initialization():
    provider = OpenAILlmProvider(api_key='test_key', model='gpt-3', temperature=0.5, extra_arg='extra_val')
    assert provider._api_key == 'test_key'
    assert provider._model == 'gpt-3'
    assert provider._temperature == 0.5

def test_chat_response_initialization():
    relevant_documents = [Document(id='1', text='Doc 1', created_at=None, vector=None)]
    response = ChatResponse(content='Test response', relevant_documents=relevant_documents)

    assert response.content == 'Test response'
    assert response.relevant_documents == relevant_documents

def test_base_chat_bot_initialization_with_extra_args():
    mock_llm_provider = Mock(spec=OpenAILlmProvider)
    mock_llm_provider.model = 'gpt-4'
    mock_retriever = Mock(spec=BaseRetriever)
    mock_whole_prompt_template = Mock(spec=PromptTemplate)
    mock_document_prompt_template = Mock(spec=PromptTemplate)

    bot = BaseChatBot(
        llm_provider=mock_llm_provider, 
        retriever=mock_retriever,
        whole_prompt_template=mock_whole_prompt_template,
        document_prompt_tempate=mock_document_prompt_template,
        max_num_tokens_for_context=3500,
        extra_arg='extra_val'
    )

    assert bot._llm_provider == mock_llm_provider