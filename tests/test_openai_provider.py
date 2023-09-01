import os
import pytest
import requests_mock
from databricks.doc_qa.llm_providers.openai_provider import request_openai, StatusCode429Error

def setup_module(module):
    os.environ['OPENAI_API_KEY'] = 'test_openai_key'
    os.environ['OPENAI_ORGANIZATION'] = 'test_openai_org'
    os.environ['PAT_TOKEN'] = 'test_pat_token'

def teardown_module(module):
    os.environ.pop('OPENAI_API_KEY')
    os.environ.pop('OPENAI_ORGANIZATION')
    os.environ.pop('PAT_TOKEN')

def test_request_openai_successful():
    with requests_mock.Mocker() as m:
        url = "https://api.openai.com/v1/chat/completions"
        m.post(url, text='{"choices": [{"message": "test_message"}]}')
        response_message = request_openai(["test_message"])
        assert response_message == "test_message"

def test_request_openai_status_code_not_200():
    with requests_mock.Mocker() as m:
        url = "https://api.openai.com/v1/chat/completions"
        m.post(url, status_code=400)
        with pytest.raises(Exception) as e:
            request_openai(["test_message"])
        assert str(e.value) == "Request failed with status code 400 and response "