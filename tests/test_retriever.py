import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from datetime import datetime
from databricks.labs.doc_qa.chatbot.retriever import (
    CsvRetriever,
    OpenAIEmbeddingProvider,
    Document,
)
import json


def test_openai_embedding_provider_single_query(mocker):
    mocker.patch(
        "openai.Embedding.create",
        return_value={"data": [{"embedding": [0.1, 0.2, 0.3]}]},
    )
    provider = OpenAIEmbeddingProvider(api_key="test-key")
    result = provider.embed_query("test query")
    assert result == [0.1, 0.2, 0.3]


def test_openai_embedding_provider_batch_query(mocker):
    mocker.patch(
        "openai.Embedding.create",
        return_value={
            "data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]
        },
    )
    provider = OpenAIEmbeddingProvider(api_key="test-key")
    result = provider.embed_queries(["test query 1", "test query 2"])
    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


@pytest.fixture
def mock_faiss_index(mocker):
    index = mocker.MagicMock()
    index.search.return_value = ([0.1], [[0]])  # Distance and indices
    return index


@pytest.fixture
def mock_embedding_provider(mocker):
    provider = mocker.MagicMock()
    provider.embed_query.return_value = np.array([0.1, 0.2, 0.3])
    return provider


@pytest.fixture
def mock_document():
    document = Document(created_at=datetime.now())
    document.vector = [0.1, 0.2, 0.3]
    document.title = "test title"
    return document


@patch("faiss.IndexFlatL2")
def test_build_faiss_index(mock_faiss_index, mock_document, mocker):
    mock_document.vector = np.array([0.1, 0.2, 0.3])
    retriever = CsvRetriever(
        embedding_provider=mocker.MagicMock(),
        documents=[mock_document],
        column_names=["title", "content"],
    )
    retriever._build_faiss_index()
    mock_faiss_index.assert_called_once_with(len(mock_document.vector))
    assert retriever._id_to_document == {0: mock_document}


@patch("pandas.DataFrame.to_csv")
def test_save_as_csv(mock_to_csv, mock_document):
    mock_document.vector = np.array([0.1, 0.2, 0.3]).tolist()
    mock_document.content = "test content"
    retriever = CsvRetriever(
        embedding_provider=MagicMock(),
        documents=[mock_document],
        column_names=["title", "content"],
    )
    retriever.save_as_csv("test_path.csv")
    mock_to_csv.assert_called_once_with("test_path.csv", index=False)


@patch("pandas.read_csv")
def test_load_from_csv(mock_read_csv, mock_embedding_provider, mock_document):
    mock_read_csv.return_value = pd.DataFrame.from_records(
        [
            {
                "title": "test title",
                "content": "test content",
                "vector": json.dumps([0.1, 0.2, 0.3]),
                "created_at": str(mock_document.created_at),
            }
        ]
    )
    retriever = CsvRetriever.load_from_csv(
        "test_path.csv", embedding_provider=mock_embedding_provider
    )
    assert retriever._documents[0].vector == [0.1, 0.2, 0.3]


@patch("pandas.read_csv")
def test_index_from_csv(mock_read_csv, mock_embedding_provider, mocker):
    mock_embedding_provider.embed_queries.return_value = [[0.1, 0.2, 0.3]]
    mock_read_csv.return_value = pd.DataFrame.from_records(
        [{"title": "test title", "content": "test content", "vector": [0.1, 0.2, 0.3]}]
    )
    template = mocker.MagicMock()
    retriever = CsvRetriever.index_from_csv(
        "test_path.csv",
        embed_prompt_template=template,
        embedding_provider=mock_embedding_provider,
    )
    assert (
        retriever._documents[0].vector
        == mock_embedding_provider.embed_queries.return_value[0]
    )
