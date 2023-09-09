from datetime import datetime
import pandas as pd
from databricks.labs.doc_qa.llm_utils import PromptTemplate
import openai
import logging
import faiss
import numpy as np
import json


logger = logging.getLogger(__name__)


class EmbeddingProvider:

    def embed_query(self, query):
        """
        Embed the query using the embedding provider.
        """
        pass
    
    def embed_queries(self, queries):
        """
        Embed the queries using the embedding provider.
        """
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    An OpenAIEmbeddingProvider object contains the input data for the evaluation dataframe.
    """
    def __init__(self, api_key: str, **kwargs):
        self._api_key = api_key
        openai.api_key = api_key
    
    def embed_query(self, query):
        """
        Embed the query using the embedding provider.
        """
        response = openai.Embedding.create(
            input=query,
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']


    def embed_queries(self, queries):
        """
        Embed the queries using the embedding provider.
        """
        if len(queries) == 0:
            return []
        BATCH_SIZE = 500
        embeddings = []
        for i in range(0, len(queries), BATCH_SIZE):
            batch = queries[i:i+BATCH_SIZE]
            response = openai.Embedding.create(
                input=batch,
                model="text-embedding-ada-002"
            )
            embeddings.extend([item['embedding'] for item in response['data']])
        return embeddings


class Document:
    """
    A RowInput object contains the input data for a single row in the evaluation dataframe.
    """
    def __init__(self, created_at: datetime, **kwargs):
        self.created_at = created_at
        for key, value in kwargs.items():
            setattr(self, key, value)

class BaseRetriever:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def find_similar_docs(self, query, top_k=3):
        """
        Find the top_k most similar documents to the query.
        """
        pass


class CsvRetriever(BaseRetriever):
    """
    A VectorStore object contains the input data for the evaluation dataframe.
    """
    def __init__(self, embedding_provider: EmbeddingProvider, documents: list = None, column_names: list = None, **kwargs):
        self._documents = documents
        self._additional_column_names = column_names
        self._embedding_provider = embedding_provider
        self._faiss_index = None
        self._id_to_document = {}

    def _build_faiss_index(self):
        """
        Build the Faiss index for the documents.
        """
        logger.info("Start building Faiss index")
        vectors = [doc.vector for doc in self._documents]
        vectors = np.array(vectors).astype('float32')  # Faiss works with float32 type
        self._faiss_index = faiss.IndexFlatL2(vectors.shape[1])
        self._faiss_index.add(vectors)
        self._id_to_document = {i: doc for i, doc in enumerate(self._documents)}
        logger.info("Finished building Faiss index")

    def find_similar_docs(self, query, top_k=3):
        """
        Find the top_k most similar documents to the query.
        """
        query_vector = self._embedding_provider.embed_query(query)
        ## Query FAISS to get the top_k most similar documents
        distances, indices = self._faiss_index.search(np.array([query_vector]).astype('float32'), top_k)
        docs = [self._id_to_document[i] for i in indices[0]]  # Return the top_k documents
        # add the distance as the 'distance'  attribute to each document
        for doc, distance in zip(docs, distances[0]):
            doc.distance = distance
        return docs


    def save_as_csv(self, path):
        """
        Save the vector store as a csv file.
        """
        # Convert each document into a row in the csv file
        column_names = self._additional_column_names + ['vector', 'created_at']
        rows = []
        for document in self._documents:
            row = [getattr(document, column_name) for column_name in column_names]
            rows.append(row)
        # Save the csv file
        df = pd.DataFrame(rows, columns=column_names)
        df.to_csv(path, index=False)
        
    @classmethod
    def load_from_csv(cls, path, embedding_provider: EmbeddingProvider):
        """
        Load the vector store from a csv file.
        """
        # First, load as pandas dataframe
        df = pd.read_csv(path)
        # Convert each row into a document
        documents = []
        for index, row in df.iterrows():
            document = Document(**row.to_dict())
            # Convert the vector from string to list of floats
            document.vector = json.loads(document.vector)
            documents.append(document)
        # excluding the vector and created_at columns
        additional_column_names = [column_name for column_name in df.columns if column_name not in ['vector', 'created_at']]
        vector_store = cls(documents=documents, column_names=additional_column_names, embedding_provider=embedding_provider)
        vector_store._build_faiss_index()
        return vector_store

    @classmethod
    def index_from_csv(cls, csv_path, embed_prompt_template: PromptTemplate, embedding_provider: EmbeddingProvider):
        # Load the csv as a pandas dataframe
        df = pd.read_csv(csv_path)
        additional_column_names = [column_name for column_name in df.columns if column_name not in ['vector', 'created_at']]
        logger.info(f"Got additional column names: {additional_column_names}")
        # Convert each row into a document
        documents = []
        queries = []
        for index, row in df.iterrows():
            created_at = datetime.now()
            document = Document(created_at=created_at, **row.to_dict())
            query = embed_prompt_template.format_prompt(**{column_name: getattr(document, column_name) for column_name in additional_column_names})
            documents.append(document)
            queries.append(query)
        logger.info(f"Loaded {len(documents)} documents from {csv_path}")
        embeddings = embedding_provider.embed_queries(queries)
        for document, embedding in zip(documents, embeddings):
            document.vector = embedding
        logger.info(f"Embedded {len(documents)} documents")

        vector_store = cls(documents=documents, column_names=additional_column_names, embedding_provider=embedding_provider)
        vector_store._build_faiss_index()
        return vector_store
    

class BricksIndexRetriever(BaseRetriever):
    def __init__(self, workspace_url: str, token: str, columns: list, index_name: str, **kwargs):
        self._workspace_url = workspace_url
        self._token = token
        from databricks.vector_search.client import VectorSearchClient
        self._vs = VectorSearchClient(workspace_url=workspace_url, token=token)
        self._columns = columns
        self._index_name = index_name
        super().__init__(**kwargs)

    def find_similar_docs(self, query, top_k=3):
        result = self._vs.similarity_search(
            index_name=self._index_name,
            query_text=query,
            columns=self._columns,
            num_results=top_k
        )
        documents = []
        for item in result['result']['data_array']:
            # Under the item, the value for the corresponding column is stored under the corresponding index
            document = Document(created_at=datetime.now())
            for index, column in enumerate(self._columns):
                setattr(document, column, item[index])
            documents.append(document)
        return documents


class SerperRetriever(BaseRetriever):
    def __init__(self, api_key: str, **kwargs):
        self._api_key = api_key
        super().__init__(**kwargs)

    def find_similar_docs(self, query, top_k=3):
        import requests
        import json

        url = "https://google.serper.dev/search"

        payload = json.dumps({
            "q": query,
            "num": top_k,
        })
        headers = {
            'X-API-KEY': self._api_key,
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        response_json = response.json()
        documents = []
        for result in response_json['organic']:
            document = Document(created_at=datetime.now(), title=result['title'], link=result['link'], snippet=result['snippet'])
            documents.append(document)
        return documents