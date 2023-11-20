from datetime import datetime
import pandas as pd
from databricks.labs.doc_qa.llm_utils import PromptTemplate
from databricks.labs.doc_qa.logging_utils import logger
import openai
import faiss
import numpy as np
import json
import logging


class EmbeddingProvider:
    def embed_text(self, text, is_query=True):
        """
        Embed the query using the embedding provider.
        """
        pass

    def embed_texts(self, texts, is_query=True):
        """
        Embed the queries using the embedding provider.
        """
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    An OpenAIEmbeddingProvider object contains the input data for the evaluation dataframe.
    """

    def __init__(
        self,
        api_key: str,
        batch_size: int = 1000,
        model_name: str = "text-embedding-ada-002",
    ):
        self._api_key = api_key
        openai.api_key = api_key
        self.batch_size = batch_size
        self.model_name = model_name

    def embed_text(self, text, is_query=True):
        """
        Embed the query using the embedding provider.
        """
        response = openai.Embedding.create(input=text, model=self.model_name)
        return response["data"][0]["embedding"]

    def embed_texts(self, texts, is_query=True):
        """
        Embed the queries using the embedding provider.
        """
        if len(texts) == 0:
            return []
        BATCH_SIZE = self.batch_size
        embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            response = openai.Embedding.create(input=batch, model=self.model_name)
            embeddings.extend([item["embedding"] for item in response["data"]])
        return embeddings


class InstructorEmbeddingProvider(EmbeddingProvider):
    """
    An embedding provider for InstructorXL
    """

    def __init__(
        self,
        model_name: str = "hkunlp/instructor-xl",
        batch_size: int = 100,
        query_instruction: str = "Represent the question for retrieving supporting documents:",
        passage_instruction: str = "Represent the document for retrieval",
    ):
        from InstructorEmbedding import INSTRUCTOR
        import torch

        self._model_name = model_name
        self.model = INSTRUCTOR(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.passage_instruction = passage_instruction

    def embed_text(self, text, is_query=True):
        """
        Embed the query using the embedding provider.
        """
        return self.embed_texts(texts=[text], is_query=is_query)[0]

    def embed_texts(self, texts, is_query=True):
        """
        Embed the tetxs using the embedding provider.
        """
        import torch

        if len(texts) == 0:
            return []
        if is_query:
            logging.info(f"Embedding {len(texts)} queries as query")
            texts = [[self.query_instruction, text] for text in texts]
        else:
            logging.info(f"Embedding {len(texts)} queries as passage")
            texts = [[self.passage_instruction, text] for text in texts]

        BATCH_SIZE = self.batch_size
        total_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            # Tokenize the input texts
            input_texts = texts[i : i + BATCH_SIZE]
            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model.encode(input_texts)
                sentence_embeddings = torch.from_numpy(model_output)
            # normalize embeddings (TODO: figure out if this is needed for InstructorXL)
            sentence_embeddings = torch.nn.functional.normalize(
                sentence_embeddings, p=2, dim=1
            )

            total_embeddings.extend(sentence_embeddings.tolist())
        return total_embeddings


class GTEEmbeddingProvider(EmbeddingProvider):
    """
    An embedding provider for GTE-large
    """

    def __init__(self, model_name: str = "thenlper/gte-large", batch_size: int = 100):
        from sentence_transformers import (
            InputExample,
            models,
            SentenceTransformer,
            losses,
        )
        import torch

        self._model_name = model_name
        word_embedding_model = models.Transformer(model_name)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension()
        )
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size = batch_size

    def embed_text(self, text, is_query=True):
        """
        Embed the query using the embedding provider.
        """
        return self.embed_texts(texts=[text], is_query=is_query)[0]

    def embed_texts(self, texts, is_query=True):
        """
        Embed the tetxs using the embedding provider.
        """
        import torch

        if len(texts) == 0:
            return []
        if is_query:
            logging.info(
                f"GTE Embeddings doesn't discriminate based on query or passage"
            )

        BATCH_SIZE = self.batch_size
        total_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            # Tokenize the input texts
            input_texts = texts[i : i + BATCH_SIZE]
            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model.encode(input_texts)
                sentence_embeddings = torch.from_numpy(model_output)
            sentence_embeddings = torch.nn.functional.normalize(
                sentence_embeddings, p=2, dim=1
            )
            total_embeddings.extend(sentence_embeddings.tolist())
        return total_embeddings


class BgeEmbeddingProvider(EmbeddingProvider):
    """
    An OpenAIEmbeddingProvider object contains the input data for the evaluation dataframe.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        query_instruction="Represent this sentence for searching relevant passages:",
        batch_size: int = 100,
    ):
        from transformers import AutoTokenizer, AutoModel
        import torch

        self._model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.query_instruction = query_instruction
        self.batch_size = batch_size

    def embed_text(self, text, is_query=True):
        """
        Embed the query using the embedding provider.
        """
        return self.embed_texts(texts=[text], is_query=is_query)[0]

    def embed_texts(self, texts, is_query=True):
        """
        Embed the queries using the embedding provider.
        """
        import torch

        if len(texts) == 0:
            return []
        if is_query:
            logging.info(f"Embedding {len(texts)} queries as query")
            texts = [self.query_instruction + text for text in texts]
        else:
            logging.info(f"Embedding {len(texts)} queries as passage")

        BATCH_SIZE = self.batch_size
        total_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            # Tokenize the input texts
            input_texts = texts[i : i + BATCH_SIZE]

            # Tokenize sentences
            encoded_input = self.tokenizer(
                input_texts, padding=True, truncation=True, return_tensors="pt"
            )

            # Move the data to the device.
            for key, value in encoded_input.items():
                encoded_input[key] = value.to(self.device)

            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                # Perform pooling. In this case, cls pooling.
                sentence_embeddings = model_output[0][:, 0]
            # normalize embeddings
            sentence_embeddings = torch.nn.functional.normalize(
                sentence_embeddings, p=2, dim=1
            )

            total_embeddings.extend(sentence_embeddings.tolist())
        return total_embeddings


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

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        documents: list = None,
        column_names: list = None,
        **kwargs,
    ):
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
        vectors = np.array(vectors).astype("float32")  # Faiss works with float32 type
        self._faiss_index = faiss.IndexFlatL2(vectors.shape[1])
        self._faiss_index.add(vectors)
        self._id_to_document = {i: doc for i, doc in enumerate(self._documents)}
        logger.info("Finished building Faiss index")

    def find_similar_docs(self, query, top_k=3):
        """
        Find the top_k most similar documents to the query.
        """
        query_vector = self._embedding_provider.embed_text(text=query, is_query=True)
        ## Query FAISS to get the top_k most similar documents
        distances, indices = self._faiss_index.search(
            np.array([query_vector]).astype("float32"), top_k
        )
        docs = [
            self._id_to_document[i] for i in indices[0]
        ]  # Return the top_k documents
        # add the distance as the 'distance'  attribute to each document
        for doc, distance in zip(docs, distances[0]):
            doc.distance = distance
        return docs

    def save_as_csv(self, path):
        """
        Save the vector store as a csv file.
        """
        # Convert each document into a row in the csv file
        column_names = self._additional_column_names + ["vector", "created_at"]
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
        additional_column_names = [
            column_name
            for column_name in df.columns
            if column_name not in ["vector", "created_at"]
        ]
        vector_store = cls(
            documents=documents,
            column_names=additional_column_names,
            embedding_provider=embedding_provider,
        )
        vector_store._build_faiss_index()
        return vector_store

    @classmethod
    def index_from_csv(
        cls,
        csv_path,
        embed_prompt_template: PromptTemplate,
        embedding_provider: EmbeddingProvider,
    ):
        # Load the csv as a pandas dataframe
        df = pd.read_csv(csv_path)
        return cls.index_from_dataframe(
            df=df,
            embed_prompt_template=embed_prompt_template,
            embedding_provider=embedding_provider,
        )

    @classmethod
    def index_from_dataframe(
        cls,
        df: pd.DataFrame,
        embed_prompt_template: PromptTemplate,
        embedding_provider: EmbeddingProvider,
    ):
        additional_column_names = [
            column_name
            for column_name in df.columns
            if column_name not in ["vector", "created_at"]
        ]
        logger.info(f"Got additional column names: {additional_column_names}")
        # Convert each row into a document
        documents = []
        queries = []
        for index, row in df.iterrows():
            created_at = datetime.now()
            document = Document(created_at=created_at, **row.to_dict())
            query = embed_prompt_template.format(
                **{
                    column_name: getattr(document, column_name)
                    for column_name in additional_column_names
                }
            )
            documents.append(document)
            queries.append(query)
        logger.info(f"Loaded {len(documents)} documents")
        embeddings = embedding_provider.embed_texts(texts=queries, is_query=False)
        for document, embedding in zip(documents, embeddings):
            document.vector = embedding
        logger.info(f"Embedded {len(documents)} documents")

        vector_store = cls(
            documents=documents,
            column_names=additional_column_names,
            embedding_provider=embedding_provider,
        )
        vector_store._build_faiss_index()
        return vector_store


class BricksIndexRetriever(BaseRetriever):
    def __init__(
        self,
        workspace_url: str,
        token: str,
        columns: list,
        endpoint_name: str,
        index_name: str,
        **kwargs,
    ):
        # VectorSearchClient won't work with trailing "/"
        self._workspace_url = (
            workspace_url[:-1] if workspace_url.endswith("/") else workspace_url
        )
        self._token = token
        from databricks.vector_search.client import VectorSearchClient

        self._vs = VectorSearchClient(
            workspace_url=self._workspace_url, personal_access_token=self._token
        )
        self._columns = columns
        self._endpoint_name = endpoint_name
        self._index_name = index_name
        self._index = self._vs.get_index(
            endpoint_name=self._endpoint_name, index_name=self._index_name
        )
        super().__init__(**kwargs)

    def find_similar_docs(self, query, top_k=3):
        result = self._index.similarity_search(
            columns=self._columns,
            query_text=query,
            num_results=top_k,
        )
        documents = []
        for item in result["result"]["data_array"]:
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

        payload = json.dumps(
            {
                "q": query,
                "num": top_k,
            }
        )
        headers = {"X-API-KEY": self._api_key, "Content-Type": "application/json"}

        response = requests.request("POST", url, headers=headers, data=payload)
        response_json = response.json()
        documents = []
        for result in response_json["organic"]:
            document = Document(
                created_at=datetime.now(),
                title=result["title"],
                link=result["link"],
                snippet=result["snippet"],
            )
            documents.append(document)
        return documents


class ChromaPersistedRetriever(BaseRetriever):
    """
    A VectorStore object contains the input data for the evaluation dataframe.
    """

    def __init__(
        self,
        file_path: str,
        collection_name: str,
        openai_key: str,
        embed_prompt: PromptTemplate = None,
        batch_size: int = 1000,
    ):
        import chromadb
        from chromadb.utils import embedding_functions

        self._embed_prompt = embed_prompt

        self.client = chromadb.PersistentClient(path=file_path)

        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_key, model_name="text-embedding-ada-002"
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.openai_ef,
        )
        self._batch_size = batch_size

    def add_documents(self, input_df: pd.DataFrame, id_column: str = None):
        import uuid

        embeddings = []
        metadatas = []
        texts = []
        ids = []
        # store the rows as dictionary objects in metadatas
        for _, row in input_df.iterrows():
            metadata = row.to_dict()
            # if metadata doesn't have created_at, use string of now
            if "created_at" not in metadata:
                metadata["created_at"] = str(datetime.now())
            metadatas.append(metadata)
            texts.append(self._embed_prompt.format(**row.to_dict()))
            if id_column is None:
                ids.append(str(uuid.uuid4()))
            else:
                ids.append(metadata[id_column])
        # split into batches
        for i in range(0, len(texts), self._batch_size):
            batch_embeddings = self.openai_ef(texts[i : i + self._batch_size])
            embeddings.extend(batch_embeddings)
        logging.info(f"Successfully generated {len(embeddings)} embeddings")
        # add the embeddings to the collection in batches
        for i in range(0, len(embeddings), self._batch_size):
            self.collection.add(
                embeddings=embeddings[i : i + self._batch_size],
                metadatas=metadatas[i : i + self._batch_size],
                ids=ids[i : i + self._batch_size],
            )
        logging.info(f"Successfully added {len(embeddings)} documents to collection")

    def find_similar_docs(self, query, top_k=3):
        """
        Find the top_k most similar documents to the query.
        """
        results = self.collection.query(query_texts=[query], n_results=top_k)
        metadatas = results["metadatas"][0]
        texts = results["documents"][0]
        docs = []
        for index, metadata in enumerate(metadatas):
            doc = Document(**metadata)
            docs.append(doc)
            setattr(doc, "text", texts[index])
        return docs
