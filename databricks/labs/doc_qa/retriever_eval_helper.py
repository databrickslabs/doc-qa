from databricks.labs.doc_qa.llm_utils import PromptTemplate
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import logging
import numpy as np
from databricks.labs.doc_qa.llm_utils import PromptTemplate
from databricks.labs.doc_qa.chatbot.retriever import (
    EmbeddingProvider,
    CsvRetriever,
    BgeEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
import tiktoken


def split_text_chunks(
    input_df, text_column_name, tokenizer, max_sequence_length, concurrency=100
):
    def split_text(text):
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_sequence_length:
            return [text]
        chunks = []
        for i in range(0, len(tokens), max_sequence_length):
            chunk = tokens[i : i + max_sequence_length]
            chunks.append(chunk)
        text_chunks = []
        for chunk in chunks:
            text_chunks.append(tokenizer.decode(chunk))
        return text_chunks

    df = input_df.copy()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        df["chunk"] = list(executor.map(split_text, df[text_column_name]))
    df = df.explode("chunk")
    logging.info(f"Split {len(input_df)} original rows into {len(df)} chunks")
    return df


def check_retrieved_rank(retriever, query, correct_source, top_k):
    documents = retriever.find_similar_docs(query=query, top_k=top_k)
    for i, doc in enumerate(documents):
        if doc.source == correct_source:
            return i
    return -1


def benchmark_retrieval(reference_df, retriever, top_k=5):
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = executor.map(
            lambda x: check_retrieved_rank(retriever, x[0], x[1], top_k),
            zip(reference_df["question"], reference_df["source"]),
        )
    ranks = list(results)
    # get the distribution of -1, 0, 1, 2 ... top_k
    ranks = pd.Series(ranks).value_counts().sort_index()
    ranks = ranks.reindex(range(-1, top_k))
    total_count = reference_df.shape[0]
    hit_so_far = 0
    accuracies = []
    for i in range(0, top_k):
        # i in ranks and not equal
        if i in ranks and ranks[i] and ranks[i] > 0:
            hit_so_far += ranks[i]
        accuracies.append(hit_so_far / total_count)

    accuracy_df = pd.DataFrame(accuracies, columns=["accuracy"])
    accuracy_df["rank"] = accuracy_df.index + 1
    return accuracy_df


def split_chunk_and_benchmark(
    ground_truth_df: pd.DataFrame,
    datasource_df: pd.DataFrame,
    tokenizer,
    embedding_provider: EmbeddingProvider,
    text_column_name="full_text",
    max_sequence_length=512,
    top_k=20,
):
    df_chunks = split_text_chunks(
        input_df=datasource_df,
        text_column_name=text_column_name,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        concurrency=500,
    )
    logging.info(f"Split {len(datasource_df)} rows into {len(df_chunks)} chunks")

    embed_prompt = PromptTemplate("""{chunk}""")

    print("Creating vector store from csv")
    csv_retriever = CsvRetriever.index_from_dataframe(
        df=df_chunks,
        embedding_provider=embedding_provider,
        embed_prompt_template=embed_prompt,
    )
    logging.info(f"Created vector store with {len(df_chunks)} documents")
    accuracies = benchmark_retrieval(ground_truth_df, csv_retriever, top_k=top_k)

    return accuracies, csv_retriever


def split_and_benchmark_bge(
    ground_truth_df: pd.DataFrame,
    datasource_df: pd.DataFrame,
    model_name: str = "BAAI/bge-base-en-v1.5",
    text_column_name="full_text",
    max_sequence_length=512,
    top_k=20,
    batch_size=500,
):
    embedding_provider = BgeEmbeddingProvider(
        model_name=model_name, batch_size=batch_size
    )
    from transformers import AutoTokenizer, AutoModel

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return split_chunk_and_benchmark(
        ground_truth_df=ground_truth_df,
        datasource_df=datasource_df,
        tokenizer=tokenizer,
        embedding_provider=embedding_provider,
        text_column_name=text_column_name,
        max_sequence_length=max_sequence_length,
        top_k=top_k,
    )


def split_and_benchmark_openai(
    ground_truth_df: pd.DataFrame,
    datasource_df: pd.DataFrame,
    api_key: str,
    model_name: str = "text-embedding-ada-002",
    text_column_name="full_text",
    max_sequence_length=8191,
    top_k=20,
    batch_size=1000,
):
    embedding_provider = OpenAIEmbeddingProvider(
        model_name=model_name, api_key=api_key, batch_size=batch_size
    )
    tokenizer = tiktoken.encoding_for_model(model_name)
    return split_chunk_and_benchmark(
        ground_truth_df=ground_truth_df,
        datasource_df=datasource_df,
        tokenizer=tokenizer,
        embedding_provider=embedding_provider,
        text_column_name=text_column_name,
        max_sequence_length=max_sequence_length,
        top_k=top_k,
    )
