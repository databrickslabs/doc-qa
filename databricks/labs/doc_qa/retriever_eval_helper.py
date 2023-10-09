from databricks.labs.doc_qa.llm_utils import PromptTemplate
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import logging
import numpy as np


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
    precisions = []
    for i in range(0, top_k):
        # i in ranks and not equal
        if i in ranks and ranks[i] and ranks[i] > 0:
            hit_so_far += ranks[i]
        precisions.append(hit_so_far / total_count)
    return precisions
