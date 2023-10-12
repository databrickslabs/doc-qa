from databricks.labs.doc_qa.llm_utils import PromptTemplate
import pandas as pd
import os
from databricks.labs.doc_qa.evaluators.templated_evaluator import (
    OpenAIEvaluator,
    RetryPolicy,
)
from databricks.labs.doc_qa.variables.doc_qa_template_variables import (
    get_openai_grading_template_and_function,
)
from databricks.labs.doc_qa.logging_utils import logger


def gpt_4_evaluator():
    retry_policy = RetryPolicy(max_retry_on_invalid_result=3, max_retry_on_exception=3)
    (
        openai_grading_prompt,
        openai_grading_function,
    ) = get_openai_grading_template_and_function(scale=10, level_of_details=2)
    return OpenAIEvaluator(
        model="gpt-4",
        temperature=0.0,
        grading_prompt_tempate=openai_grading_prompt,
        input_columns=["question", "answer", "context"],
        openai_function=openai_grading_function,
        retry_policy=retry_policy,
    )


def vllm_vicuna_model_generator(url, pat_token, model_name):
    from databricks.labs.doc_qa.model_generators.model_generator import (
        vLllmOpenAICompletionFormatModelGenerator,
    )
    from databricks.labs.doc_qa.variables.doc_qa_template_variables import (
        vicuna_prompt_format_func,
        doc_qa_task_prompt_template,
    )

    return vLllmOpenAICompletionFormatModelGenerator(
        url=url,
        pat_token=pat_token,
        prompt_formatter=doc_qa_task_prompt_template,
        batch_size=1,
        model_name=model_name,
        format_prompt_func=vicuna_prompt_format_func,
        concurrency=100,
    )


def vllm_llama2_model_generator(url, pat_token, model_name):
    from databricks.labs.doc_qa.model_generators.model_generator import (
        vLllmOpenAICompletionFormatModelGenerator,
    )
    from databricks.labs.doc_qa.variables.doc_qa_template_variables import (
        llama2_prompt_format_func,
        doc_qa_task_prompt_template,
    )

    return vLllmOpenAICompletionFormatModelGenerator(
        url=url,
        pat_token=pat_token,
        prompt_formatter=doc_qa_task_prompt_template,
        batch_size=1,
        model_name=model_name,
        format_prompt_func=llama2_prompt_format_func,
        concurrency=100,
    )


def generate_and_evaluate(
    input_df, model_generator, evaluator, temperature=0, max_tokens=200
):
    generate_result = model_generator.run_tasks(
        input_df=input_df, temperature=temperature, max_tokens=max_tokens
    )

    result_df = generate_result.to_dataframe()

    logger.info(f"Finished generating {len(result_df)} rows, starting evaluation")
    return evaluator.run_eval(dataset_df=result_df, concurrency=20, catch_error=True)


def evaluate_using_vllm_locally(
    input_df,
    hf_model_name,
    prompt_tempate_format_func,
    temperature=0,
    max_tokens=200,
    max_num_batched_tokens=None,
):
    from databricks.labs.doc_qa.model_generators.model_generator import (
        vLllmLocalModelGenerator,
    )
    from databricks.labs.doc_qa.variables.doc_qa_template_variables import (
        doc_qa_task_prompt_template,
    )

    model_generator = vLllmLocalModelGenerator(
        hf_model_name=hf_model_name,
        format_prompt_func=prompt_tempate_format_func,
        prompt_formatter=doc_qa_task_prompt_template,
        max_num_batched_tokens=max_num_batched_tokens,
        trust_remote_code=True,
    )

    evaluator = gpt_4_evaluator()
    generate_result = model_generator.run_tasks(
        input_df=input_df, temperature=temperature, max_tokens=max_tokens
    )

    result_df = generate_result.to_dataframe()

    logger.info(f"Finished generating {len(result_df)} rows, starting evaluation")
    return evaluator.run_eval(dataset_df=result_df, concurrency=20, catch_error=True)
