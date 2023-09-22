from databricks.labs.doc_qa.llm_utils import PromptTemplate
import pandas as pd
import os
from databricks.labs.doc_qa.evaluators.templated_evaluator import (
    OpenAIEvaluator,
    AnthropicEvaluator,
    ParameterDef,
    NoRetryPolicy,
    RetryPolicy,
)
from databricks.labs.doc_qa.variables.doc_qa_template_variables import (
    anthropic_grading_template_scale_3,
    anthropic_grading_template_scale_1,
)
from databricks.labs.doc_qa.variables.doc_qa_template_variables import (
    get_openai_grading_template_and_function,
)
from databricks.labs.doc_qa.llm_utils import PromptTemplate
from databricks.labs.doc_qa.model_generators.model_generator import (
    OpenAiModelGenerator,
    BaseModelGenerator,
    BatchGenerateResult,
    RowGenerateResult,
    DriverProxyModelGenerator,
)

# show debug log for all loggers
import logging

logging.basicConfig(level=logging.INFO)


def gpt_4_evaluator():
    retry_policy = RetryPolicy(max_retry_on_invalid_result=3, max_retry_on_exception=3)
    (
        openai_grading_prompt,
        openai_grading_function,
    ) = get_openai_grading_template_and_function(scale=10, level_of_details=2)
    openai_gpt_4_evaluator = OpenAIEvaluator(
        model="gpt-4",
        temperature=0.0,
        grading_prompt_tempate=openai_grading_prompt,
        input_columns=["question", "answer", "context"],
        openai_function=openai_grading_function,
        retry_policy=retry_policy,
    )
    return openai_gpt_4_evaluator


def vllm_vicuna_model_generator(url, pat_token, model_name):
    from databricks.labs.doc_qa.model_generators.model_generator import (
        vLllmOpenAICompletionFormatModelGenerator,
    )
    from databricks.labs.doc_qa.variables.doc_qa_template_variables import (
        vicuna_format_prompt_func,
        doc_qa_task_prompt_template,
    )

    model_generator = vLllmOpenAICompletionFormatModelGenerator(
        url=url,
        pat_token=pat_token,
        prompt_formatter=doc_qa_task_prompt_template,
        batch_size=1,
        model_name=model_name,
        format_prompt_func=vicuna_format_prompt_func,
        concurrency=100,
    )
    return model_generator


def vllm_llama_model_generator(url, pat_token, model_name):
    from databricks.labs.doc_qa.model_generators.model_generator import (
        vLllmOpenAICompletionFormatModelGenerator,
    )
    from databricks.labs.doc_qa.variables.doc_qa_template_variables import (
        llama2_format_prompt_func,
        doc_qa_task_prompt_template,
    )

    model_generator = vLllmOpenAICompletionFormatModelGenerator(
        url=url,
        pat_token=pat_token,
        prompt_formatter=doc_qa_task_prompt_template,
        batch_size=1,
        model_name=model_name,
        format_prompt_func=llama2_format_prompt_func,
        concurrency=100,
    )
    return model_generator
