from databricks.labs.doc_qa.llm_utils import PromptTemplate
from databricks.labs.doc_qa.evaluators.templated_evaluator import (
    OpenAIEvaluator,
    RetryPolicy,
)
from databricks.labs.doc_qa.variables.rag_eval_variables import (
    grading_prompt_template,
    openai_evaluator_function,
)


def gpt_4_evaluator():
    from databricks.labs.doc_qa.llm_utils import PromptTemplate

    retry_policy = RetryPolicy(max_retry_on_invalid_result=3, max_retry_on_exception=3)
    evaluator = OpenAIEvaluator(
        model="gpt-4",
        temperature=0.0,
        grading_prompt_tempate=grading_prompt_template,
        input_columns=["question", "answer", "context", "source"],
        openai_function=openai_evaluator_function,
        retry_policy=retry_policy,
        openai_retry_timeout=15,
    )

    return evaluator
