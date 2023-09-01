import pytest
import pandas as pd
from unittest import mock
from databricks.doc_qa.evaluators.templated_evaluator import BaseLlmEvaluator, PromptTemplate, DefaultRetryPolicy, ParameterType, ParameterDef, RowInput, RowEvalResult, EvalResult, RetryPolicy
from unittest.mock import patch
from databricks.doc_qa.evaluators.templated_evaluator import OpenAIEvaluator, AnthropicEvaluator, RowInput, RowEvalResult, ParameterDef, ParameterType, PromptTemplate, DefaultRetryPolicy, NoRetryPolicy, RetryPolicy
import json

# Sample function to create a BaseLlmEvaluator instance
def create_sample_evaluator():
    grading_prompt_template = PromptTemplate(
        template_str="This is a template with {var1} and {var2}.",
        variables=['var1', 'var2']
    )
    return BaseLlmEvaluator(
        grading_prompt_tempate=grading_prompt_template,
        input_columns=['var1', 'var2'],
        model="model_name",
        temperature=0.8,
        output_parameters=[ParameterType.STRING]
    )

def test_init_base_llm_evaluator():
    evaluator = create_sample_evaluator()
    
    # Checking the properties of the created evaluator
    assert evaluator.input_columns == ['var1', 'var2']
    assert evaluator.model == "model_name"
    assert evaluator.temperature == 0.8

@patch('databricks.doc_qa.evaluators.templated_evaluator.openai_provider.request_openai')
def test_openaievaluator(mock_request_openai):
    # Prepare the mock response
    mock_request_openai.return_value = {"function_call": {"arguments": json.dumps({"result": 5.0})}}

    # Initialize the evaluator
    openai_evaluator = OpenAIEvaluator(
        model="gpt-4", 
        temperature=0.5, 
        grading_prompt_tempate=PromptTemplate("{input}"), 
        input_columns=["input"], 
        openai_function={"name": "addition", "parameters": {"properties": {"result": {"type": "number"}}}}
    )

    # Test the grade_row method
    row_input = RowInput(input=3)
    result = openai_evaluator.grade_row(row_input)
    
    # Assert the result
    assert result.is_successful
    assert result.result == 5.0
    
@patch('databricks.doc_qa.evaluators.templated_evaluator.anthropic_provider.request_anthropic')
def test_anthropicevaluator(mock_request_anthropic):
    # Prepare the mock response
    mock_request_anthropic.return_value = "addition result: 8.0\n"

    # Initialize the evaluator
    anthropic_evaluator = AnthropicEvaluator(
        model="claude-1", 
        temperature=0.5, 
        grading_prompt_tempate=PromptTemplate("{input}"), 
        input_columns=["input"], 
        output_parameters=[ParameterDef(name="addition_result", type=ParameterType.NUMBER, display_name="addition result")]
    )

    # Test the grade_row method
    row_input = RowInput(input=3)
    result = anthropic_evaluator.grade_row(row_input)
    
    # Assert the result
    assert result.is_successful
    assert result.addition_result == 8.0

def test_retry_policy():
    retry_policy = RetryPolicy(3, 3)
    assert retry_policy.max_retry_on_invalid_result == 3
    assert retry_policy.max_retry_on_exception == 3

def test_default_retry_policy():
    default_retry_policy = DefaultRetryPolicy()
    assert default_retry_policy.max_retry_on_invalid_result == 3
    assert default_retry_policy.max_retry_on_exception == 3

def test_no_retry_policy():
    no_retry_policy = NoRetryPolicy()
    assert no_retry_policy.max_retry_on_invalid_result == 0
    assert no_retry_policy.max_retry_on_exception == 0

@mock.patch("databricks.doc_qa.evaluators.templated_evaluator.openai_provider.request_openai")
def test_openai_evaluator(mock_request_openai):
    # Define a mock response from the openai_provider
    mock_response = {
        "function_call": {
            "arguments": '{"score": 3.5}'
        }
    }
    mock_request_openai.return_value = mock_response

    # Define the OpenAIEvaluator instance
    openai_evaluator = OpenAIEvaluator(
        model="gpt-4",
        temperature=0.5,
        grading_prompt_tempate=PromptTemplate("Grade this: {text}", ["text"]),
        input_columns=["text"],
        openai_function={"name": "grading_function", "parameters": {"properties": {"score": {"type": "number"}}}}
    )

    # Define a RowInput
    row_input = RowInput(text="This is a test.")

    # Test the grade_row method
    result = openai_evaluator.grade_row(row_input)
    assert isinstance(result, RowEvalResult)
    assert result.is_successful
    assert result.score == 3.5


@mock.patch("databricks.doc_qa.evaluators.templated_evaluator.anthropic_provider.request_anthropic")
def test_anthropic_evaluator(mock_request_anthropic):
    # Define a mock response from the anthropic_provider
    mock_response = "Score: 3.5\n"
    mock_request_anthropic.return_value = mock_response

    # Define the AnthropicEvaluator instance
    anthropic_evaluator = AnthropicEvaluator(
        model="claude-2",
        temperature=0.5,
        grading_prompt_tempate=PromptTemplate("Grade this: {text}", ["text"]),
        input_columns=["text"],
        output_parameters=[ParameterDef(name="score", type="number")]
    )

    # Define a RowInput
    row_input = RowInput(text="This is a test.")

    # Test the grade_row method
    result = anthropic_evaluator.grade_row(row_input)
    assert isinstance(result, RowEvalResult)
    assert result.is_successful
    assert result.score == 3.5

def test_parameter_def():
    param_def = ParameterDef(name="count", type="number")
    assert param_def.name == "count"
    assert param_def.type == ParameterType.NUMBER
    assert param_def.display_name == "Count"
    assert param_def.extract("Count: 5\n") == 5.0
    assert param_def.extract("Count: abc\n") == None


def test_base_llm_evaluator_extract_parameters():
    function_json = {
        "parameters": {
            "properties": {
                "result": {
                    "type": "number",
                    "description": "This is a number parameter"
                },
                "comment": {
                    "type": "string",
                    "description": "This is a string parameter"
                }
            }
        }
    }

    evaluator = create_sample_evaluator()
    parameter_defs = evaluator.extract_parameters(function_json)
    
    assert len(parameter_defs) == 2
    assert parameter_defs[0].name == "result"
    assert parameter_defs[0].type == ParameterType.NUMBER
    assert parameter_defs[0].description == "This is a number parameter"
    assert parameter_defs[1].name == "comment"
    assert parameter_defs[1].type == ParameterType.STRING
    assert parameter_defs[1].description == "This is a string parameter"

def test_base_llm_evaluator_init_exceptions():
    grading_prompt_template = PromptTemplate(
        template_str="This is a template with {var1} and {var2}.",
        variables=['var1', 'var2']
    )

    with pytest.raises(ValueError):
        BaseLlmEvaluator(
            grading_prompt_tempate=grading_prompt_template,
            input_columns=['var1', 'var2'],
            model="model_name",
            temperature=0.8,
            # passing both openai_function and output_parameters
            openai_function={"name": "func"},
            output_parameters=[ParameterType.STRING]
        )

    with pytest.raises(ValueError):
        BaseLlmEvaluator(
            grading_prompt_tempate=grading_prompt_template,
            input_columns=['var1', 'var2'],
            model="model_name",
            temperature=0.8,
            # not passing openai_function and output_parameters
        )

    with pytest.raises(ValueError):
        BaseLlmEvaluator(
            grading_prompt_tempate=grading_prompt_template,
            input_columns=['var3'],  # var3 is not in the template
            model="model_name",
            temperature=0.8,
            output_parameters=[ParameterType.STRING]
        )

def test_eval_result():
    rows = [RowEvalResult(is_successful=True, error_msg=None, score=3.5), RowEvalResult(is_successful=True, error_msg=None, score=4.5)]
    eval_result = EvalResult(num_rows=2, num_successful_rows=2, rows=rows)
    df = eval_result.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "score" in df.columns

def test_eval_result_no_rows():
    eval_result = EvalResult(num_rows=0, num_successful_rows=0, rows=[])
    df = eval_result.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert eval_result.summary() == "num_rows: 0\nnum_successful_rows: 0\n"

def test_parameter_def_string_type():
    param_def = ParameterDef(name="comment", type="string")
    assert param_def.name == "comment"
    assert param_def.type == ParameterType.STRING
    assert param_def.display_name == "Comment"
    assert param_def.extract("Comment: Hello world\n") == "Hello world"
    assert param_def.extract("Comment: 123\n") == "123"