import pytest
from databricks.labs.doc_qa.variables.doc_qa_template_variables import (
    get_openai_grading_template_and_function,
    openai_evaluator_function_scale_1,
    openai_evaluator_function_scale_3,
    openai_evaluator_function_scale_4,
    openai_evaluator_function_scale_10,
)
from databricks.labs.doc_qa.llm_utils import PromptTemplate


def test_get_openai_grading_template_and_function():
    # Test for valid scale and level_of_details
    for scale in [1, 3, 4, 10]:
        for level_of_details in [0, 1, 2]:
            template, function = get_openai_grading_template_and_function(
                scale, level_of_details
            )
            assert template is not None
            assert function is not None
            assert isinstance(template, PromptTemplate)
            assert isinstance(function, dict)
            assert "name" in function
            assert function["name"] == "grading_function"

            if scale == 1:
                assert function == openai_evaluator_function_scale_1
            elif scale == 3:
                assert function == openai_evaluator_function_scale_3
            elif scale == 4:
                assert function == openai_evaluator_function_scale_4
            elif scale == 10:
                assert function == openai_evaluator_function_scale_10

    # Test for invalid scale
    with pytest.raises(ValueError):
        get_openai_grading_template_and_function(5, 1)

    # Test for invalid level_of_details
    with pytest.raises(ValueError):
        get_openai_grading_template_and_function(1, 3)


def test_prompt_templates():
    # Test that the prompt templates are correctly formed
    for scale in [1, 3, 4, 10]:
        for level_of_details in [0, 1, 2]:
            prompt_template, _ = get_openai_grading_template_and_function(
                scale, level_of_details
            )

            # Replace placeholders with dummy data
            filled_template = prompt_template.format(
                question="dummy_question",
                answer="dummy_answer",
                context="dummy_context",
            )

            # Check that the dummy data appears in the filled template
            assert "dummy_question" in filled_template
            assert "dummy_answer" in filled_template
            assert "dummy_context" in filled_template
