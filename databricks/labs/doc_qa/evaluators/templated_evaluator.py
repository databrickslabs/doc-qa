from dataclasses import dataclass
from typing import Any
from pandas import DataFrame
import concurrent.futures
import pandas as pd
from databricks.labs.doc_qa.llm_utils import PromptTemplate
from databricks.labs.doc_qa.llm_providers import openai_provider
from databricks.labs.doc_qa.llm_providers import anthropic_provider
import json
from databricks.labs.doc_qa.logging_utils import logger
from tenacity import retry, stop_after_attempt, retry_if_result, retry_if_exception
import re
from enum import Enum
import json


class ParameterType(Enum):
    STRING = "string"
    NUMBER = "number"


class ParameterDef:
    def __init__(
        self, name, type, display_name=None, description=None, regex_rule=None
    ):
        self.name = name
        self.type = ParameterType(type) if isinstance(type, str) else type
        self.display_name = display_name or self.default_display_name()
        self.description = description
        self.regex_rule = (
            regex_rule or self.default_regex()
        )  # use default if none provided

    def __repr__(self):
        return f"ParameterDef(name={self.name}, type={self.type}, description={self.description})"

    def default_display_name(self):
        """Converts parameter name to human-readable format."""
        words = self.name.split("_")
        return " ".join([word.capitalize() for word in words])

    def default_regex(self):
        """Generates a default regex rule based on display name."""
        # Making it case-insensitive, accommodating spaces, and assuming that the parameter value comes after ':'.
        pattern_name = re.escape(self.display_name)
        pattern_name = pattern_name.replace(r"\ ", r"\s+")
        return f"(?i){pattern_name}\s*:\s*(.*?)\n"

    def extract(self, text):
        """Extracts the value from the text based on the regex rule."""
        match = re.search(self.regex_rule, text)
        if not match:
            return None

        value = match.group(1).strip()
        if self.type == ParameterType.NUMBER:
            try:
                return float(value)
            except ValueError:
                return None
        return value


class RetryPolicy:
    def __init__(self, max_retry_on_invalid_result: int, max_retry_on_exception: int):
        self.max_retry_on_invalid_result = max_retry_on_invalid_result
        self.max_retry_on_exception = max_retry_on_exception


class DefaultRetryPolicy(RetryPolicy):
    def __init__(self):
        super().__init__(max_retry_on_invalid_result=3, max_retry_on_exception=3)


class NoRetryPolicy(RetryPolicy):
    def __init__(self):
        super().__init__(max_retry_on_invalid_result=0, max_retry_on_exception=0)


class RowInput:
    """
    A RowInput object contains the input data for a single row in the evaluation dataframe.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class RowEvalResult:
    """
    A RowEvalResult object contains the evaluation result for a single row in the evaluation dataframe.
    """

    def __init__(self, is_successful, error_msg, **kwargs):
        self.is_successful = is_successful
        self.error_msg = error_msg
        for key, value in kwargs.items():
            setattr(self, key, value)


class EvalResult:
    num_rows: int
    num_successful_rows: int
    rows: list

    def __init__(self, num_rows, num_successful_rows, rows, **kwargs):
        self.num_rows = num_rows
        self.num_successful_rows = num_successful_rows
        self.rows = rows
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dataframe(self):
        # Convert the rows to a dataframe
        row_dicts = [row.__dict__ for row in self.rows]
        eval_result_df = pd.DataFrame(row_dicts)
        return eval_result_df

    def summary(self):
        summary_str = ""
        # Traverse the kwargs of the EvalResult object
        for key, value in self.__dict__.items():
            # Skip the rows attribute
            if key == "rows":
                continue
            # Add the key-value pair to the string
            summary_str += f"{key}: {value}\n"
        return summary_str


# Define the base evaluator class
class BaseLlmEvaluator:
    def __init__(
        self,
        grading_prompt_tempate: PromptTemplate,
        input_columns: list,
        model: str,
        temperature: float,
        openai_function: dict = None,
        output_parameters: list = None,
        system_prompt_template: PromptTemplate = None,
        retry_policy: RetryPolicy = DefaultRetryPolicy(),
    ):
        # either openai function or output_extract_regex_dict should be provided
        if openai_function is None and output_parameters is None:
            raise ValueError(
                "Either openai_function or output_parameters should be provided."
            )
        # if output_parameters is provided, then openai_function should not be provided
        if openai_function is not None and output_parameters is not None:
            raise ValueError(
                "Only one of openai_function and output_parameters should be provided."
            )
        if openai_function is not None:
            self.openai_function = openai_function
            self.output_parameters = self.extract_parameters(self.openai_function)
        else:
            self.output_parameters = output_parameters
        self.grading_prompt_tempate = grading_prompt_tempate
        self.system_prompt_template = system_prompt_template
        # Check all input_columns have been included in grading_prompt_tempate's variables
        for input_column in input_columns:
            if input_column not in self.grading_prompt_tempate.variables:
                raise ValueError(
                    f"Input column '{input_column}' not found in grading prompt template variables."
                )
        self.input_columns = input_columns
        self.model = model
        self.temperature = temperature
        self.retry_policy = retry_policy

    def extract_parameters(self, function_json):
        parameters = function_json.get("parameters", {}).get("properties", {})
        parameter_defs = []

        for name, parameter in parameters.items():
            if parameter.get("type") in ["number", "integer", "float"]:
                parameter_type = ParameterType.NUMBER
            else:
                parameter_type = ParameterType.STRING
            description = parameter.get("description")
            parameter_defs.append(
                ParameterDef(name=name, type=parameter_type, description=description)
            )

        return parameter_defs

    def grade_row_retry_wrapper(self, row_input: RowInput) -> RowEvalResult:
        # Apply the retry policy for invalid result
        _retry_policy_invalid = retry(
            stop=stop_after_attempt(self.retry_policy.max_retry_on_invalid_result),
            retry=retry_if_result(lambda result: result is None),
            reraise=True,
        )

        # Apply the retry policy for exception
        _retry_policy_exception = retry(
            stop=stop_after_attempt(self.retry_policy.max_retry_on_exception),
            retry=retry_if_exception(type(Exception)),
            reraise=True,
        )
        return _retry_policy_exception(_retry_policy_invalid(self.grade_row))(row_input)

    def grade_row(self, row_input: RowInput) -> RowEvalResult:
        # This method should be implemented in the subclasses
        raise NotImplementedError

    def run_eval(
        self, concurrency: int = 10, dataset_df: DataFrame = None, catch_error=True
    ) -> EvalResult:
        rows = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_row = {
                executor.submit(self.grade_row_retry_wrapper, RowInput(**row)): row
                for index, row in dataset_df.iterrows()
            }
            for future in concurrent.futures.as_completed(future_to_row):
                # Include the attribute and value for these variables in the result: [is_successful, error_msg] + output_parameters names + input_columns
                expected_attributes = (
                    ["is_successful", "error_msg"]
                    + [parameter.name for parameter in self.output_parameters]
                    + self.input_columns
                )
                original_row = future_to_row[future]
                try:
                    result = future.result()
                    # filter for the expected attributes
                    result_dict = {
                        key: value
                        for key, value in result.__dict__.items()
                        if key in expected_attributes
                    }
                    # add the original row to the result, only on the input columns
                    result_dict.update(
                        {
                            key: value
                            for key, value in original_row.items()
                            if key in self.input_columns
                        }
                    )
                    logger.debug(f"Base Evaluator got result {result_dict} ")
                    # Translate into RowEvalResult
                    row_eval_result = RowEvalResult(**result_dict)
                    rows.append(row_eval_result)
                except Exception as exc:
                    if catch_error:
                        logger.warn(
                            f"Encountered error while processing row, error: {exc}, original row {original_row}"
                        )
                        # print stack trace into the logger debug
                        logger.warn(f"Traceback: {exc.__traceback__}")
                        # Give is_successful=False and error_msg=str(exc) to the row, and None to the other attributes
                        row_eval_result = RowEvalResult(
                            is_successful=False,
                            error_msg=str(exc),
                            **{
                                key: None
                                for key in expected_attributes
                                if key not in ["is_successful", "error_msg"]
                            },
                        )
                        rows.append(row_eval_result)
                    else:
                        raise exc
        num_rows = len(rows)
        if num_rows == 0:
            return EvalResult(num_rows=0, num_successful_rows=0, rows=[])
        num_successful_rows = len([row for row in rows if row.is_successful])
        eval_result = EvalResult(
            num_rows=num_rows,
            num_successful_rows=num_successful_rows,
            rows=rows,
        )

        # Calculate the average value for all the output parameters whose type is ParameterType.NUMBER, and give it attribute name avg_{parameter name} for the EvalResult
        for parameter in self.output_parameters:
            if parameter.type == ParameterType.NUMBER:
                # Get the values for this parameter
                parameter_values = [
                    row.__dict__[parameter.name] for row in rows if row.is_successful
                ]
                # Calculate the average value
                if len(parameter_values) == 0:
                    avg_value = 0
                else:
                    avg_value = sum(parameter_values) / len(parameter_values)
                # Add the average value as an attribute to the EvalResult
                setattr(eval_result, f"avg_{parameter.name}", avg_value)
        return eval_result


class OpenAIEvaluator(BaseLlmEvaluator):
    ALLOWED_MODEL_NAMES = ["gpt-4", "gpt-3.5-turbo-16k"]

    # Override the constructor to only allow models from gpt-4 and gpt-3.5-turbo-16k
    def __init__(
        self,
        model: str,
        temperature: float,
        grading_prompt_tempate: PromptTemplate,
        input_columns: list,
        openai_function: dict = None,
        system_prompt_template: PromptTemplate = None,
        retry_policy: RetryPolicy = DefaultRetryPolicy(),
        openai_retry_timeout: int = 1200,
    ):
        if model not in self.ALLOWED_MODEL_NAMES:
            raise ValueError(
                f"Unsupported model {model} provided. Only gpt-4 and gpt-3.5-turbo-16k are supported."
            )
        self._openai_retry_timeout = openai_retry_timeout
        super().__init__(
            model=model,
            temperature=temperature,
            grading_prompt_tempate=grading_prompt_tempate,
            input_columns=input_columns,
            openai_function=openai_function,
            system_prompt_template=system_prompt_template,
            retry_policy=retry_policy,
        )

    def grade_row(self, row_input: RowInput) -> RowEvalResult:
        if self.system_prompt_template is not None:
            system_prompt = self.system_prompt_template.format(
                **{key: getattr(row_input, key) for key in self.input_columns}
            )
            messages = [
                {"role": "system", "content": system_prompt},
            ]
        else:
            messages = []

        user_prompt = self.grading_prompt_tempate.format(
            **{key: getattr(row_input, key) for key in self.input_columns}
        )
        messages.append({"role": "user", "content": user_prompt})

        # For gpt-3.5-turbo-16k, putting everything into the system prompt will get it to not call function at around 50% chances
        functions = [self.openai_function]
        response_message = openai_provider.request_openai(
            messages=messages,
            functions=functions,
            model=self.model,
            temperature=self.temperature,
            retry_timeout=self._openai_retry_timeout,
        )
        if "function_call" in response_message:
            function_call_obj = response_message["function_call"]
            arguments = json.loads(function_call_obj["arguments"])
            # extract the output values from the arguments according to the output columns
            output_values = [
                arguments[output_parameter.name]
                for output_parameter in self.output_parameters
            ]
            row_eval_result = RowEvalResult(
                is_successful=True,
                error_msg=None,
                **{
                    output_parameter.name: output_value
                    for output_parameter, output_value in zip(
                        self.output_parameters, output_values
                    )
                },
            )
            logger.debug(f"Successfully got result {row_eval_result} ")
            return row_eval_result
        else:
            logger.warning(
                f"Retrying, OpenAI doesn't call function for row {row_input}, response_message: {response_message}"
            )
            return None


class AnthropicEvaluator(BaseLlmEvaluator):
    ALLOWED_MODEL_NAMES = ["claude-1", "claude-2"]

    # Override the constructor to only allow models from claude-1 and claude-2
    def __init__(
        self,
        model: str,
        temperature: float,
        grading_prompt_tempate: PromptTemplate,
        input_columns: list,
        output_parameters: list = None,
        retry_policy: RetryPolicy = DefaultRetryPolicy(),
    ):
        if model not in self.ALLOWED_MODEL_NAMES:
            raise ValueError(
                f"Unsupported model {model} provided. Only claude-1 and claude-2 are supported."
            )
        super().__init__(
            model=model,
            temperature=temperature,
            grading_prompt_tempate=grading_prompt_tempate,
            input_columns=input_columns,
            output_parameters=output_parameters,
            retry_policy=retry_policy,
        )

    def grade_row(self, row_input: RowInput) -> RowEvalResult:
        user_prompt = self.grading_prompt_tempate.format(
            **{key: getattr(row_input, key) for key in self.input_columns}
        )
        response_message = anthropic_provider.request_anthropic(
            prompt=user_prompt, temperature=self.temperature, model=self.model
        )
        # Add a new line to the response_message at the end
        response_message += "\n"
        logger.debug(f"Got response message {response_message}")
        extracted_data = {}
        for param_def in self.output_parameters:
            extracted_data[param_def.name] = param_def.extract(response_message)

        # Check if all the output parameters have been extracted
        if None in extracted_data.values():
            attributes_missing_values = [
                key for key, value in extracted_data.items() if value is None
            ]
            logger.warning(
                f"We are missing value for some output parameters {attributes_missing_values}"
            )
            return None
        else:
            # If yes, return is_successful=True and error_msg=None
            return RowEvalResult(is_successful=True, error_msg=None, **extracted_data)
