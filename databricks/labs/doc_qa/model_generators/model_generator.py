from databricks.labs.doc_qa.llm_utils import PromptTemplate
from databricks.labs.doc_qa.llm_providers import openai_provider
import pandas as pd
import logging
import concurrent.futures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RowGenerateResult:
    """
    A RowEvalResult object contains the evaluation result for a single row in the evaluation dataframe.
    """

    def __init__(self, is_successful, error_msg, **kwargs):
        self.is_successful = is_successful
        self.error_msg = error_msg
        for key, value in kwargs.items():
            setattr(self, key, value)


class BatchGenerateResult:
    num_rows: int
    num_successful_rows: int
    rows: list
    """
    A RowEvalResult object contains the evaluation result for a single row in the evaluation dataframe.
    """

    def __init__(self, is_successful, error_msg, **kwargs):
        self.is_successful = is_successful
        self.error_msg = error_msg
        for key, value in kwargs.items():
            setattr(self, key, value)

class GenerateResult:
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



class BaseModelGenerator:

    def __init__(self, prompt_formatter: PromptTemplate, batch_size: int=1, concurrency=1) -> None:
        """
        Args:
            prompt_formatter (PromptTemplate): the prompt format to format the input dataframe into prompts row by row according to the column names
            batch_size (int, optional): Batch size that will be used to run tasks. Defaults to 1, which means it's sequential.
            concurrency (int, optional): concurrency of the tasks. Defaults to 1.
        """
        self._prompt_formatter = prompt_formatter
        self._batch_size = batch_size
        self._concurrency = concurrency

    def _generate(self, prompts: list, temperature: float, max_tokens=256, system_prompt=None) -> BatchGenerateResult:
        raise NotImplementedError
    

    def run_tasks(self, input_df, temperature: float, max_tokens=256, system_prompt=None) -> GenerateResult:
        """
        Run the model on the input dataframe.
        Args:
            input_df (pd.DataFrame): the input dataframe
            concurrency (int, optional): concurrency of the tasks. Defaults to 1.
        Returns:
            EvalResult: the evaluation result
        """
        prompt_batches = []
        # First, traverse the input dataframe using batch size
        for i in range(0, len(input_df), self._batch_size):
            # Get the current batch
            batch_df = input_df.iloc[i:i+self._batch_size]
    
            # Format the input dataframe into prompts row by row
            prompts = []
            for index, row in batch_df.iterrows():
                # Format the input dataframe into prompts
                prompt = self._prompt_formatter.format(**row)
                prompts.append(prompt)
            prompt_batches.append(prompts)
        logger.info(f"Generated total number of batches for prompts: {len(prompt_batches)}")

        # Call the _generate in parallel using multiple threads, each call with a batch of prompts
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._concurrency) as executor:
            future_to_batch = {executor.submit(self._generate, prompts, temperature, max_tokens, system_prompt): prompts for prompts in prompt_batches}
            batch_generate_results = []
            for future in concurrent.futures.as_completed(future_to_batch):
                prompts = future_to_batch[future]
                try:
                    result = future.result()
                    batch_generate_results.append(result)
                except Exception as exc:
                    logger.error(f"Exception occurred when running the task: {exc}")
                    # generate the same amount of RowEvalResult as the number of rows in the batch, with is_successful=False and error_msg=exc
                    rows = [RowGenerateResult(is_successful=False, error_msg=str(exc)) for _ in range(len(prompts))]
                    # append a failed result with the error message
                    batch_generate_results.append(BatchGenerateResult(num_rows=len(prompts), num_successful_rows=0, rows=rows, is_successful=False, error_msg=str(exc)))
                    raise exc
        logger.info(f"Generated total number of results: {len(batch_generate_results)}")
        # Translate batch generate results to a single generate result
        num_rows = 0
        num_successful_rows = 0
        rows = []
        for batch_generate_result in batch_generate_results:
            num_rows += batch_generate_result.num_rows
            num_successful_rows += batch_generate_result.num_successful_rows
            rows.extend(batch_generate_result.rows)
        generate_result = GenerateResult(num_rows, num_successful_rows, rows)
        return generate_result


class OpenAiModelGenerator(BaseModelGenerator):
    ALLOWED_MODEL_NAMES = ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]

    def __init__(self, prompt_formatter: PromptTemplate, model_name: str, batch_size: int=1, concurrency: int=1) -> None:
        """
        Args:
            prompt_formatter (PromptTemplate): the prompt format to format the input dataframe into prompts row by row according to the column names
            model_name (str): the model name
            batch_size (int, optional): Batch size that will be used to run tasks. Defaults to 1, which means it's sequential.
        
        """
        super().__init__(prompt_formatter, batch_size, concurrency)
        # require the batch size to be 1
        if batch_size != 1:
            raise ValueError("OpenAiModelGenerator currently only supports batch size 1")
        if model_name not in self.ALLOWED_MODEL_NAMES:
            raise ValueError(f"model_name {model_name} is not supported. Supported model names: {self.ALLOWED_MODEL_NAMES}")
        self._model_name = model_name


    def _generate(self, prompts: list, temperature: float, max_tokens=256, system_prompt=None) -> BatchGenerateResult:
        if system_prompt is not None:
            messages = [
                {"role": "system", "content": system_prompt},
            ]
        else:
            messages = []

        # we can assume the prompts list has only one element
        user_prompt = prompts[0]
        messages.append({"role": "user", "content": user_prompt})

        response_message = openai_provider.request_openai(messages=messages, functions=[], model=self._model_name, temperature=temperature)
        content = response_message['content']
        logger.debug(f"Got response content: {content}")
        row_generate_result = RowGenerateResult(is_successful=True, error_msg=None, content=content, temperature=temperature, max_tokens=max_tokens, model_name=self._model_name)
        return BatchGenerateResult(num_rows=1, num_successful_rows=1, rows=[row_generate_result], is_successful=True, error_msg=None)