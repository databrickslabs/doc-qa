import json
import time
import os
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    stop_after_delay,
    wait_fixed,
    retry_if_exception_type,
    retry_if_exception,
)
from databricks.labs.doc_qa.logging_utils import logger


openai_token = os.getenv("OPENAI_API_KEY")


class StatusCode429Error(Exception):
    pass


def request_openai(
    messages, functions=[], temperature=0.0, model="gpt-4", retry_timeout=None
):
    if retry_timeout is None:
        retry_timeout = 300

    @retry(
        stop=stop_after_delay(retry_timeout),
        wait=wait_fixed(3),
        retry=retry_if_exception_type(StatusCode429Error),
        reraise=True,
    )
    @retry(
        stop=stop_after_attempt(3),
        retry=retry_if_exception(lambda ex: not isinstance(ex, StatusCode429Error)),
        reraise=True,
    )
    def make_request():
        logger.debug(
            f"Calling open-ai API with {len(messages)} messages and {len(functions)} functions and model {model}"
        )

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_token}",
        }
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if len(functions) > 0:
            data["functions"] = functions
        logger.debug(f"Calling open-ai API with data: {data}")
        response = requests.post(
            url, headers=headers, data=json.dumps(data), timeout=60
        )
        if response.status_code == 429:
            logger.debug(f"Got 429 status code from openAI, response: {response.text}")
            raise StatusCode429Error("Too many requests")
        elif response.status_code != 200:
            raise Exception(
                f"Request failed with status code {response.status_code} and response {response.text}"
            )

        response_json = response.json()

        completion = response_json
        response_message = completion["choices"][0]["message"]
        return response_message

    return make_request()
