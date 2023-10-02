import json
import time
import os
import requests
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_fixed, retry_if_exception_type, retry_if_exception
from databricks.labs.doc_qa.logging_utils import logger
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import logging

anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

def supress_httpx_logs():
    # Get the logger for 'httpx'
    logger = logging.getLogger('httpx')

    # Set the log level to warning
    logger.setLevel(logging.WARNING)

supress_httpx_logs()


def request_anthropic(prompt, temperature=0.0, model="claude-2", max_tokens_to_sample=300):
    logger.debug(f"Calling anthropic API with model {model} using prompt: {prompt}")
    anthropic = Anthropic(
        api_key=anthropic_api_key,
    )

    completion = anthropic.completions.create(
        model=model,
        max_tokens_to_sample=max_tokens_to_sample,
        temperature=temperature,
        prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
    )
    return completion.completion
