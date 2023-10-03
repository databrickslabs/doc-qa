from databricks.labs.doc_qa.llm_utils import PromptTemplate

grading_prompt_template = PromptTemplate(
    """ Please act as an impartial judge and evaluate the quality of the provided answer which attempts to answer the provided question based on a provided context.

You'll be given a function grading_function which you'll call for each provided context, question and answer to submit your reasoning and score for the context_relevance which measures the relevance of the context to the question, and correctness which measures the correctness of the answer in regard to the question.  
Please make sure you always call the function to submit result

Below is your grading rubric: 

- context_relevance: If the provided context is relevant to the question, below are the details of the different scores:
- Score 0: the context is completely irrelevant to the question
- Score 1: the context is partially relevant to the question, but is missing one or two critical pieces of information
- Score 2: the context is relevant to the question, and doesn't miss any critical pieces of information
- correctness: how correct is the answer in regard to the question, below are the details of the different scores:
- Score 0: the answer is completely incorrect, e.g. the answer is about a different question, or the answer doesn't mention anything about the question or the answer doesn't make sense at all
- Score 1: if the answer only partially answer the question, e.g. the answer only answer one aspect of the question or is missing one or two critical pieces of information
- Score 2: the answer is correct and doesn't miss any critical pieces of information

Requirements:
- If the context is "empty" or empty string, then the context_relevance score should be 0
- If the answer is "empty" or empty string, then the correctness score should be 0

Provided question:
{question}

Provided answer: 
{answer}

Provided context: 
{context}
"""
)

openai_evaluator_function = {
    "name": "grading_function",
    "description": "Call this function to submit the grading for the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "reasoning_for_context_relevance": {
                "type": "string",
                "description": "Your reasoning for giving the grading for the context relevance of the provided context in regards to the question. Provide 5 to 30 words explanation.",
            },
            "context_relevance": {
                "type": "integer",
                "description": "Your integer grading between 0 to 2 for the context relevance of the answer.",
            },
            "reasoning_for_correctness": {
                "type": "string",
                "description": "Your reasoning for giving the grading for the correctness of the answer. Provide 5 to 30 words explanation.",
            },
            "correctness": {
                "type": "integer",
                "description": "Your integer grading between 0 to 2 for the correctness of the answer.",
            },
        },
        "required": [
            "reasoning_for_context_relevance",
            "context_relevance",
            "reasoning_for_correctness",
            "correctness",
        ],
    },
}



