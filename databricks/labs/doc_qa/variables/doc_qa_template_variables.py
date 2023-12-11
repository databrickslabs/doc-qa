from databricks.labs.doc_qa.llm_utils import PromptTemplate


grading_system_prompt_intro = """ Please act as an impartial judge and evaluate the quality of the provided answer which attempts to answer the provided question based on a provided context."""

openai_submission_instruction = """You'll be given a function grading_function which you'll call for each provided context, question and answer to submit your reasoning and score for the correctness, comprehensiveness and readability of the answer.  
Please make sure you always call the function to submit result"""

grading_instruction_scale_10 = """
  Below is your grading rubric: 

- Correctness: If the answer correctly answer the question, below are the details for different scores:
  - Score 0: the answer is completely incorrect, doesn’t mention anything about the question or is completely contrary to the correct answer.
      - For example, when asked “How to terminate a databricks cluster”, the answer is empty string, or content that’s completely irrelevant, or sorry I don’t know the answer.
  - Score 4: the answer provides some relevance to the question and answer one aspect of the question correctly.
      - Example:
          - Question: How to terminate a databricks cluster
          - Answer: Databricks cluster is a cloud-based computing environment that allows users to process big data and run distributed data processing tasks efficiently.
          - Or answer:  In the Databricks workspace, navigate to the "Clusters" tab. And then this is a hard question that I need to think more about it
  - Score 7: the answer mostly answer the question but is missing or hallucinating on one critical aspect.
      - Example:
          - Question: How to terminate a databricks cluster”
          - Answer: “In the Databricks workspace, navigate to the "Clusters" tab.
          Find the cluster you want to terminate from the list of active clusters.
          And then you’ll find a button to terminate all clusters at once”
  - Score 10: the answer correctly answer the question and not missing any major aspect
      - Example:
          - Question: How to terminate a databricks cluster
          - Answer: In the Databricks workspace, navigate to the "Clusters" tab.
          Find the cluster you want to terminate from the list of active clusters.
          Click on the down-arrow next to the cluster name to open the cluster details.
          Click on the "Terminate" button. A confirmation dialog will appear. Click "Terminate" again to confirm the action.”
- Comprehensiveness: How comprehensive is the answer, does it fully answer all aspects of the question and provide comprehensive explanation and other necessary information. Below are the details for different scores:
  - Score 0: typically if the answer is completely incorrect, then the comprehensiveness is also zero score.
  - Score 3: if the answer is correct but too short to fully answer the question, then we can give score 1 for comprehensiveness.
      - Example:
          - Question: How to use databricks API to create a cluster?
          - Answer: First, you will need a Databricks access token with the appropriate permissions. You can generate this token through the Databricks UI under the 'User Settings' option. And then (the rest is missing)
  - Score 7: the answer is correct and roughly answer the main aspects of the question, but it’s missing description about details. Or is completely missing details about one minor aspect.
      - Example:
          - Question: How to use databricks API to create a cluster?
          - Answer: You will need a Databricks access token with the appropriate permissions. Then you’ll need to set up the request URL, then you can make the HTTP Request. Then you can handle the request response.
      - Example:
          - Question: How to use databricks API to create a cluster?
          - Answer: You will need a Databricks access token with the appropriate permissions. Then you’ll need to set up the request URL, then you can make the HTTP Request. Then you can handle the request response.
  - Score 10: the answer is correct, and covers all the main aspects of the question
- Readability: How readable is the answer, does it have redundant information or incomplete information that hurts the readability of the answer.
  - Score 0: the answer is completely unreadable, e.g. fully of symbols that’s hard to read; e.g. keeps repeating the words that it’s very hard to understand the meaning of the paragraph. No meaningful information can be extracted from the answer.
  - Score 3: the answer is slightly readable, there are irrelevant symbols or repeated words, but it can roughly form a meaningful sentence that cover some aspects of the answer.
      - Example:
          - Question: How to use databricks API to create a cluster?
          - Answer: You you  you  you  you  you  will need a Databricks access token with the appropriate permissions. And then then you’ll need to set up the request URL, then you can make the HTTP Request. Then Then Then Then Then Then Then Then Then
  - Score 7: the answer is correct and mostly readable, but there is one obvious piece that’s affecting the readability (mentioning of irrelevant pieces, repeated words)
      - Example:
          - Question: How to terminate a databricks cluster
          - Answer: In the Databricks workspace, navigate to the "Clusters" tab.
          Find the cluster you want to terminate from the list of active clusters.
          Click on the down-arrow next to the cluster name to open the cluster details.
          Click on the "Terminate" button…………………………………..
          A confirmation dialog will appear. Click "Terminate" again to confirm the action.
  - Score 10: the answer is correct and reader friendly, no obvious piece that affect readability.

"""

grading_instruction_scale_10_no_example = """
  Below is your grading rubric: 

- Correctness: If the answer correctly answer the question, below are the details for different scores:
  - Score 0: the answer is completely incorrect, doesn’t mention anything about the question or is completely contrary to the correct answer.
  - Score 4: the answer provides some relevance to the question and answer one aspect of the question correctly.
  - Score 7: the answer mostly answer the question but is missing or hallucinating on one critical aspect.
  - Score 10: the answer correctly answer the question and not missing any major aspect
- Comprehensiveness: How comprehensive is the answer, does it fully answer all aspects of the question and provide comprehensive explanation and other necessary information. Below are the details for different scores:
  - Score 0: typically if the answer is completely incorrect, then the comprehensiveness is also zero score.
  - Score 3: if the answer is correct but too short to fully answer the question, then we can give score 1 for comprehensiveness.
  - Score 7: the answer is correct and roughly answer the main aspects of the question, but it’s missing description about details. Or is completely missing details about one minor aspect.
  - Score 10: the answer is correct, and covers all the main aspects of the question
- Readability: How readable is the answer, does it have redundant information or incomplete information that hurts the readability of the answer.
  - Score 0: the answer is completely unreadable, e.g. fully of symbols that’s hard to read; e.g. keeps repeating the words that it’s very hard to understand the meaning of the paragraph. No meaningful information can be extracted from the answer.
  - Score 3: the answer is slightly readable, there are irrelevant symbols or repeated words, but it can roughly form a meaningful sentence that cover some aspects of the answer.
  - Score 7: the answer is correct and mostly readable, but there is one obvious piece that’s affecting the readability (mentioning of irrelevant pieces, repeated words)
  - Score 10: the answer is correct and reader friendly, no obvious piece that affect readability.

"""

grading_instruction_no_detail = """
  Below is your grading rubric: 

- Correctness: If the answer correctly answer the question
- Comprehensiveness: How comprehensive is the answer, does it fully answer all aspects of the question and provide comprehensive explanation and other necessary information. 
- Readability: How readable is the answer, does it have redundant information or incomplete information that hurts the readability of the answer.
"""


grading_instruction_scale_4 = """
  Below is your grading rubric: 

- Correctness: If the answer correctly answer the question, below are the details for different scores:
  - Score 0: the answer is completely incorrect, doesn’t mention anything about the question or is completely contrary to the correct answer.
      - For example, when asked “How to terminate a databricks cluster”, the answer is empty string, or content that’s completely irrelevant, or sorry I don’t know the answer.
  - Score 1: the answer provides some relevance to the question and answer one aspect of the question correctly.
      - Example:
          - Question: How to terminate a databricks cluster
          - Answer: Databricks cluster is a cloud-based computing environment that allows users to process big data and run distributed data processing tasks efficiently.
          - Or answer:  In the Databricks workspace, navigate to the "Clusters" tab. And then this is a hard question that I need to think more about it
  - Score 2: the answer mostly answer the question but is missing or hallucinating on one critical aspect.
      - Example:
          - Question: How to terminate a databricks cluster”
          - Answer: “In the Databricks workspace, navigate to the "Clusters" tab.
          Find the cluster you want to terminate from the list of active clusters.
          And then you’ll find a button to terminate all clusters at once”
  - Score 4: the answer correctly answer the question and not missing any major aspect
      - Example:
          - Question: How to terminate a databricks cluster
          - Answer: In the Databricks workspace, navigate to the "Clusters" tab.
          Find the cluster you want to terminate from the list of active clusters.
          Click on the down-arrow next to the cluster name to open the cluster details.
          Click on the "Terminate" button. A confirmation dialog will appear. Click "Terminate" again to confirm the action.”
- Comprehensiveness: How comprehensive is the answer, does it fully answer all aspects of the question and provide comprehensive explanation and other necessary information. Below are the details for different scores:
  - Score 0: typically if the answer is completely incorrect, then the comprehensiveness is also zero score.
  - Score 1: if the answer is correct but too short to fully answer the question, then we can give score 1 for comprehensiveness.
      - Example:
          - Question: How to use databricks API to create a cluster?
          - Answer: First, you will need a Databricks access token with the appropriate permissions. You can generate this token through the Databricks UI under the 'User Settings' option. And then (the rest is missing)
  - Score 3: the answer is correct and roughly answer the main aspects of the question, but it’s missing description about details. Or is completely missing details about one minor aspect.
      - Example:
          - Question: How to use databricks API to create a cluster?
          - Answer: You will need a Databricks access token with the appropriate permissions. Then you’ll need to set up the request URL, then you can make the HTTP Request. Then you can handle the request response.
      - Example:
          - Question: How to use databricks API to create a cluster?
          - Answer: You will need a Databricks access token with the appropriate permissions. Then you’ll need to set up the request URL, then you can make the HTTP Request. Then you can handle the request response.
  - Score 4: the answer is correct, and covers all the main aspects of the question
- Readability: How readable is the answer, does it have redundant information or incomplete information that hurts the readability of the answer.
  - Score 0: the answer is completely unreadable, e.g. fully of symbols that’s hard to read; e.g. keeps repeating the words that it’s very hard to understand the meaning of the paragraph. No meaningful information can be extracted from the answer.
  - Score 1: the answer is slightly readable, there are irrelevant symbols or repeated words, but it can roughly form a meaningful sentence that cover some aspects of the answer.
      - Example:
          - Question: How to use databricks API to create a cluster?
          - Answer: You you  you  you  you  you  will need a Databricks access token with the appropriate permissions. And then then you’ll need to set up the request URL, then you can make the HTTP Request. Then Then Then Then Then Then Then Then Then
  - Score 3: the answer is correct and mostly readable, but there is one obvious piece that’s affecting the readability (mentioning of irrelevant pieces, repeated words)
      - Example:
          - Question: How to terminate a databricks cluster
          - Answer: In the Databricks workspace, navigate to the "Clusters" tab.
          Find the cluster you want to terminate from the list of active clusters.
          Click on the down-arrow next to the cluster name to open the cluster details.
          Click on the "Terminate" button…………………………………..
          A confirmation dialog will appear. Click "Terminate" again to confirm the action.
  - Score 4: the answer is correct and reader friendly, no obvious piece that affect readability.

"""

grading_instruction_scale_4_no_example = """
  Below is your grading rubric: 

- Correctness: If the answer correctly answer the question, below are the details for different scores:
  - Score 0: the answer is completely incorrect, doesn’t mention anything about the question or is completely contrary to the correct answer.
  - Score 1: the answer provides some relevance to the question and answer one aspect of the question correctly.
  - Score 2: the answer mostly answer the question but is missing or hallucinating on one critical aspect.
  - Score 4: the answer correctly answer the question and not missing any major aspect
- Comprehensiveness: How comprehensive is the answer, does it fully answer all aspects of the question and provide comprehensive explanation and other necessary information. Below are the details for different scores:
  - Score 0: typically if the answer is completely incorrect, then the comprehensiveness is also zero score.
  - Score 1: if the answer is correct but too short to fully answer the question, then we can give score 1 for comprehensiveness.
  - Score 3: the answer is correct and roughly answer the main aspects of the question, but it’s missing description about details. Or is completely missing details about one minor aspect.
  - Score 4: the answer is correct, and covers all the main aspects of the question
- Readability: How readable is the answer, does it have redundant information or incomplete information that hurts the readability of the answer.
  - Score 0: the answer is completely unreadable, e.g. fully of symbols that’s hard to read; e.g. keeps repeating the words that it’s very hard to understand the meaning of the paragraph. No meaningful information can be extracted from the answer.
  - Score 1: the answer is slightly readable, there are irrelevant symbols or repeated words, but it can roughly form a meaningful sentence that cover some aspects of the answer.
  - Score 3: the answer is correct and mostly readable, but there is one obvious piece that’s affecting the readability (mentioning of irrelevant pieces, repeated words)
  - Score 4: the answer is correct and reader friendly, no obvious piece that affect readability.

"""


grading_instruction_scale_3 = """
  Below is your grading rubric: 

- Correctness: If the answer correctly answer the question, below are the details for different scores:
  - Score 0: the answer is completely incorrect, doesn’t mention anything about the question or is completely contrary to the correct answer.
      - For example, when asked “How to terminate a databricks cluster”, the answer is empty string, or content that’s completely irrelevant, or sorry I don’t know the answer.
  - Score 1: the answer provides some relevance to the question and answer one aspect of the question correctly.
      - Example:
          - Question: How to terminate a databricks cluster
          - Answer: Databricks cluster is a cloud-based computing environment that allows users to process big data and run distributed data processing tasks efficiently.
          - Or answer:  In the Databricks workspace, navigate to the "Clusters" tab. And then this is a hard question that I need to think more about it
  - Score 2: the answer mostly answer the question but is missing or hallucinating on one critical aspect.
      - Example:
          - Question: How to terminate a databricks cluster”
          - Answer: “In the Databricks workspace, navigate to the "Clusters" tab.
          Find the cluster you want to terminate from the list of active clusters.
          And then you’ll find a button to terminate all clusters at once”
  - Score 3: the answer correctly answer the question and not missing any major aspect
      - Example:
          - Question: How to terminate a databricks cluster
          - Answer: In the Databricks workspace, navigate to the "Clusters" tab.
          Find the cluster you want to terminate from the list of active clusters.
          Click on the down-arrow next to the cluster name to open the cluster details.
          Click on the "Terminate" button. A confirmation dialog will appear. Click "Terminate" again to confirm the action.”
- Comprehensiveness: How comprehensive is the answer, does it fully answer all aspects of the question and provide comprehensive explanation and other necessary information. Below are the details for different scores:
  - Score 0: typically if the answer is completely incorrect, then the comprehensiveness is also zero score.
  - Score 1: if the answer is correct but too short to fully answer the question, then we can give score 1 for comprehensiveness.
      - Example:
          - Question: How to use databricks API to create a cluster?
          - Answer: First, you will need a Databricks access token with the appropriate permissions. You can generate this token through the Databricks UI under the 'User Settings' option. And then (the rest is missing)
  - Score 2: the answer is correct and roughly answer the main aspects of the question, but it’s missing description about details. Or is completely missing details about one minor aspect.
      - Example:
          - Question: How to use databricks API to create a cluster?
          - Answer: You will need a Databricks access token with the appropriate permissions. Then you’ll need to set up the request URL, then you can make the HTTP Request. Then you can handle the request response.
      - Example:
          - Question: How to use databricks API to create a cluster?
          - Answer: You will need a Databricks access token with the appropriate permissions. Then you’ll need to set up the request URL, then you can make the HTTP Request. Then you can handle the request response.
  - Score 3: the answer is correct, and covers all the main aspects of the question
- Readability: How readable is the answer, does it have redundant information or incomplete information that hurts the readability of the answer.
  - Score 0: the answer is completely unreadable, e.g. fully of symbols that’s hard to read; e.g. keeps repeating the words that it’s very hard to understand the meaning of the paragraph. No meaningful information can be extracted from the answer.
  - Score 1: the answer is slightly readable, there are irrelevant symbols or repeated words, but it can roughly form a meaningful sentence that cover some aspects of the answer.
      - Example:
          - Question: How to use databricks API to create a cluster?
          - Answer: You you  you  you  you  you  will need a Databricks access token with the appropriate permissions. And then then you’ll need to set up the request URL, then you can make the HTTP Request. Then Then Then Then Then Then Then Then Then
  - Score 2: the answer is correct and mostly readable, but there is one obvious piece that’s affecting the readability (mentioning of irrelevant pieces, repeated words)
      - Example:
          - Question: How to terminate a databricks cluster
          - Answer: In the Databricks workspace, navigate to the "Clusters" tab.
          Find the cluster you want to terminate from the list of active clusters.
          Click on the down-arrow next to the cluster name to open the cluster details.
          Click on the "Terminate" button…………………………………..
          A confirmation dialog will appear. Click "Terminate" again to confirm the action.
  - Score 3: the answer is correct and reader friendly, no obvious piece that affect readability.

"""

grading_instruction_scale_3_no_example = """
  Below is your grading rubric: 

- Correctness: If the answer correctly answer the question, below are the details for different scores:
  - Score 0: the answer is completely incorrect, doesn’t mention anything about the question or is completely contrary to the correct answer.
  - Score 1: the answer provides some relevance to the question and answer one aspect of the question correctly.
  - Score 2: the answer mostly answer the question but is missing or hallucinating on one critical aspect.
  - Score 3: the answer correctly answer the question and not missing any major aspect
- Comprehensiveness: How comprehensive is the answer, does it fully answer all aspects of the question and provide comprehensive explanation and other necessary information. Below are the details for different scores:
  - Score 0: typically if the answer is completely incorrect, then the comprehensiveness is also zero score.
  - Score 1: if the answer is correct but too short to fully answer the question, then we can give score 1 for comprehensiveness.
  - Score 2: the answer is correct and roughly answer the main aspects of the question, but it’s missing description about details. Or is completely missing details about one minor aspect.
  - Score 3: the answer is correct, and covers all the main aspects of the question
- Readability: How readable is the answer, does it have redundant information or incomplete information that hurts the readability of the answer.
  - Score 0: the answer is completely unreadable, e.g. fully of symbols that’s hard to read; e.g. keeps repeating the words that it’s very hard to understand the meaning of the paragraph. No meaningful information can be extracted from the answer.
  - Score 1: the answer is slightly readable, there are irrelevant symbols or repeated words, but it can roughly form a meaningful sentence that cover some aspects of the answer.
  - Score 2: the answer is correct and mostly readable, but there is one obvious piece that’s affecting the readability (mentioning of irrelevant pieces, repeated words)
  - Score 3: the answer is correct and reader friendly, no obvious piece that affect readability.

"""


grading_instruction_scale_1 = """
  Below is your grading rubric: 

- Correctness: If the answer correctly answer the question, below are the details for different scores:
  - Score 0: the answer is completely incorrect, doesn’t mention anything about the question or is completely contrary to the correct answer.
      - Example: when asked “How to terminate a databricks cluster”, the answer is empty string, or content that’s completely irrelevant, or sorry I don’t know the answer.
  - Score 0: the answer provides some relevance to the question and answer one aspect of the question correctly.
      - Example:
          - Question: How to terminate a databricks cluster
          - Answer: Databricks cluster is a cloud-based computing environment that allows users to process big data and run distributed data processing tasks efficiently.
          - Or answer:  In the Databricks workspace, navigate to the "Clusters" tab. And then this is a hard question that I need to think more about it
  - Score 1: the answer mostly answer the question but is missing or hallucinating on one critical aspect.
      - Example:
          - Question: How to terminate a databricks cluster”
          - Answer: “In the Databricks workspace, navigate to the "Clusters" tab.
          Find the cluster you want to terminate from the list of active clusters.
          And then you’ll find a button to terminate all clusters at once”
  - Score 1: the answer correctly answer the question and not missing any major aspect
      - Example:
          - Question: How to terminate a databricks cluster
          - Answer: In the Databricks workspace, navigate to the "Clusters" tab.
          Find the cluster you want to terminate from the list of active clusters.
          Click on the down-arrow next to the cluster name to open the cluster details.
          Click on the "Terminate" button. A confirmation dialog will appear. Click "Terminate" again to confirm the action.”
- Comprehensiveness: How comprehensive is the answer, does it fully answer all aspects of the question and provide comprehensive explanation and other necessary information. Below are the details for different scores:
  - Score 0: typically if the answer is completely incorrect, then the comprehensiveness is also zero score.
  - Score 0: if the answer is correct but too short to fully answer the question, then we can give score 1 for comprehensiveness.
      - Example:
          - Question: How to use databricks API to create a cluster?
          - Answer: First, you will need a Databricks access token with the appropriate permissions. You can generate this token through the Databricks UI under the 'User Settings' option. And then (the rest is missing)
  - Score 1: the answer is correct and roughly answer the main aspects of the question, but it’s missing description about details. Or is completely missing details about one minor aspect.
      - Example:
          - Question: How to use databricks API to create a cluster?
          - Answer: You will need a Databricks access token with the appropriate permissions. Then you’ll need to set up the request URL, then you can make the HTTP Request. Then you can handle the request response.
      - Example:
          - Question: How to use databricks API to create a cluster?
          - Answer: You will need a Databricks access token with the appropriate permissions. Then you’ll need to set up the request URL, then you can make the HTTP Request. Then you can handle the request response.
  - Score 1: the answer is correct, and covers all the main aspects of the question
- Readability: How readable is the answer, does it have redundant information or incomplete information that hurts the readability of the answer.
  - Score 0: the answer is completely unreadable, e.g. fully of symbols that’s hard to read; e.g. keeps repeating the words that it’s very hard to understand the meaning of the paragraph. No meaningful information can be extracted from the answer.
  - Score 0: the answer is slightly readable, there are irrelevant symbols or repeated words, but it can roughly form a meaningful sentence that cover some aspects of the answer.
      - Example:
          - Question: How to use databricks API to create a cluster?
          - Answer: You you  you  you  you  you  will need a Databricks access token with the appropriate permissions. And then then you’ll need to set up the request URL, then you can make the HTTP Request. Then Then Then Then Then Then Then Then Then
  - Score 1: the answer is correct and mostly readable, but there is one obvious piece that’s affecting the readability (mentioning of irrelevant pieces, repeated words)
      - Example:
          - Question: How to terminate a databricks cluster
          - Answer: In the Databricks workspace, navigate to the "Clusters" tab.
          Find the cluster you want to terminate from the list of active clusters.
          Click on the down-arrow next to the cluster name to open the cluster details.
          Click on the "Terminate" button…………………………………..
          A confirmation dialog will appear. Click "Terminate" again to confirm the action.
  - Score 1: the answer is correct and reader friendly, no obvious piece that affect readability.

"""

grading_instruction_scale_1_no_example = """
  Below is your grading rubric: 

- Correctness: If the answer correctly answer the question, below are the details for different scores:
  - Score 0: the answer is completely incorrect, doesn’t mention anything about the question or is completely contrary to the correct answer.
  - Score 0: the answer provides some relevance to the question and answer one aspect of the question correctly.
  - Score 1: the answer mostly answer the question but is missing or hallucinating on one critical aspect.
  - Score 1: the answer correctly answer the question and not missing any major aspect
- Comprehensiveness: How comprehensive is the answer, does it fully answer all aspects of the question and provide comprehensive explanation and other necessary information. Below are the details for different scores:
  - Score 0: typically if the answer is completely incorrect, then the comprehensiveness is also zero score.
  - Score 0: if the answer is correct but too short to fully answer the question, then we can give score 1 for comprehensiveness.
  - Score 1: the answer is correct and roughly answer the main aspects of the question, but it’s missing description about details. Or is completely missing details about one minor aspect.
  - Score 1: the answer is correct, and covers all the main aspects of the question
- Readability: How readable is the answer, does it have redundant information or incomplete information that hurts the readability of the answer.
  - Score 0: the answer is completely unreadable, e.g. fully of symbols that’s hard to read; e.g. keeps repeating the words that it’s very hard to understand the meaning of the paragraph. No meaningful information can be extracted from the answer.
  - Score 0: the answer is slightly readable, there are irrelevant symbols or repeated words, but it can roughly form a meaningful sentence that cover some aspects of the answer.
  - Score 1: the answer is correct and mostly readable, but there is one obvious piece that’s affecting the readability (mentioning of irrelevant pieces, repeated words)
  - Score 1: the answer is correct and reader friendly, no obvious piece that affect readability.

"""


def get_openai_grading_template_and_function(scale: int, level_of_details: int):
    """
    scale has these possible values: 1, 3, 4, 10
    level_of_details has these possible values: 0, 1, 2
    - 0 means no example, no detail
    - 1 means no example, but has detail
    - 2 means has example, has detail
    """
    if level_of_details == 0:
        # No detail, no example
        grading_instruction = grading_instruction_no_detail
    elif level_of_details == 1:
        # No detail, has example
        if scale == 1:
            # 0~1:
            grading_instruction = grading_instruction_scale_1_no_example
        elif scale == 3:
            # 0~3:
            grading_instruction = grading_instruction_scale_3_no_example
        elif scale == 4:
            # 0~4:
            grading_instruction = grading_instruction_scale_4_no_example
        elif scale == 10:
            # 0~10:
            grading_instruction = grading_instruction_no_detail
        else:
            raise ValueError(f"scale {scale} is not supported")
    elif level_of_details == 2:
        if scale == 1:
            # 0~1:
            grading_instruction = grading_instruction_scale_1
        elif scale == 3:
            # 0~3:
            grading_instruction = grading_instruction_scale_3
        elif scale == 4:
            # 0~4:
            grading_instruction = grading_instruction_scale_4
        elif scale == 10:
            # 0~10:
            grading_instruction = grading_instruction_scale_10
        else:
            raise ValueError(f"scale {scale} is not supported")
    else:
        raise ValueError(f"level_of_details {level_of_details} is not supported")

    if scale == 1:
        openai_grading_function = openai_evaluator_function_scale_1
    elif scale == 3:
        openai_grading_function = openai_evaluator_function_scale_3
    elif scale == 4:
        openai_grading_function = openai_evaluator_function_scale_4
    elif scale == 10:
        openai_grading_function = openai_evaluator_function_scale_10
    else:
        raise ValueError(f"scale {scale} is not supported")
    prompt_template = PromptTemplate(
        """ {grading_system_prompt_intro}

{submission_instruction}

{grading_instruction}

Provided question:
{question}

Provided answer: 
{answer}

Provided context: 
{context}
"""
    ).partial_format(
        submission_instruction=openai_submission_instruction,
        grading_instruction=grading_instruction,
        grading_system_prompt_intro=grading_system_prompt_intro,
    )

    return prompt_template, openai_grading_function


openai_evaluator_function_scale_10 = {
    "name": "grading_function",
    "description": "Call this function to submit the grading for the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "reasoning_for_correctness": {
                "type": "string",
                "description": "Your reasoning for giving the grading for the correctness of the answer. Provide 5 to 30 words explanation.",
            },
            "correctness": {
                "type": "integer",
                "description": "Your integer grading between 0 to 10 for the correctness of the answer.",
            },
            "reasoning_for_comprehensiveness": {
                "type": "string",
                "description": "Your reasoning for giving the grading for the comprehensiveness of the answer. Provide 5 to 30 words explanation.",
            },
            "comprehensiveness": {
                "type": "integer",
                "description": "Your integer grading between 0 to 10 for the comprehensiveness of the answer.",
            },
            "reasoning_for_readability": {
                "type": "string",
                "description": "Your reasoning for giving the grading for the readability of the answer. Provide 5 to 30 words explanation.",
            },
            "readability": {
                "type": "integer",
                "description": "Your integer grading between 0 to 10 for the readability of the answer.",
            },
        },
        "required": [
            "reasoning_for_correctness",
            "correctness",
            "reasoning_for_comprehensiveness",
            "comprehensiveness",
            "reasoning_for_readability",
            "readability",
        ],
    },
}

openai_evaluator_function_scale_4 = {
    "name": "grading_function",
    "description": "Call this function to submit the grading for the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "reasoning_for_correctness": {
                "type": "string",
                "description": "Your reasoning for giving the grading for the correctness of the answer. Provide 5 to 30 words explanation.",
            },
            "correctness": {
                "type": "integer",
                "description": "Your integer grading between 0 to 4 for the correctness of the answer.",
            },
            "reasoning_for_comprehensiveness": {
                "type": "string",
                "description": "Your reasoning for giving the grading for the comprehensiveness of the answer. Provide 5 to 30 words explanation.",
            },
            "comprehensiveness": {
                "type": "integer",
                "description": "Your integer grading between 0 to 4 for the comprehensiveness of the answer.",
            },
            "reasoning_for_readability": {
                "type": "string",
                "description": "Your reasoning for giving the grading for the readability of the answer. Provide 5 to 30 words explanation.",
            },
            "readability": {
                "type": "integer",
                "description": "Your integer grading between 0 to 4 for the readability of the answer.",
            },
        },
        "required": [
            "reasoning_for_correctness",
            "correctness",
            "reasoning_for_comprehensiveness",
            "comprehensiveness",
            "reasoning_for_readability",
            "readability",
        ],
    },
}


openai_evaluator_function_scale_3 = {
    "name": "grading_function",
    "description": "Call this function to submit the grading for the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "reasoning_for_correctness": {
                "type": "string",
                "description": "Your reasoning for giving the grading for the correctness of the answer. Provide 5 to 30 words explanation.",
            },
            "correctness": {
                "type": "integer",
                "description": "Your integer grading between 0 to 3 for the correctness of the answer.",
            },
            "reasoning_for_comprehensiveness": {
                "type": "string",
                "description": "Your reasoning for giving the grading for the comprehensiveness of the answer. Provide 5 to 30 words explanation.",
            },
            "comprehensiveness": {
                "type": "integer",
                "description": "Your integer grading between 0 to 3 for the comprehensiveness of the answer.",
            },
            "reasoning_for_readability": {
                "type": "string",
                "description": "Your reasoning for giving the grading for the readability of the answer. Provide 5 to 30 words explanation.",
            },
            "readability": {
                "type": "integer",
                "description": "Your integer grading between 0 to 3 for the readability of the answer.",
            },
        },
        "required": [
            "reasoning_for_correctness",
            "correctness",
            "reasoning_for_comprehensiveness",
            "comprehensiveness",
            "reasoning_for_readability",
            "readability",
        ],
    },
}

openai_evaluator_function_scale_1 = {
    "name": "grading_function",
    "description": "Call this function to submit the grading for the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "reasoning_for_correctness": {
                "type": "string",
                "description": "Your reasoning for giving the grading for the correctness of the answer. Provide 5 to 30 words explanation.",
            },
            "correctness": {
                "type": "integer",
                "description": "Your integer grading between 0 or 1 for the correctness of the answer.",
            },
            "reasoning_for_comprehensiveness": {
                "type": "string",
                "description": "Your reasoning for giving the grading for the comprehensiveness of the answer. Provide 5 to 30 words explanation.",
            },
            "comprehensiveness": {
                "type": "integer",
                "description": "Your integer grading between 0 or 1 for the comprehensiveness of the answer.",
            },
            "reasoning_for_readability": {
                "type": "string",
                "description": "Your reasoning for giving the grading for the readability of the answer. Provide 5 to 30 words explanation.",
            },
            "readability": {
                "type": "integer",
                "description": "Your integer grading between 0 or 1 for the readability of the answer.",
            },
        },
        "required": [
            "reasoning_for_correctness",
            "correctness",
            "reasoning_for_comprehensiveness",
            "comprehensiveness",
            "reasoning_for_readability",
            "readability",
        ],
    },
}


anthropic_submission_instruction = """And you'll need to submit your grading for the correctness, comprehensiveness and readability of the answer, using the following format:
Reasoning for correctness: [your one line step by step reasoning about the correctness of the answer]
Score for correctness: [your score number for the correctness of the answer]
Reasoning for comprehensiveness: [your one line step by step reasoning about the comprehensiveness of the answer]
Score for comprehensiveness: [your score number for the comprehensiveness of the answer]
Reasoning for readability: [your one line step by step reasoning about the readability of the answer]
Score for readability: [your score number for the readability of the answer]

"""

anthropic_end_prompt = """Please provide your grading for the correctness, comprehensiveness and readability of the answer"""

anthropic_grading_template_scale_10 = PromptTemplate(
    """{grading_system_prompt_intro}

{submission_instruction}

{grading_instruction}

Provided question:
{question}

Provided answer: 
{answer}

Provided context: 
{context}

{end_prompt}
"""
).partial_format(
    submission_instruction=anthropic_submission_instruction,
    grading_instruction=grading_instruction_scale_10,
    grading_system_prompt_intro=grading_system_prompt_intro,
    end_prompt=anthropic_end_prompt,
)

anthropic_grading_template_scale_4 = PromptTemplate(
    """{grading_system_prompt_intro}

{submission_instruction}

{grading_instruction}

Provided question:
{question}

Provided answer: 
{answer}

Provided context: 
{context}

{end_prompt}
"""
).partial_format(
    submission_instruction=anthropic_submission_instruction,
    grading_instruction=grading_instruction_scale_4,
    grading_system_prompt_intro=grading_system_prompt_intro,
    end_prompt=anthropic_end_prompt,
)


anthropic_grading_template_scale_3 = PromptTemplate(
    """{grading_system_prompt_intro}

{submission_instruction}

{grading_instruction}

Provided question:
{question}

Provided answer: 
{answer}

Provided context: 
{context}

{end_prompt}
"""
).partial_format(
    submission_instruction=anthropic_submission_instruction,
    grading_instruction=grading_instruction_scale_3,
    grading_system_prompt_intro=grading_system_prompt_intro,
    end_prompt=anthropic_end_prompt,
)

anthropic_grading_template_scale_1 = PromptTemplate(
    """{grading_system_prompt_intro}

{submission_instruction}

{grading_instruction}

Provided question:
{question}

Provided answer: 
{answer}

Provided context: 
{context}

{end_prompt}
"""
).partial_format(
    submission_instruction=anthropic_submission_instruction,
    grading_instruction=grading_instruction_scale_1,
    grading_system_prompt_intro=grading_system_prompt_intro,
    end_prompt=anthropic_end_prompt,
)


def vicuna_prompt_format_func(message: str, system_prompt_opt: str) -> str:
    if system_prompt_opt is not None:
        return f"""{system_prompt_opt}

        USER: {message}
        ASSISTANT:
        """
    else:
        return f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

        USER: {message}
        ASSISTANT:
        """


def llama2_prompt_format_func(message: str, system_prompt_opt: str) -> str:
    if system_prompt_opt is not None:
        texts = [f"[INST] <<SYS>>\n{system_prompt_opt}\n<</SYS>>\n\n"]
        texts.append(f"{message.strip()} [/INST]")
        return "".join(texts)
    else:
        texts = [f"[INST] \n\n"]
        texts.append(f"{message.strip()} [/INST]")
        return "".join(texts)


def llama2_prompt_format_func_without_sys(message: str, system_prompt_opt: str) -> str:
    texts = [f"[INST] \n\n"]
    texts.append(f"{message.strip()} [/INST]")
    return "".join(texts)


def open_assistant_prompt_format_func(message: str, system_prompt_opt: str) -> str:
    if system_prompt_opt is not None:
        return f"""{system_prompt_opt}

        ### Human: {message}
        ### Assistant: 
        """
    else:
        return f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

        ### Human: {message}
        ### Assistant: 
        """


def raw_prompt_format_func(message: str, system_prompt_opt: str) -> str:
    if system_prompt_opt is not None:
        return f"""{system_prompt_opt}\n\n {message.strip()}"""
    else:
        return message


doc_qa_task_prompt_template = PromptTemplate(
    """You are a helpful, respectful and honest assistant. Your task is to help answering a question based on the provided context, please think step by step to make sure your answer is correct, comprehensive and easy to read. Please do not ask for further information, what you will be provided is already the complete information. 

Here is the question: 
{question}


Here is the context:
{context}

Please give your answer:
"""
)
