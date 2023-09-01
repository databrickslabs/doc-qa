
## Doc QA Evaluation Tool based on Databricks documentation

### How to use it:

**Step 1: Install the repo as pip dependency:**

```
!pip install git+https://x-access-token:[your github token]@github.com/databricks/doc_qa.git
```

**Step 2: use it in your code:**
* You can find the detailed examples in try_templated_evaluator.ipynb

```
## Load the environment variables:

import dotenv
dotenv.load_dotenv('.env')

## Setup the evaluator prompts

from doc_qa.llm_utils import PromptTemplate
import pandas as pd
from doc_qa.evaluators.templated_evaluator import OpenAIEvaluator, RetryPolicy
from doc_qa.variables.doc_qa_template_variables import get_openai_grading_template_and_function

import logging
logging.basicConfig(level=logging.INFO)


retry_policy = RetryPolicy(max_retry_on_invalid_result=3, max_retry_on_exception=3)
catch_error = True

openai_grading_prompt, openai_grading_function = get_openai_grading_template_and_function(scale=1, level_of_details=2)

## Load answer sheet (csv)

target_df = pd.read_csv("pre_grade_datasets/pre_grade_doc_qa_mpt_30b_chat.csv")


## Evaluate and show results:

from doc_qa.evaluators.templated_evaluator import OpenAIEvaluator

openai_gpt_4_evaluator = OpenAIEvaluator(model="gpt-4", temperature=0.1, 
    grading_prompt_tempate=openai_grading_prompt, 
    input_columns=["question", "answer", "context"], openai_function=openai_grading_function,
    retry_policy=retry_policy)
eval_result = openai_gpt_4_evaluator.run_eval(dataset_df=target_df, concurrency=20, catch_error=catch_error)
result_df = eval_result.to_dataframe()
print(eval_result.summary())
result_df

```


### Learnings:
* For gpt-3.5-turbo-16k, putting all instructions and context into single user prompt has better chance to call function than separating them into system and user prompt or putting all of the into system prompt
* * Everything into user prompt: 98% chance to call function
* * Putting grading rules into system prompt and context into user prompt: 86% chance to call function 
* For non-function-compatible models, it's highly necessary to add an instruction at the end of the prompt to guide it to provide the result format, otherwise the model will easily miss to give the result or follow the format: (82% success rate for claude-2, 98% success rate for claude-1) vs (96% success rate for claude-2, 100% success rate for claude-1) with end instruction for all. 