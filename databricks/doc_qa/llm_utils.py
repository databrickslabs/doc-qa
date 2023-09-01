import re

class PromptTemplate:
    def __init__(self, template_str: str, variables: list = None):
        self.template_str = template_str
        if variables:
            self.variables = variables
        else:
            # Automatically parse variables from template string
            self.variables = re.findall(r"\{(\w+)\}", template_str)

    def format_prompt(self, **kwargs: dict):
        # Only keep the kwargs that are in the variables
        kwargs = {k: v for k, v in kwargs.items() if k in self.variables}

        # Format the prompt with the provided values
        return self.template_str.format(**kwargs)

    def partial_fill(self, **kwargs: dict):
        # Create a safe dictionary that returns the key if it doesn't exist in the dictionary
        safe_dict = {k: kwargs.get(k, '{' + k + '}') for k in self.variables}

        # Fill in the provided values, and return a new PromptTemplate
        new_template_str = self.template_str.format_map(safe_dict)
        unfilled_variables = [var for var in self.variables if var not in kwargs.keys()]
        return PromptTemplate(template_str=new_template_str, variables=unfilled_variables)