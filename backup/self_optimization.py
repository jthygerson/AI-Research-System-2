# self_optimization.py

import openai
import logging
from config import OPENAI_API_KEY, MODEL_NAME

openai.api_key = OPENAI_API_KEY

def optimize_system(results):
    prompt = (
        f"The AI Research System has obtained the following results:\n{results}\n"
        "Based on these results, suggest specific changes to the system's code or parameters to improve its performance on AI benchmark tests. "
        "Provide the suggested code changes in the following format:\n\n"
        "```python\n"
        "# File: <filename>\n"
        "# Original Code:\n"
        "<original_code>\n"
        "# Updated Code:\n"
        "<updated_code>\n"
        "```\n\n"
        "Explain how these changes will improve the system. Ensure the suggestions are safe and do not introduce errors."
    )

    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            n=1,
            temperature=0.7,
        )

        suggestions = response['choices'][0]['message']['content'].strip()
        logging.info(f"Optimization Suggestions:\n{suggestions}")

        # Parse the code changes from the suggestions
        code_changes = parse_code_changes(suggestions)

        # Apply code changes (optional)
        # For safety, you may choose to manually review changes before applying them

        return code_changes  # Return the code changes for inclusion in the report

    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI API error during system optimization: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during system optimization: {e}")
        return None

def parse_code_changes(suggestions):
    """
    Parses the code changes from the suggestions.
    Expects the suggestions to be in the specified format.
    """
    import re

    code_changes = []

    pattern = r"```python\n# File: (.+?)\n# Original Code:\n(.+?)\n# Updated Code:\n(.+?)```"
    matches = re.findall(pattern, suggestions, re.DOTALL)

    for match in matches:
        filename, original_code, updated_code = match
        code_changes.append({
            'filename': filename.strip(),
            'original_code': original_code.strip(),
            'updated_code': updated_code.strip()
        })

    return code_changes
