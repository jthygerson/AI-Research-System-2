# feedback_loop.py

import openai
import logging
from config import OPENAI_API_KEY, MODEL_NAME

openai.api_key = OPENAI_API_KEY

def refine_experiment(experiment_plan, results):
    prompt = (
        f"The experiment was conducted as per the following plan:\n{experiment_plan}\n"
        f"The results obtained are: {results}\n"
        "Based on these results, suggest improvements to enhance performance. "
        "Update the experiment plan accordingly. Ensure the updated plan has the same sections as before."
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

        refined_plan = response['choices'][0]['message']['content'].strip()
        logging.info(f"Refined Experiment Plan:\n{refined_plan}")

        return refined_plan

    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI API error during experiment refinement: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during experiment refinement: {e}")
        return None
