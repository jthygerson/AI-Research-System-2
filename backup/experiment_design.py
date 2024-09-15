# experiment_design.py

import openai
import logging
from config import OPENAI_API_KEY, MODEL_NAME
from utils import parse_experiment_plan

openai.api_key = OPENAI_API_KEY

def design_experiment(idea):
    prompt = (
        f"Design a detailed experiment in the field of AI/ML to test the following idea, which aims to improve the AI Research System's own performance:\nIdea: {idea}\n"
        "Provide the experiment plan with the following sections:\n"
        "1. Objective\n"
        "2. Methodology\n"
        "3. Datasets (specify dataset names or sources available on Hugging Face Datasets)\n"
        "4. Model Architecture (specify model types)\n"
        "5. Hyperparameters (list them as key-value pairs)\n"
        "6. Evaluation Metrics\n"
        "Ensure that each section is clearly labeled and relevant to AI/ML."
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

        experiment_plan = response['choices'][0]['message']['content'].strip()
        logging.info(f"Experiment Plan:\n{experiment_plan}")

        # Parse the experiment plan into parameters
        parameters = parse_experiment_plan(experiment_plan)
        return experiment_plan, parameters

    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI API error during experiment design: {e}")
        return None, {}
    except Exception as e:
        logging.error(f"Unexpected error during experiment design: {e}")
        return None, {}
