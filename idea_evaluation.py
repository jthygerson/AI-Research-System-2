# idea_evaluation.py

import openai
import logging
import re
from config import OPENAI_API_KEY, MODEL_NAME

openai.api_key = OPENAI_API_KEY

def evaluate_ideas(ideas):
    best_idea = None
    highest_score = 0

    for idea in ideas:
        prompt = (
            f"Evaluate the following idea on a scale of 1-10 for novelty and probability "
            f"of success in improving AI systems, considering current AI research trends.\nIdea: {idea}\n"
            "Provide ONLY your evaluation in the EXACT format: 'Novelty: X, Success Probability: Y' "
            "without any additional text."
        )

        try:
            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                n=1,
                temperature=0.3,
                stop=["\n"]
            )

            scores_text = response['choices'][0]['message']['content'].strip()
            novelty_score, success_score = parse_scores(scores_text)

            if novelty_score > 0 and success_score > 0:
                total_score = novelty_score + success_score
                if total_score > highest_score:
                    highest_score = total_score
                    best_idea = idea

                logging.info(f"Idea: {idea}, Novelty: {novelty_score}, Success: {success_score}")
            else:
                logging.error(f"Invalid scores for idea '{idea}'. Response was: '{scores_text}'")

        except openai.error.OpenAIError as e:
            logging.error(f"OpenAI API error during idea evaluation: {e}")
            continue
        except Exception as e:
            logging.error(f"Unexpected error during idea evaluation: {e}")
            continue

    return best_idea

def parse_scores(scores_text):
    try:
        novelty_match = re.search(r'Novelty:\s*(\d+)', scores_text, re.IGNORECASE)
        success_match = re.search(r'Success Probability:\s*(\d+)', scores_text, re.IGNORECASE)

        if novelty_match and success_match:
            novelty = int(novelty_match.group(1))
            success = int(success_match.group(1))
            if 1 <= novelty <= 10 and 1 <= success <= 10:
                return novelty, success
            else:
                logging.error(f"Scores out of range: Novelty={novelty}, Success={success}")
                return 0, 0
        else:
            logging.error(f"Could not parse scores from: '{scores_text}'")
            return 0, 0
    except Exception as e:
        logging.error(f"Error parsing scores '{scores_text}': {e}")
        return 0, 0
