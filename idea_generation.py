# idea_generation.py

import openai
import logging
from config import OPENAI_API_KEY, MODEL_NAME, NUM_IDEAS

openai.api_key = OPENAI_API_KEY

def generate_ideas():
    prompt = (
        f"Generate {NUM_IDEAS} specific, measurable research ideas to improve this AI research system's performance. "
        "Each idea should focus on enhancing one of the following metrics:"
        "\n1. Idea relevance: Increase the percentage of generated ideas directly applicable to AI/ML research"
        "\n2. Idea novelty: Boost the originality score of generated ideas as evaluated by domain experts"
        "\n3. Execution efficiency: Reduce the average time taken to generate a set of ideas"
        "\n4. Resource utilization: Optimize GPU memory usage during idea generation"
        "\n5. Output consistency: Improve the coherence and logical flow between generated ideas"
        "\nIdeas should be implementable within one week using a single GPU. For each idea, specify:"
        "\n- The target metric for improvement"
        "\n- A proposed method for implementation"
        "\n- An expected quantitative outcome (e.g., 20% increase in relevance score)"
        "\nList ideas as bullet points."
    )

    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            n=1,
            temperature=0.7,
        )

        ideas_text = response['choices'][0]['message']['content'].strip()
        ideas = [idea.strip('-• ').strip() for idea in ideas_text.split('\n') if idea.strip()]
        logging.info(f"Generated ideas: {ideas}")
        return ideas

    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI API error during idea generation: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error during idea generation: {e}")
        return []
