# main.py

import logging
from utils import initialize_logging, save_report
from idea_generation import generate_ideas
from idea_evaluation import evaluate_ideas
from experiment_design import design_experiment
from experiment_execution import execute_experiment
from feedback_loop import refine_experiment
from self_optimization import optimize_system
from benchmarking import benchmark_system
from code_updater import update_code
from config import MAX_ATTEMPTS

def main():
    initialize_logging()
    logging.info("AI Research System Started.")

    try:
        attempts = 0
        success = False

        while attempts < MAX_ATTEMPTS and not success:
            attempts += 1
            logging.info(f"Attempt {attempts} of {MAX_ATTEMPTS}")

            # Step 1: Idea Generation
            ideas = generate_ideas()
            if not ideas:
                logging.error("Failed to generate ideas.")
                continue

            # Step 2: Idea Evaluation
            best_idea = evaluate_ideas(ideas)
            if not best_idea:
                logging.error("Failed to evaluate ideas.")
                continue

            # Step 3: Experiment Design
            experiment_plan, parameters = design_experiment(best_idea)
            if not experiment_plan:
                logging.error("Failed to design experiment.")
                continue

            # Step 4: Experiment Execution
            results = execute_experiment(parameters)
            if not results:
                logging.error("Experiment execution failed.")
                continue

            # Step 5: Feedback Loop
            refined_plan = refine_experiment(experiment_plan, results)
            if refined_plan:
                # Step 6: Refined Experiment Execution
                refined_results = execute_experiment(parameters)
                if not refined_results:
                    logging.error("Refined experiment execution failed.")
                    continue
            else:
                refined_results = results

            # Step 7: Self-Optimization
            optimization_success = optimize_system(refined_results)
            if not optimization_success:
                logging.error("System optimization failed.")
                continue

            # Step 8: Benchmarking
            benchmark_results = benchmark_system()
            logging.info(f"Benchmark Results: {benchmark_results}")

            # Generate experiment report
            generate_report(best_idea, experiment_plan, refined_results, benchmark_results)

            success = True

        if not success:
            logging.error("Maximum attempts reached without success.")

    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
        # Rollback code changes if any
        update_code(rollback=True)

    logging.info("AI Research System Finished.")

def generate_report(idea, experiment_plan, results, benchmark_results):
    from utils import save_report
    report_content = f"""
# Experiment Report: {idea[:50]}

## Idea
{idea}

## Experiment Plan
{experiment_plan}

## Results
{results}

## Benchmark Results
{benchmark_results}

## Conclusion
Based on the experiment results and benchmarking, the AI Research System has been updated to improve its performance.
"""

    save_report(idea, report_content)

if __name__ == "__main__":
    main()
