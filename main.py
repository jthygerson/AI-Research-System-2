# main.py

import logging
import argparse
from utils import initialize_logging, save_report, parse_experiment_plan, get_latest_log_file
from idea_generation import generate_ideas
from idea_evaluation import evaluate_ideas
from experiment_design import design_experiment
from experiment_execution import execute_experiment
from feedback_loop import refine_experiment
from self_optimization import optimize_system, analyze_log_file
from benchmarking import benchmark_system
from code_updater import update_code
from config import set_config_parameters, MAX_ATTEMPTS

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='AI Research System')
    parser.add_argument('--model', type=str, default='gpt-4o', help='OpenAI model name')
    parser.add_argument('--num_ideas', type=int, default=5, help='Number of ideas to generate')
    parser.add_argument('--max_attempts', type=int, default=3, help='Maximum number of attempts')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs to execute')
    args = parser.parse_args()

    # Set configuration parameters
    set_config_parameters(args)

    initialize_logging()
    logging.info("AI Research System Started.")

    try:
        for run_number in range(1, args.num_runs + 1):
            logging.info(f"Starting run {run_number} of {args.num_runs}")

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
                    refined_parameters = parameters.copy()
                    refined_parameters.update(parse_experiment_plan(refined_plan))
                    refined_results = execute_experiment(refined_parameters)
                    if not refined_results:
                        logging.error("Refined experiment execution failed.")
                        continue
                else:
                    refined_results = results

                # Step 7: Self-Optimization
                optimization_results = optimize_system(refined_results)
                if len(optimization_results) >= 2:
                    code_changes = optimization_results[0]
                    improvement_descriptions = optimization_results[1]
                    # If there are additional return values, we can handle them here if needed
                else:
                    logging.error("System optimization failed or returned unexpected results.")
                    continue

                if code_changes:
                    # Benchmark before applying changes
                    before_benchmark = benchmark_system()
                    
                    # Apply code changes
                    update_code(code_changes)
                    
                    # Benchmark after applying changes
                    after_benchmark = benchmark_system()
                else:
                    logging.warning("No code changes suggested by optimization.")
                    before_benchmark = after_benchmark = None

                try:
                    # Generate experiment report
                    generate_report(best_idea, experiment_plan, refined_results, 
                                    before_benchmark, after_benchmark, 
                                    code_changes, improvement_descriptions)
                    logging.info(f"Report generated successfully for run {run_number}, attempt {attempts}")
                    success = True
                except Exception as report_error:
                    logging.error(f"Failed to generate report for run {run_number}, attempt {attempts}: {report_error}")
                    # If report generation fails, we'll try again in the next attempt

            if not success:
                logging.error(f"Maximum attempts reached without success for run {run_number}.")

            # After each run, analyze the log file
            latest_log_file = get_latest_log_file()
            if latest_log_file:
                suggested_changes = analyze_log_file(latest_log_file)
                if suggested_changes:
                    logging.info("Applying suggested changes from log analysis")
                    update_code(suggested_changes)
                else:
                    logging.info("No changes suggested from log analysis")
            else:
                logging.warning("No log file found for analysis")

    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
        # Rollback code changes if any
        update_code(rollback=True)

    logging.info("AI Research System Finished.")

def generate_report(idea, experiment_plan, results, before_benchmark, after_benchmark, 
                    code_changes, improvement_descriptions):
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
### Before Code Changes
{before_benchmark}

### After Code Changes
{after_benchmark}
"""

    if code_changes:
        report_content += "\n## Code Changes\n"
        for i, change in enumerate(code_changes):
            report_content += f"\n### Change {i+1}: {change['filename']}\n"
            report_content += f"**Original Code:**\n```python\n{change['original_code']}\n```\n"
            report_content += f"**Updated Code:**\n```python\n{change['updated_code']}\n```\n"
            report_content += f"**Expected Improvement:**\n{improvement_descriptions[i]}\n"

    report_content += """
## Conclusion
Based on the experiment results and benchmarking, the AI Research System has been updated to improve its performance.
The before and after benchmark results demonstrate the impact of these changes.
"""

    save_report(idea, report_content)

if __name__ == "__main__":
    main()
