import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import argparse

def run_script(script_path, args):
    """
    Run a Python script (with absolute or relative path) and pass in arguments.
    :param script_path: Full path to the script file
    :param args: List of arguments, e.g. ["--var1", "A1", ...]
    """
    print(f"Starting to run script: {script_path}, passing parameters: {args}")
    try:
        subprocess.run(["python", script_path] + args, check=True)
        print(f"Finished running script: {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {script_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adjust the path and others.')
    parser.add_argument('--book', type=str, help='select the book')
    parser.add_argument('--technique_dir', type=str, help='create the technique dir')
    parser.add_argument('--technique', type=str, help='select the technique')
    args = parser.parse_args()
    # Define script directory, e.g., Project/scripts
    scripts_dir = f"./mutate/{args.book}/inference_scaling"

    # Define script name list
    script_names = ["1_zero_shot_with_judge_inference_scaling.py", "2_zero_shot_without_judge_inference_scaling.py", "3_few_shots_with_judge_inference_scaling.py", "4_few_shots_without_judge_inference_scaling.py"]

    # Concatenate full script paths
    # If scripts need parameters, they can be specified here or through a dictionary
    scripts_info = {
        "1_zero_shot_with_judge_inference_scaling.py": {
            "path": os.path.join(scripts_dir, "1_zero_shot_with_judge_inference_scaling.py"),
            "args": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"]
        },
        "2_zero_shot_without_judge_inference_scaling.py": {
            "path": os.path.join(scripts_dir, "2_zero_shot_without_judge_inference_scaling.py"),
            "args": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"]
        },
        "3_few_shots_with_judge_inference_scaling.py": {
            "path": os.path.join(scripts_dir, "3_few_shots_with_judge_inference_scaling.py"),
            "args": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"]
        },
        "4_few_shots_without_judge_inference_scaling.py": {
            "path": os.path.join(scripts_dir, "4_few_shots_without_judge_inference_scaling.py"),
            "args": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"]
        }
    }

    # Run four scripts in parallel
    with ThreadPoolExecutor() as executor:
        futures = []
        for script_name in script_names:
            script_path = scripts_info[script_name]["path"]
            script_args = scripts_info[script_name]["args"]
            futures.append(executor.submit(run_script, script_path, script_args))

        # Wait for all parallel tasks to complete
        for f in futures:
            f.result()

    print("All scripts have finished running.")
