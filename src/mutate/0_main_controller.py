import os
import subprocess
import argparse

def run_script(script_name, args):
    """
    Run a Python script with specified command line arguments.
    
    :param script_name: Script filename (e.g., "scriptA.py")
    :param args: Parameter list (e.g., ["--var1", "A1", "--mode", "test"])
    """
    # Construct the full path to the script (script and control script are in the same directory)
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    print(f"Starting to run {script_name}, parameters: {args}")
    try:
        # Use subprocess.run to call
        # Note: concatenate "python", script path, and parameter list
        subprocess.run(["python", script_path] + args, check=True)
        print(f"Finished running {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adjust the path and others.')
    parser.add_argument('--book', type=str, help='select the book')
    parser.add_argument('--technique_dir', type=str, help='create the technique dir')
    parser.add_argument('--technique', type=str, help='select the technique')
    args = parser.parse_args()
    # 1. Define the execution order of three scripts
    scripts_in_order = ["1_run.py", "2_inference_scaling_all.py", "3_data_statistics.py"]

    # 2. Specify parameters for each script separately
    #   - If the script does not need parameters, provide an empty list []
    script_args = {
        "1_run.py": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"],
        "2_inference_scaling_all.py": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"],
        "3_data_statistics.py": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"]  # Can also use non --key value format
    }

    # 3. Run scripts in order
    for script_name in scripts_in_order:
        args_for_script = script_args.get(script_name, [])
        run_script(script_name, args_for_script)

    print("All three scripts have been executed in the specified order.")
