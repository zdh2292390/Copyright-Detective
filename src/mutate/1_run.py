import os
import subprocess
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import argparse

def count_total_rows(csv_files):
    """
    Count the total number of rows across multiple CSV files.
    :param csv_files: list of CSV file paths
    :return: total rows across all CSV files
    """
    total_rows = 0
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            total_rows += len(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return total_rows

def run_scripts_in_directory(directory, script_sequence, script_args_map, script_csv_map):
    """
    Run scripts in a single directory in a specific order. Some scripts may auto-repeat.
    Each auto-repeat script uses a different set of CSV files for the row-count check.
    
    :param directory: path to the directory
    :param script_sequence: list of (script_name, auto_repeat)
    :param script_args_map: dict => script_name -> [arg1, arg2, ...] (if needed)
    :param script_csv_map: dict => script_name -> list_of_csv_files
                           If a script needs repeated check, we pick its csv list from here
    """
    print(f"[{directory}] Starting scripts...")

    for script_name, auto_repeat in script_sequence:
        script_path = os.path.join(directory, script_name)
        # get args for this script
        args = script_args_map.get(script_name, [])

        # which csv files does this script check? if not specified, default to None
        csv_files_for_script = script_csv_map.get(script_name, [])

        while True:
            print(f"[{directory}] Running {script_name} with args: {args}")
            try:
                subprocess.run(["python", script_path] + args, check=True)
                print(f"[{directory}] Finished {script_name}")
            except subprocess.CalledProcessError as e:
                print(f"[{directory}] Error running {script_name}: {e}")
                # If error, decide to skip or break
                break

            if auto_repeat:
                # If auto_repeat == True, do row-count check on script-specific CSV
                if csv_files_for_script:
                    total_rows = count_total_rows(csv_files_for_script)
                    print(f"[{directory}] {script_name} row check: total_rows={total_rows}")
                    if total_rows <= 60:
                        print(f"[{directory}] {script_name}: Rows <= 60, repeating...")
                        # keep looping (not break)
                    else:
                        print(f"[{directory}] {script_name}: Rows > 60, move on.")
                        break
                else:
                    # If no CSV specified for a repeated script, we can't check anything.
                    # We'll just run once.
                    print(f"[{directory}] {script_name}: no CSV set, no check performed, proceed next.")
                    break
            else:
                # script doesn't repeat
                break
    
    print(f"[{directory}] All scripts done in this directory.")

def run_all_directories(directories, script_sequence, script_args_map, script_csv_map):
    """
    Run scripts for each directory in parallel. Each directory runs script_sequence in order.
    :param directories: list of directory paths
    :param script_sequence: e.g. [("scriptA.py", True), ("scriptB.py", False), ...]
    :param script_args_map: dict => { script_name: [args] }
    :param script_csv_map: dict => { script_name: [csv1, csv2, ...] } for row-check
    """
    with ThreadPoolExecutor() as executor:
        futures = []
        for d in directories:
            futures.append(
                executor.submit(run_scripts_in_directory, d, script_sequence, script_args_map, script_csv_map)
            )
        for f in futures:
            f.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adjust the path and others.')
    parser.add_argument('--book', type=str, help='select the book')
    parser.add_argument('--technique_dir', type=str, help='create the technique dir')
    parser.add_argument('--technique', type=str, help='select the technique')
    args = parser.parse_args()

    # 1. directories: 6 directories
    directories = [
        f"""./mutate/{args.book}/original_question_1""",
        f"""./mutate/{args.book}/original_question_2""",
        f"""./mutate/{args.book}/original_question_3""",
        f"""./mutate/{args.book}/original_question_4""",
        f"""./mutate/{args.book}/original_question_5""",
        f"""./mutate/{args.book}/original_question_6"""
    ]

    # 2. script_sequence => list of (script_name, auto_repeat)
    #    auto_repeat=True means it will keep re-running until row sum>60
    script_sequence = [
        ("1_zero_shot_with_judge.py", True),   # uses csv 1~6_zero_shot_with_judge.csv
        ("2_eval_zero_shot_with_judge.py", False), 
        ("3_zero_shot_without_judge.py", False),  
        ("4_eval_zero_shot_without_judge.py", False),
        ("5_few_shots_with_judge.py", True),   # uses csv 1~6_few_shots_with_judge.csv
        ("6_eval_few_shots_with_judge.py", False),
        ("7_few_shots_without_judge.py", False),
        ("8_eval_few_shots_without_judge.py", False),
    ]
    # Configuration section
    # 3. script_args_map => if you want to pass arguments to certain scripts
    script_args_map = {
        "1_zero_shot_with_judge.py": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"],
        "2_eval_zero_shot_with_judge.py": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"],
        "3_zero_shot_without_judge.py": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"],
        "4_eval_zero_shot_without_judge.py": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"],
        "5_few_shots_with_judge.py": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"],
        "6_eval_few_shots_with_judge.py": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"],
        "7_few_shots_without_judge.py": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"],
        "8_eval_few_shots_without_judge.py": ["--book", f"{args.book}", "--technique", f"{args.technique}", "--technique_dir", f"{args.technique_dir}"]
        # others default to []
    }

    # 4. script_csv_map => define which CSV set each repeated script checks
    #    scriptA uses csv_files_A, scriptC uses csv_files_B, etc.
    script_csv_map = {
        "1_zero_shot_with_judge.py": [
            f"""./outputs/2_persuasion_prompts/{args.book}/{args.technique_dir}/1_zero_shot_with_judge.csv""",
            f"""./outputs/2_persuasion_prompts/{args.book}/{args.technique_dir}/2_zero_shot_with_judge.csv""",
            f"""./outputs/2_persuasion_prompts/{args.book}/{args.technique_dir}/3_zero_shot_with_judge.csv""",
            f"""./outputs/2_persuasion_prompts/{args.book}/{args.technique_dir}/4_zero_shot_with_judge.csv""",
            f"""./outputs/2_persuasion_prompts/{args.book}/{args.technique_dir}/5_zero_shot_with_judge.csv""",
            f"""./outputs/2_persuasion_prompts/{args.book}/{args.technique_dir}/6_zero_shot_with_judge.csv"""
        ],
        "5_few_shots_with_judge.py": [
            f"""./outputs/2_persuasion_prompts/{args.book}/{args.technique_dir}/1_few_shots_with_judge.csv""",
            f"""./outputs/2_persuasion_prompts/{args.book}/{args.technique_dir}/2_few_shots_with_judge.csv""",
            f"""./outputs/2_persuasion_prompts/{args.book}/{args.technique_dir}/3_few_shots_with_judge.csv""",
            f"""./outputs/2_persuasion_prompts/{args.book}/{args.technique_dir}/4_few_shots_with_judge.csv""",
            f"""./outputs/2_persuasion_prompts/{args.book}/{args.technique_dir}/5_few_shots_with_judge.csv""",
            f"""./outputs/2_persuasion_prompts/{args.book}/{args.technique_dir}/6_few_shots_with_judge.csv"""
        ]
        # scripts B and D do not auto-repeat, so we can skip
    }

    # 5. run
    run_all_directories(directories, script_sequence, script_args_map, script_csv_map)
    print("All directories completed.")

'''
python run.py --book 1_HP --technique "Foot-in-the-Door" --technique_dir 16_Foot-in-the-Door
'''
