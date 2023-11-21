import subprocess

# Directory where the scripts are located
script_directory = "Functions"

# List of programs to run in order
programs = ['cortarcsv.py', 'concat.py', 'features.py', 'RF.py']  # Replace 'RF.py' with 'RFevaluator.py' if needed

# Loop through each program and execute it
for program in programs:
    script_path = f"{script_directory}/{program}"
    print(f"Running {program}...")
    subprocess.run(['python', script_path])
    print(f"Finished running {program}")

print("All programs executed successfully.")
