import os
import subprocess

# Function to convert .ipynb to .py
def convert_ipynb_to_py(ipynb_file):
    try:
        os.system(f'jupyter nbconvert --to script --no-prompt --output-dir=. {ipynb_file}')
    except Exception as e:
        print(f"An error occurred while converting {ipynb_file} to .py: {e}")


# Function to commit and push changes to Git
def git_commit_and_push(commit_message):
    try:
        subprocess.run(['git', 'add', '.'])
        subprocess.run(['git', 'commit', '-m', commit_message])
        subprocess.run(['git', 'push'])
        print("Changes committed and pushed to Git successfully!")
    except Exception as e:
        print(f"An error occurred while committing and pushing changes to Git: {e}")

if __name__ == "__main__":
    # Specify the path to your .ipynb file
    ipynb_file = "./grad_engine.ipynb"

    # Ask user for commit message
    commit_message = input("Enter the commit message: ")

    # Convert .ipynb to .py
    convert_ipynb_to_py(ipynb_file)

    # Commit and push changes to Git
    git_commit_and_push(commit_message)
