import subprocess
import sys
import os
# runs the scripts to create a vector db in sequence
# useful when needing to update the db with new information.
def main():
    scripts = [
        "wikiscraper.py",
        "article_cleanup.py",
        "chunk_articles.py"
    ]

    for script in scripts:
        if not os.path.exists(script):
            print(f"Error: File '{script}' not found.")
            return

        print(f"--- Running {script} ---")
        try:
            subprocess.run([sys.executable, script], check=True)
        except subprocess.CalledProcessError:
            print(f"Pipeline stopped: '{script}' failed.")
            return
        print("") 

    print("Pipeline finished successfully.")

if __name__ == "__main__":
    main()