# capstone_analysis.py
import subprocess
import sys

def run(cmd):
    print("RUN:", " ".join(cmd))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    # 1) Clean data
    run([sys.executable, "notebooks/1_data_cleaning.py"])
    # 2) Train models
    run([sys.executable, "src/train_models.py"])
    print("Pipeline complete. Models are in models/")
