import os
import glob
import subprocess
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run multiple training component scripts.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of components to run")
    args = parser.parse_args()

    scripts = sorted(glob.glob("train_*_component.py"))

    if args.limit is not None:
        scripts = scripts[:args.limit]

    print(f"Starting execution of {len(scripts)} components...")

    success_count = 0
    failure_count = 0

    for script in scripts:
        print(f"Running {script}...")
        try:
            result = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                success_count += 1
            else:
                failure_count += 1
        except subprocess.TimeoutExpired:
            failure_count += 1
        except Exception as e:
            failure_count += 1

    print(f"Finished run. Success: {success_count}, Failures: {failure_count}")

if __name__ == "__main__":
    main()
