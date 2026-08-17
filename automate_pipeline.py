import os
import glob
import subprocess
import re

def get_next_sequence_number(docs_dir="docs"):
    files = glob.glob(os.path.join(docs_dir, "*.md"))
    max_seq = 0
    for f in files:
        basename = os.path.basename(f)
        match = re.match(r"^(\d{4})_", basename)
        if match:
            max_seq = max(max_seq, int(match.group(1)))
    return max_seq + 1

def generate_markdown(script_name, success, output, seq_num):
    component_name = script_name.replace("train_", "").replace("_component.py", "").replace("_", " ").title()
    status = "Success" if success else "Failure"

    md_content = f"""# Experiment: {component_name} Component Training

**Script:** `{script_name}`
**Status:** {status}

## Objective
Automatically generated report for the training and evaluation of the {component_name} component.

## Methodology
The component was executed via the automated pipeline.

## Results
```text
{output.strip()}
```

## Conclusion
The component execution finished with status: {status}.
"""
    doc_filename = f"{seq_num:04d}_{script_name.replace('.py', '.md')}"
    doc_filepath = os.path.join("docs", doc_filename)

    with open(doc_filepath, "w") as f:
        f.write(md_content)

    return doc_filename

def update_tracking_files(script_name, doc_filename):
    def add_to_file(filepath, line):
        if not os.path.exists(filepath):
            lines = []
        else:
            with open(filepath, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
        if line not in lines:
            lines.append(line)
        lines = list(set(lines))
        lines.sort()
        with open(filepath, "w") as f:
            for l in lines:
                f.write(l + "\n")

    base_name = script_name.replace(".py", "")

    add_to_file("actual_scripts.txt", script_name)
    add_to_file("documented_scripts.txt", script_name)
    add_to_file("available_scripts.txt", script_name)

    add_to_file("scripts.txt", base_name)
    add_to_file("sorted_scripts.txt", base_name)
    add_to_file("all_scripts.txt", base_name)

    add_to_file("all_scripts_dir.txt", f"./{script_name}")

    doc_base = doc_filename.replace(".md", "")
    add_to_file("docs.txt", doc_base)
    add_to_file("doc_scripts.txt", script_name)
    add_to_file("all_docs.txt", f"docs/{doc_filename}")

def main():
    scripts = glob.glob("train_*_component.py")
    docs = glob.glob("docs/*_train_*_component.md")

    documented_scripts = []
    for doc in docs:
        basename = os.path.basename(doc)
        match = re.match(r"^\d{4}_(.*)\.md$", basename)
        if match:
            documented_scripts.append(match.group(1) + ".py")

    unrun_scripts = [s for s in scripts if s not in documented_scripts]

    if not unrun_scripts:
        print("All scripts have been documented.")
        return

    print(f"Found {len(unrun_scripts)} undocumented scripts.")

    for script in unrun_scripts:
        print(f"Running {script}...")
        seq_num = get_next_sequence_number()

        try:
            result = subprocess.run(["python3", script], capture_output=True, text=True, timeout=300)
            success = result.returncode == 0
            output = result.stdout + "\n" + result.stderr if not success else result.stdout
        except subprocess.TimeoutExpired:
            success = False
            output = "Execution timed out after 300 seconds."
        except Exception as e:
            success = False
            output = str(e)

        doc_filename = generate_markdown(script, success, output, seq_num)
        update_tracking_files(script, doc_filename)
        print(f"Generated {doc_filename}")

if __name__ == "__main__":
    main()
