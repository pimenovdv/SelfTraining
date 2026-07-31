import os
import glob

scripts = [f.replace('.py', '') for f in glob.glob('train_*.py')]
docs = glob.glob('docs/*.md')
doc_contents = []
for d in docs:
    with open(d, 'r') as f:
        doc_contents.append(f.read())

missing = []
for s in scripts:
    found = False
    for content in doc_contents:
        if s in content:
            found = True
            break
    if not found:
        missing.append(s)

print("Missing docs for:", missing)
