import json

# Read the broken notebook
with open('loan_approval_prediction.ipynb', 'r') as f:
    nb = json.load(f)

# Fix indentation in all code cells
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Work with source as list or string
        if isinstance(cell['source'], list):
            # Remove 8-space indent from each line
            new_source = []
            for line in cell['source']:
                if line.startswith('        ') and line.strip():  # Only if non-empty and indented
                    new_source.append(line[8:])
                else:
                    new_source.append(line)
            cell['source'] = new_source
        else:
            # If it's a string, convert to lines, fix, and back to list
            lines = cell['source'].split('\n')
            new_source = []
            for line in lines:
                if line.startswith('        ') and line.strip():
                    new_source.append(line[8:])
                else:
                    new_source.append(line)
            cell['source'] = new_source

# Write the fixed notebook
with open('loan_approval_prediction.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("✓ Notebook indentation fixed successfully")
