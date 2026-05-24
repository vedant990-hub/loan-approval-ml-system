import json

with open('loan_approval_prediction.ipynb', 'r') as f:
    nb = json.load(f)

fixed_count = 0
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        if isinstance(source, list):
            new_source = []
            changed = False
            for line in source:
                if isinstance(line, str) and line.startswith('        '):
                    new_source.append(line[8:])
                    changed = True
                else:
                    new_source.append(line)
            if changed:
                cell['source'] = new_source
                fixed_count += 1

with open('loan_approval_prediction.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print(f'✓ Fixed {fixed_count} code cells by removing 8-space indentation')
