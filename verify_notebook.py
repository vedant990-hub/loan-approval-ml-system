import json

with open('loan_approval_prediction.ipynb', 'r') as f:
    data = f.read()

# Check for the exact source content
if '"        print' in data:
    print('❌ ERROR: File still has leading spaces in JSON strings')
    # Show a sample
    idx = data.find('"        print')
    print(f'Sample: {data[idx:idx+50]}')
else:
    print('✓ OK: Leading spaces removed from JSON')

nb = json.loads(data)
code_cells = [c for c in nb['cells'] if c.get('cell_type') == 'code']
print(f'Total code cells: {len(code_cells)}')

# Check Section 4 specifically
for idx, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        if isinstance(source, list) and len(source) > 5:
            if 'Missing values' in str(source):
                print(f'\nSection 4 (cell {idx}):')
                first_line = source[0] if source else ''
                print(f'  First line: {repr(first_line[:50])}')
                if first_line.startswith('print'):
                    print('  ✓ Indentation fixed!')
                else:
                    print('  ✗ Indentation problem persists')
                break
