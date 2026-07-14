import os
path = r"D:\SouceCode\python2net\SAMALL\SAM_TorchSharp\build_run.txt"
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
# Show only errors and the summary
lines = content.split('\n')
errors = [l for l in lines if 'error ' in l.lower() or '错误' in l]
print("=== ERRORS ===")
for e in errors:
    print(e)
print("\n=== LAST 10 LINES ===")
for l in lines[-10:]:
    print(l)
print("\n=== TOTAL WARNINGS ===")
warnings = [l for l in lines if 'warning' in l.lower() or '警告' in l]
print(len(warnings), "warnings")
