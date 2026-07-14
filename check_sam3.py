import os
path = r"D:\SouceCode\python2net\SAMALL\SAM_TorchSharp\build_run.txt"
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
lines = content.split('\n')
# Show only SAM3-related lines
sam3_lines = [l for l in lines if 'Sam3' in l or 'sam3' in l]
print("=== SAM3 Lines ===")
for l in sam3_lines:
    print(l)
if not sam3_lines:
    print("(none - no SAM3 specific warnings/errors)")
