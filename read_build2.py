import os, sys
path = r"D:\SouceCode\python2net\SAMALL\SAM_TorchSharp\build_result.txt"
out_path = r"D:\SouceCode\python2net\SAMALL\SAM_TorchSharp\build_readout.txt"
if os.path.exists(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(content[:10000])
    print("Written to " + out_path)
else:
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("File not found: " + path)
    print("Not found")
