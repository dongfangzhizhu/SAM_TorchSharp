import os
path = r"D:\SouceCode\python2net\SAMALL\SAM_TorchSharp\build_result.txt"
if os.path.exists(path):
    with open(path, 'r', encoding='utf-8') as f:
        print(f.read())
else:
    print("File not found")
