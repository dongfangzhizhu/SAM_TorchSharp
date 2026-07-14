import subprocess
r = subprocess.run(['git', 'add', '-A'], cwd=r'D:\SouceCode\python2net\SAMALL\SAM_TorchSharp', capture_output=True, text=True)
r2 = subprocess.run(['git', 'commit', '-m', '清理临时构建脚本'], cwd=r'D:\SouceCode\python2net\SAMALL\SAM_TorchSharp', capture_output=True, text=True)
print(r2.stdout[:300])
