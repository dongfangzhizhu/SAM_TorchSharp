import subprocess, sys
result = subprocess.run(
    [sys.executable, '-c', '''
import subprocess, sys
r = subprocess.run(["dotnet", "build", r"D:\\SouceCode\\python2net\\SAMALL\\SAM_TorchSharp\\SAMTorchSharp\\SAMTorchSharp.csproj"], capture_output=True, text=True, timeout=120)
with open(r"D:\\SouceCode\\python2net\\SAMALL\\SAM_TorchSharp\\build_run.txt", "w", encoding="utf-8") as f:
    f.write("RC=" + str(r.returncode) + "\\n")
    f.write("STDOUT:\\n" + r.stdout[:10000] + "\\n")
    f.write("STDERR:\\n" + r.stderr[:10000] + "\\n")
print("Build done, RC=" + str(r.returncode))
'''],
    capture_output=True, text=True, timeout=130, cwd=r"D:\SouceCode\python2net\SAMALL"
)
print(result.stdout)
if result.stderr:
    print("ERR:", result.stderr)
