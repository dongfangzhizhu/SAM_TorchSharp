import subprocess
r = subprocess.run(['git', 'add', '-A'], cwd=r'D:\SouceCode\python2net\SAMALL\SAM_TorchSharp', capture_output=True, text=True)
print('add rc:', r.returncode)
r2 = subprocess.run(['git', 'commit', '-m', '阶段3.16: 实现SAM3完整推理管线集成 - Sam3ImagePredictor调用Sam3Base.RunInferenceFromFeatures完成端到端推理'], cwd=r'D:\SouceCode\python2net\SAMALL\SAM_TorchSharp', capture_output=True, text=True)
print('commit rc:', r2.returncode)
print('stdout:', r2.stdout[:500])
print('stderr:', r2.stderr[:500])
