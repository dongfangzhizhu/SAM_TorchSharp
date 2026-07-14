@echo off
cd /d D:\SouceCode\python2net\SAMALL\SAM_TorchSharp
dotnet build SAMTorchSharp/SAMTorchSharp.csproj > build_result.txt 2>&1
type build_result.txt
