[English](README.md)

# SAM_TorchSharp

SAM_TorchSharp 基于 [TorchSharp](https://github.com/dotnet/TorchSharp)，将 Meta Segment Anything 模型家族移植到 .NET 8。仓库包含 SAM 1 图像分割、SAM 2 图像/视频组件与 checkpoint 工具，以及实验性的 SAM 3 文本条件检测器。

> 本项目是独立的 .NET 移植。仓库不包含模型 checkpoint；请从官方模型仓库获取，并遵守对应的许可证和访问要求。

## 当前状态

| 模型 | 状态 | 说明 |
| --- | --- | --- |
| SAM 1 | 已支持 | 提供 ViT-H、ViT-L、ViT-B 和 MobileSAM/TinyViT builder，支持点、框和 mask 提示。 |
| SAM 2 / 2.1 | 开发中，可使用 | 提供图像预测、多 mask 输出、checkpoint 校验、视频/记忆组件，以及 Python/.NET 一致性工具；脚本化 CLI 当前开放 tiny 和 small 变体。 |
| SAM 3 | 实验性检测器原型 | 接收文本提示，输出归一化 `pred_boxes`/`pred_logits` 和中间特征；尚未实现最终像素级 mask 头。 |

项目当前配置的原生包是 `libtorch-cpu-win-x64`，因此仓库内项目目前面向 Windows CPU 运行；一致性 CLI 未开放 GPU 执行。

## 环境要求

- Windows x64
- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)
- 足够容纳所选 checkpoint 的内存和磁盘空间
- Python 为可选依赖，仅在生成或对比一致性测试向量时需要

主要依赖版本声明在 `SAMTorchSharp/SAMTorchSharp.csproj`：

- TorchSharp `0.102.6`
- TorchVision `0.102.6`
- TorchSharp.PyBridge `1.4.1`
- Windows x64 libtorch CPU `2.2.1.1`

## 构建与测试

```powershell
git clone https://github.com/dongfangzhizhu/SAM_TorchSharp.git
cd SAM_TorchSharp
dotnet restore .\SAM_TorchSharp.sln
dotnet build .\SAM_TorchSharp.sln -c Release --no-restore
dotnet test .\tests\ConsistencyTest.Tests\ConsistencyTest.Tests.csproj -c Release --no-restore
```

测试项目覆盖 NPY 读写、CLI 参数解析、数值比较、SAM 2 图像输入校验和 SAM 3 预处理图像校验。`testdata/` 下体积较大的生成型一致性向量会被 Git 忽略。

## 脚本化一致性 CLI

查看全部命令：

```powershell
dotnet run --project .\ConsistencyTest\ConsistencyTest.csproj -- --help
```

### SAM 2 图像预测

图像输入是 float32 HWC NPY 数组，形状为 `[H,W,3]`，RGB 值范围为 `[0,1]`。点坐标使用原图像素 `(x,y)`。

```powershell
dotnet run --project .\ConsistencyTest\ConsistencyTest.csproj -c Release -- sam2-image `
  --variant sam2.1-small `
  --checkpoint C:\models\sam2.1_hiera_small.pt `
  --image C:\inputs\image.npy `
  --points C:\inputs\points.npy `
  --labels C:\inputs\labels.npy `
  --output C:\outputs\sam2 `
  --multimask true
```

输出包括 `masks.npy`、`scores.npy`、`low_res_logits.npy` 和 `summary.json`。命令也支持框提示和低分辨率 mask 提示；完整输入约定请参阅 `ConsistencyTest/README.md`。

已使用 SAM 2 官方 README/notebook 的 truck 用例进行验证：`truck.jpg`、正点 `(500,375)`、多 mask 输出。在已验证环境中，.NET SAM 2.1 small 生成了三个 `1200x1800` mask，最高预测 IoU 分数约为 `0.937`。

### SAM 3 检测器原型

SAM 3 图像输入必须是有限值 float32 NCHW NPY 数组，形状为 `[1,3,1008,1008]`。先缩放到 `1008x1008`，将 RGB 缩放到 `[0,1]`，再使用 ImageNet mean `[0.485,0.456,0.406]` 和 std `[0.229,0.224,0.225]` 标准化。

```powershell
dotnet run --project .\ConsistencyTest\ConsistencyTest.csproj -c Release -- sam3-run `
  --checkpoint C:\models\sam3\model.safetensors `
  --image C:\inputs\sam3_image.npy `
  --caption "shoe" `
  --output C:\outputs\sam3 `
  --device cpu `
  --min-coverage 75
```

省略 `--image` 时，为向后兼容，命令仍使用固定 seed 的随机诊断输入。提供图片时会写出 `pred_boxes.npy`、`pred_logits.npy`、中间特征数组和 `summary.json`。

已使用 SAM 3 官方示例图片和 `"shoe"` 提示运行该 .NET 路径。必须如实说明：这只是 detector-only 结果，不能当作官方 SAM 3 实例 mask 输出。由于原型尚未加载或实现完整官方模型，checkpoint 覆盖率和预测质量可能存在差异。

## 仓库结构

- `SAMTorchSharp/`：模型库和 checkpoint loader
- `ConsistencyTest/`：基于 NPY 的脚本化校验和推理 CLI
- `tests/ConsistencyTest.Tests/`：xUnit 测试
- `tools/`：safetensors 对比和 SAM 2 一致性向量工具
- `WebDemo/`：ASP.NET 示例应用

## WebDemo

WebDemo 可分别配置和测试 SAM、SAM 2、SAM 2.1 点/框分割，以及 SAM 3、SAM 3.1 文本条件 detector-only 推理。模型在首次请求时独立惰性加载；缺少某个 checkpoint 不会阻止网站启动。页面中的“测试全部可用模型”会按顺序实际请求每个已配置模型并逐项报告结果。

```powershell
dotnet run --project .\WebDemo\WebDemo.csproj -c Release -- `
  --Models:Sam2:Directory=C:\models\sam2 `
  --Models:Sam2:Name=sam2_hiera_tiny
```

在 `appsettings.json` 的 `Models:Sam`、`Models:Sam2`、`Models:Sam21`、`Models:Sam3`、`Models:Sam31` 下分别设置 `Directory` 和 `Name`。`Name` 可带扩展名；不带扩展名时会探测该模型支持的 checkpoint 格式。例如 SAM 2 使用 `sam2_hiera_tiny`，SAM 2.1 使用 `sam2.1_hiera_tiny`。任一字段为空时，前端会将对应模型显示为“未配置”并禁用。也可使用环境变量，例如 `$env:Models__Sam31__Directory='C:\models\sam3.1'`。checkpoint 不会复制到构建输出或提交到 Git。

## 已知限制

- 仓库当前固定使用 Windows x64 CPU libtorch runtime。
- SAM 3 和 SAM 3.1 共用当前实验性 detector 架构，但使用各自的模型实例和 checkpoint；它们不是完整分割实现，不能输出最终 mask。
- checkpoint 和生成型测试向量因体积及许可证原因不提交到仓库。
- PyTorch `.pt` 互操作取决于对应 loader。SAM 3 CLI 接受 `.safetensors` 或显式转换的 `.bin`，不会隐式调用 Python。

## 参与贡献

欢迎提交代码和可复现的问题报告。请提供模型变体、checkpoint 格式、运行环境、输入形状和最小复现命令。提交 PR 前请运行上述 Release 构建和测试命令。

## 许可证与致谢

本仓库使用 [MIT License](LICENSE.txt)。模型代码和 checkpoint 可能适用独立的上游许可证。

感谢 Meta Segment Anything 团队、TorchSharp 项目和 [TorchSharp.PyBridge](https://github.com/shaltielshmid/TorchSharp.PyBridge) 项目。
