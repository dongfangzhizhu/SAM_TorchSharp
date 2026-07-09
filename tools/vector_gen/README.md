# vector_gen

用于 SAM2 / SAM3 迁移到 SAM_TorchSharp (.NET/TorchSharp) 过程中的数值校验测试向量生成工具。

## 用途

每迁移一个模块（比如 Hiera 的某个 Block、MemoryAttention 等），在这里新增一个脚本 `gen_<module_name>.py`，
它需要做三件事：

1. 用固定随机种子构造该模块（Python 原版实现），并用随机权重初始化（`eval()` 模式，关闭 dropout 等）。
2. 构造一个（或多个）随机输入张量，跑一次前向，得到输出张量。
3. 用 `safetensors_io.save_state_dict` 导出模块权重，用 `save_tensor_dict` 分别导出输入和输出。

生成的三个文件建议统一放到 `SAM_TorchSharp/testdata/<module_name>/` 下：
- `weights.safetensors`：模块 state_dict，供 .NET 端 `module.load_safetensors(...)` 加载
- `input.safetensors`：输入张量（key 名与 .NET 测试代码约定一致，例如 "x"、"input_0"）
- `output_py.safetensors`：Python 前向输出（key 名例如 "output"、"masks" 等）

.NET 侧对应的验证代码会：
1. 构造相同结构的 .NET 模块
2. 调用 `load_safetensors("weights.safetensors")` 加载权重
3. 加载 `input.safetensors`，跑前向
4. 将输出保存为 `output_net.safetensors`
5. 用 `tools/SafetensorsCompare` 比较 `output_py.safetensors` 与 `output_net.safetensors`

## 依赖

```
pip install torch safetensors numpy
```

sam2 / sam3 的源码通过 `repo_paths.py` 以 sys.path 方式引入，无需安装。

## 环境变量（可选）

- `SAM2_REPO_PATH`：sam2 仓库根目录，默认 `../../../sam2`（相对本文件）
- `SAM3_REPO_PATH`：sam3 仓库根目录，默认 `../../../sam3`
