[English](README.md)
# SAM_TorchSharp

## 项目简介

**SAM_TorchSharp** 是一个旨在探索.NET Core平台下人工智能开发潜力的项目。本项目基于[TorchSharp](https://github.com/dotnet/TorchSharp)和[TorchSharp.PyBridge](https://github.com/sha/未完成链接)实现，专注于将**Segment-Anything (SAM)** 模型的Python实现移植到.NET Core环境中。SAM是一种先进的图像分割模型，本移植工作支持`sam_vit_b`, `sam_vit_l`, 和 `sam_vit_h`版本，并成功集成了`mobileSam`，提供了高度灵活的模型加载机制。

## 特性

- **模型兼容性**: 支持直接通过指定weight文件路径在初始化时自动加载预训练模型。
- **异常处理**: 目前，直接加载`sam_vit_b_01ec64.pth`, `sam_vit_h_4b8939.pth`, `sam_vit_l_0b3195.pth`会遇到异常。作为解决方案，建议先在Python环境中使用`torch.save(model.state_dict(),"sam.pth")`保存模型状态字典，然后在本项目中加载这个通用的`sam.pth`文件以避免加载问题。
- **.NET Core集成**: 充分利用.NET Core跨平台特性，拓展AI应用的开发边界。

## 依赖项

- [TorchSharp](https://github.com/dotnet/TorchSharp): .NET绑定到PyTorch的库，提供深度学习功能。
- [TorchSharp.PyBridge](https://github.com/shaltielshmid/TorchSharp.PyBridge): 用于加载模型的权重文件，支持pth,pt以及safetensors文件格式

## 安装与运行

请参照`INSTALL.md`文件以获取详细的安装指导及环境配置说明。项目克隆后，确保已正确安装所有依赖，并按照指引设置好.NET Core环境。

## 贡献指南

我们欢迎任何形式的贡献，无论是代码提交、bug报告还是文档改进。请查阅`CONTRIBUTING.md`了解如何开始。

## 协议

本项目遵循[MIT License](LICENSE)。鼓励自由使用、修改和分享，但请保留原作者版权信息。

## 致谢

特别感谢以下项目及其团队：
- [TorchSharp](https://github.com/dotnet/TorchSharp)团队，为.NET开发者提供了强大的PyTorch接口。
- [TorchSharp.PyBridge](https://github.com/shaltielshmid/TorchSharp.PyBridge)项目，简化了跨语言模型部署的复杂度。
