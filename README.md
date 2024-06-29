[中文](README-zh.md)
# SAM_TorchSharp

## Project Introduction

**SAM_TorchSharp** is a project dedicated to exploring the feasibility of artificial intelligence development using .NET Core. Built upon [TorchSharp](https://github.com/dotnet/TorchSharp) and [TorchSharp.PyBridge](https://github.com/sha/incomplete-link), this project focuses on porting the Segment-Anything (SAM) model from Python to the .NET Core ecosystem. SAM, an advanced image segmentation model, is supported in its `sam_vit_b`, `sam_vit_l`, and `sam_vit_h` variants, with successful integration of `mobileSam`. It introduces a versatile model loading mechanism that accepts weight files directly during initialization.

## Features

- **Model Compatibility**: Enables automatic loading of pretrained models by specifying weight file paths upon initialization.
- **Exception Handling**: Addresses issues encountered when attempting to load `sam_vit_b_01ec64.pth`, `sam_vit_h_4b8939.pth`, and `sam_vit_l_0b3195.pth` directly. As a workaround, it's advised to first save the model's state dictionary in Python using `torch.save(model.state_dict(), "sam.pth")`, which can then be seamlessly loaded within this project.
- **.NET Core Integration**: Leverages the cross-platform capabilities of .NET Core, expanding the horizons for AI application development.

## Dependencies

- [TorchSharp](https://github.com/dotnet/TorchSharp): A .NET binding to PyTorch, providing deep learning functionalities.
- [TorchSharp.PyBridge](incomplete-link): An extension for bridging Python and .NET environments, facilitating model migration.

## Installation & Usage

Please refer to the `INSTALL.md` document for detailed installation instructions and environment setup guidelines. After cloning the project, ensure all dependencies are correctly installed and follow the provided instructions to set up your .NET Core environment.

## Contribution Guidelines

We welcome all forms of contributions, including code submissions, bug reports, and documentation improvements. Consult the `CONTRIBUTING.md` file to learn how to get started.

## License

This project is licensed under the [MIT License](LICENSE), encouraging free use, modification, and distribution while preserving original authorship credits.

## Acknowledgments

Special thanks go to:
- The [TorchSharp](https://github.com/dotnet/TorchSharp) team for providing powerful PyTorch bindings to .NET developers.
- The [TorchSharp.PyBridge](incomplete-link) project for simplifying the complexities of cross-language model deployment.
