[中文](README-zh.md)

# SAM_TorchSharp

SAM_TorchSharp ports Meta's Segment Anything model family to .NET 8 with [TorchSharp](https://github.com/dotnet/TorchSharp). The repository contains SAM 1 image segmentation, SAM 2 image/video components and checkpoint tooling, and an experimental SAM 3 text-conditioned detector.

> This is an independent .NET port. Model checkpoints are not included; obtain them from the official model repositories and comply with their licenses and access requirements.

## Current status

| Model | Status | Notes |
| --- | --- | --- |
| SAM 1 | Supported | ViT-H, ViT-L, ViT-B, and MobileSAM/TinyViT builders; point, box, and mask prompts. |
| SAM 2 / 2.1 | In progress and usable | Image prediction, multimask output, checkpoint validation, video/memory components, and Python/.NET parity tools. The scriptable CLI currently exposes tiny and small variants. |
| SAM 3 | Experimental detector prototype | Accepts a text prompt and emits normalized `pred_boxes`/`pred_logits` plus intermediate features. The final pixel-level mask head is not implemented. |

The configured native package is `libtorch-cpu-win-x64`, so the checked-in projects currently support CPU execution on Windows. GPU execution is not exposed by the consistency CLI.

## Requirements

- Windows x64
- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)
- Sufficient memory and disk space for the selected checkpoint
- Python is optional and is only needed to generate or compare parity vectors

Main package versions are declared in `SAMTorchSharp/SAMTorchSharp.csproj`:

- TorchSharp `0.102.6`
- TorchVision `0.102.6`
- TorchSharp.PyBridge `1.4.1`
- libtorch CPU for Windows x64 `2.2.1.1`

## Build and test

```powershell
git clone https://github.com/dongfangzhizhu/SAM_TorchSharp.git
cd SAM_TorchSharp
dotnet restore .\SAM_TorchSharp.sln
dotnet build .\SAM_TorchSharp.sln -c Release --no-restore
dotnet test .\tests\ConsistencyTest.Tests\ConsistencyTest.Tests.csproj -c Release --no-restore
```

The test project covers NPY I/O, CLI parsing, numeric comparison, SAM 2 image-input validation, and SAM 3 preprocessed-image validation. Large generated parity vectors under `testdata/` are intentionally ignored.

## Scriptable consistency CLI

Show all commands:

```powershell
dotnet run --project .\ConsistencyTest\ConsistencyTest.csproj -- --help
```

### SAM 2 image prediction

The image input is a float32 HWC NPY array with shape `[H,W,3]`, containing RGB values in `[0,1]`. Point coordinates use original-image `(x,y)` pixels.

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

Outputs are `masks.npy`, `scores.npy`, `low_res_logits.npy`, and `summary.json`. The command also accepts box and low-resolution mask prompts; see `ConsistencyTest/README.md` for the complete input contract.

The official SAM 2 README/notebook truck case has been exercised with `truck.jpg`, positive point `(500,375)`, and multimask output. The .NET SAM 2.1 small run produced three `1200x1800` masks; the highest predicted IoU score was approximately `0.937` in the validated environment.

### SAM 3 detector prototype

SAM 3 image input must be a finite float32 NCHW NPY array with shape `[1,3,1008,1008]`. Resize to `1008x1008`, scale RGB to `[0,1]`, and normalize using ImageNet mean `[0.485,0.456,0.406]` and standard deviation `[0.229,0.224,0.225]`.

```powershell
dotnet run --project .\ConsistencyTest\ConsistencyTest.csproj -c Release -- sam3-run `
  --checkpoint C:\models\sam3\model.safetensors `
  --image C:\inputs\sam3_image.npy `
  --caption "shoe" `
  --output C:\outputs\sam3 `
  --device cpu `
  --min-coverage 75
```

When `--image` is omitted, the command retains a seeded-random diagnostic input for backward compatibility. With an image it writes `pred_boxes.npy`, `pred_logits.npy`, intermediate feature arrays, and `summary.json`.

The official SAM 3 example image and the `"shoe"` prompt have been run through this .NET path. This is an honest detector-only result: it must not be interpreted as the official SAM 3 instance-mask output. Checkpoint coverage and prediction quality may differ because the prototype does not yet load or implement the complete official model.

## Repository layout

- `SAMTorchSharp/` — model library and checkpoint loaders
- `ConsistencyTest/` — scriptable NPY-based validation and inference CLI
- `tests/ConsistencyTest.Tests/` — xUnit tests
- `tools/` — safetensors comparison and SAM 2 parity-vector utilities
- `WebDemo/` — sample ASP.NET application

## Known limitations

- The repository currently pins a Windows x64 CPU libtorch runtime.
- SAM 3 is not a complete segmentation implementation and does not emit final masks.
- Checkpoint files and generated test vectors are not committed because of their size and licensing.
- PyTorch `.pt` interoperability depends on the relevant loader. The SAM 3 CLI accepts `.safetensors` or an explicitly converted `.bin`; it does not invoke Python implicitly.

## Contributing

Contributions and reproducible bug reports are welcome. Please include the model variant, checkpoint format, runtime, input shapes, and the smallest command that reproduces the issue. Run the Release build and test commands above before opening a pull request.

## License and acknowledgements

This repository is licensed under the [MIT License](LICENSE.txt). Model code and checkpoints may have separate upstream licenses.

Thanks to Meta's Segment Anything teams, the TorchSharp project, and the [TorchSharp.PyBridge](https://github.com/shaltielshmid/TorchSharp.PyBridge) project.
