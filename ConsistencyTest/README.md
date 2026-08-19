# ConsistencyTest CLI

`ConsistencyTest` is the scriptable parity and diagnostic entry point for the SAM TorchSharp migration. It replaces selecting one-off `Main` methods through `StartupObject`.

## Build and smoke test

```powershell
dotnet build .\ConsistencyTest.csproj
dotnet run --project .\ConsistencyTest.csproj -- self-test
dotnet run --project .\ConsistencyTest.csproj -- info
```

## Automated tests

From the repository root:

```powershell
dotnet test .\tests\ConsistencyTest.Tests\ConsistencyTest.Tests.csproj -c Release
```

The test suite covers NPY v1/v2/v3 parsing, float32 round trips, malformed headers and payloads, shape validation, numerical tolerances, and non-finite values.

## Compare NPY outputs

Only little-endian, C-order float32 NPY v1/v2/v3 arrays are accepted.

```powershell
dotnet run --project .\ConsistencyTest.csproj -- compare `
  --expected <python-output.npy> `
  --actual <csharp-output.npy> `
  --atol 1e-5 `
  --rtol 1e-4
```

The command reports `max_abs_error`, `mean_abs_error`, `max_rel_error`, mismatch count, and exits with code `3` when tolerances are exceeded.

## Validate a SAM2 checkpoint

```powershell
dotnet run --project .\ConsistencyTest.csproj -- sam2-checkpoint `
  --variant sam2.1-tiny `
  --checkpoint ..\..\checkpoints\sam2.1_hiera_tiny.pt `
  --output .\sam2.1-tiny-summary.json `
  --strict true
```

Supported variants are `sam2-tiny`, `sam2-small`, `sam2.1-tiny`, and `sam2.1-small`.
The JSON report records loaded, missing, unexpected, and shape-mismatched tensors without invoking Python.

## Run SAM2 image inference

```powershell
dotnet run --project .\ConsistencyTest.csproj -- sam2-image `
  --variant sam2.1-tiny `
  --checkpoint ..\..\checkpoints\sam2.1_hiera_tiny.pt `
  --image .\image.npy `
  --points .\points.npy `
  --labels .\labels.npy `
  --output .\sam2-image-output `
  --multimask true
```

`image.npy` must be a C-order float32 RGB array with shape `[H,W,3]` and values in `[0,1]`.
Point coordinates use original-image pixels in `(x,y)` order. `points.npy` has shape `[N,2]` and
`labels.npy` has shape `[N]`, with `0` for negative and `1` for positive points. A box can be supplied
instead of or together with points through `--box`; its shape is `[4]` or `[2,2]` in
`x0,y0,x1,y1` order. For iterative refinement, `--mask-input` accepts one previous low-resolution
logit mask with shape `[1,256,256]`.

The command writes `masks.npy`, `scores.npy`, `low_res_logits.npy`, and `summary.json`. Masks are
float32 binary arrays by default; use `--return-logits true` for full-resolution mask logits. The
low-resolution logits are always clamped to `[-32,32]` and can be used for refinement.

## Run the SAM3 detector prototype

```powershell
dotnet run --project .\ConsistencyTest.csproj -- sam3-run `
  --checkpoint ..\..\checkpoints\sam3\model.safetensors `
  --image .\image.npy `
  --output ..\..\consistency_test\cs_output `
  --caption "a dog" `
  --device cpu `
  --seed 42 `
  --min-coverage 75
```

`--image` is optional for backward compatibility. When supplied, it must be a finite float32 NCHW
array with shape `[1,3,1008,1008]`, resized and normalized with ImageNet mean
`[0.485,0.456,0.406]` and standard deviation `[0.229,0.224,0.225]`. When omitted, the command uses
the seeded random diagnostic input. The current SAM3 port is a detector prototype: it writes
normalized `pred_boxes.npy` and `pred_logits.npy`, but does not produce final segmentation masks.

The configured native dependency is CPU-only. A `.pt` checkpoint must be converted explicitly to `.bin`; the CLI does not invoke Python or create an implicit multi-gigabyte copy.

## Exit codes

| Code | Meaning |
|---:|---|
| 0 | Success |
| 1 | Runtime failure |
| 2 | Invalid command or arguments |
| 3 | Numerical/coverage validation failure |

The old `Program*.cs` runners remain as migration references but are excluded from compilation.