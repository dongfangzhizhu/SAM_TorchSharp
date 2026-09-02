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
  --box .\box.npy `
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

## Validate SAM2 video parity

Generate official Python vectors, then compare every propagated frame with the .NET predictor:

```powershell
python ..\tools\vector_gen\gen_sam2_video_real.py --config sam2.1_hiera_t `
  --checkpoint ..\..\checkpoints\sam2.1_hiera_tiny.pt `
  --video-dir ..\..\sam2\notebooks\videos\bedroom --num-frames 2
dotnet run --project .\ConsistencyTest.csproj -c Release -- sam2-video `
  --variant sam2.1-tiny --checkpoint ..\..\checkpoints\sam2.1_hiera_tiny.pt `
  --vectors ..\testdata\sam2_video_real --atol 0.02 --rtol 1e-3
```

The command strictly loads the official checkpoint and writes `parity_cs.json`. Generated vectors,
reports, and optional copied weights remain under ignored `testdata/` and must not be committed.
The default absolute tolerance is `0.02` to cover expected bfloat16 memory-storage quantization;
the report still records maximum and mean errors for every frame and intermediate memory tensor.

## Run SAM3 text-conditioned inference

Validate an official safetensors or explicitly converted binary checkpoint without running inference:

```powershell
dotnet run --project .\ConsistencyTest.csproj -- sam3-checkpoint `
  --checkpoint ..\..\checkpoints\sam3\model.safetensors `
  --output ..\testdata\sam3-checkpoint `
  --min-coverage 100
```

The command writes `checkpoint-report.json` and exits with code 3 when loadable-tensor coverage is
For SAM 3.1, point `--checkpoint` at `sam3.1_multiplex.bin` under the checkpoint directory. A `.pt`
path is also accepted when its explicitly converted sibling `.bin` exists. Tracker tensors and other explicitly unsupported tensors remain in
`SkippedKeys` and do not reduce coverage.

```powershell
dotnet run --project .\ConsistencyTest.csproj -- sam3-run `
  --checkpoint ..\..\checkpoints\sam3\model.safetensors `
  --image .\image.npy `
  --output ..\..\consistency_test\cs_output `
  --caption "a dog" `
  --device cpu `
  --seed 42 `
  --confidence-threshold 0.5 `
  --max-detections 20 `
  --min-coverage 75
```

`--image` is optional for backward compatibility. When supplied, it must be a finite float32 NCHW
array with shape `[1,3,1008,1008]`, resized and normalized with ImageNet mean
`[0.485,0.456,0.406]` and standard deviation `[0.229,0.224,0.225]`. When omitted, the command uses
the seeded random diagnostic input. For official `.safetensors` checkpoints the command always
accepts optional point and box geometry prompts in model-input pixels. `points.npy` has shape `[N,2]`,
`labels.npy` has shape `[N]` with `0` for negative and `1` for positive points, and `box.npy` has
shape `[4]` in `x0,y0,x1,y1` order. Coordinates must lie within the `1008x1008` model input. The command
writes `checkpoint-report.json` before coverage validation or inference. The report contains the
loaded, missing, skipped, and shape-mismatched keys plus the loadable-tensor coverage. Successful
inference writes the raw normalized `pred_boxes.npy`, `pred_logits.npy`, `pred_masks.npy`, and
`semantic_seg.npy` tensors. It also filters and sorts instances using `--confidence-threshold` and
`--max-detections`, then writes pixel-space `instance_boxes.npy`, sigmoid scores in
`instance_scores.npy`, resized binary `instance_masks.npy`, and a JSON-friendly `detections.json`.
Since the CLI input is already a `1008x1008` preprocessed tensor, these processed outputs use that
model-input resolution.
Before writing outputs, the CLI validates that boxes, scores, instance masks, and semantic masks
have matching shapes and contain only finite values; a NaN or infinity makes the command fail.

The configured native dependency is CPU-only. A `.pt` checkpoint must be converted explicitly to a same-name `.bin` first; passing the `.pt` path only routes to that sibling `.bin`. The CLI does not invoke Python or create an implicit multi-gigabyte copy.

## Exit codes

| Code | Meaning |
|---:|---|
| 0 | Success |
| 1 | Runtime failure |
| 2 | Invalid command or arguments |
| 3 | Numerical/coverage validation failure |

The old `Program*.cs` runners remain as migration references but are excluded from compilation.