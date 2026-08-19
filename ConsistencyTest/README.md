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

## Run the SAM3 detector prototype

```powershell
dotnet run --project .\ConsistencyTest.csproj -- sam3-run `
  --checkpoint ..\..\checkpoints\sam3\model.safetensors `
  --output ..\..\consistency_test\cs_output `
  --caption "a dog" `
  --device cpu `
  --seed 42 `
  --min-coverage 75
```

The configured native dependency is CPU-only. A `.pt` checkpoint must be converted explicitly to `.bin`; the CLI does not invoke Python or create an implicit multi-gigabyte copy.

## Exit codes

| Code | Meaning |
|---:|---|
| 0 | Success |
| 1 | Runtime failure |
| 2 | Invalid command or arguments |
| 3 | Numerical/coverage validation failure |

The old `Program*.cs` runners remain as migration references but are excluded from compilation.