// Copyright (c) Sapiens AI. All rights reserved.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SAMTorchSharp.Modeling.Sam3;

/// <summary>
/// Checkpoint loader for SAM3 models.
/// 
/// Two loading strategies:
/// 1. Safetensors: Pure C# parsing of the binary format, no Python dependency.
/// 2. TorchSave (.pt): Uses a Python helper script to extract tensors to .npy files,
///    then loads them via numpy->base64->Tensor conversion.
/// 
/// Usage:
///   Sam3CheckpointLoader.LoadCheckpoint(model, "path/to/checkpoint");
/// </summary>
public class Sam3CheckpointLoader
{
    /// <summary>
    /// Auto-detect and load checkpoint from file.
    /// </summary>
    public static void LoadCheckpoint(Sam3Base model, string checkpointPath, Device? device = null, bool strict = false)
    {
        device ??= model.device;

        if (IsSafetensorsFile(checkpointPath))
        {
            LoadSafetensors(model, checkpointPath, device, strict);
        }
        else
        {
            LoadTorchSave(model, checkpointPath, device, strict);
        }
    }

    // ================================================================
    // Safetensors Loader (Pure C#, no Python dependency)
    // ================================================================

    /// <summary>
    /// Load weights from a safetensors file directly (no Python dependency).
    /// Parses the binary format, reads tensor data, converts to TorchSharp tensors.
    /// </summary>
    public static void LoadSafetensors(Sam3Base model, string safetensorsPath, Device? device = null, bool strict = false)
    {
        device ??= model.device;

        using var stream = File.OpenRead(safetensorsPath);

        // Read header length (8 bytes, little-endian uint64)
        var lengthBytes = new byte[8];
        ReadExactly(stream, lengthBytes, 0, 8);
        var jsonLength = BitConverter.ToUInt64(lengthBytes, 0);

        // Read header JSON
        var jsonBytes = new byte[jsonLength];
        ReadExactly(stream, jsonBytes, 0, (int)jsonLength);
        var header = System.Text.Encoding.UTF8.GetString(jsonBytes);
        var index = JsonSerializer.Deserialize<Dictionary<string, SafetensorsEntryData>>(header)
            ?? throw new InvalidOperationException("Failed to parse safetensors index.");

        long dataOffset = stream.Position;
        var modelParams = CollectModelParameters(model);
        var loadedKeys = new HashSet<string>();
        var skippedKeys = new List<string>();

        foreach (var (pyKey, entry) in index)
        {
            if (pyKey == "__metadata__") continue;

            var strippedKey = StripSafetensorsPrefix(pyKey);
            var torchPath = MapPyTorchKeyToTorchSharp(strippedKey);
            if (string.IsNullOrEmpty(torchPath))
                continue;

            if (!modelParams.TryGetValue(torchPath, out var param))
            {
                if (strict)
                    Console.WriteLine($"[Safetensors] STRICT: No matching param for {pyKey} -> {torchPath}");
                continue;
            }

            var targetShape = param.size().ToArray();
            var tensorShape = entry.Shape;

            // Seek to tensor data
            stream.Position = dataOffset + entry.DataOffsets[0];
            var byteCount = (int)(entry.DataOffsets[1] - entry.DataOffsets[0]);
            var rawData = new byte[byteCount];
            ReadExactly(stream, rawData, 0, byteCount);

            var dtype = ConvertToTorchDType(entry.Dtype);

            // Convert raw bytes to Tensor
            var loadedTensor = BytesToTensor(rawData, tensorShape, dtype);

            if (loadedTensor.size().ToArray().SequenceEqual(targetShape))
            {
                param.copy_(loadedTensor.to(device));
                loadedKeys.Add(pyKey);
            }
            else
            {
                // Try transposing for 2D weights
                var transposed = TryTranspose(loadedTensor);
                if (transposed.size().ToArray().SequenceEqual(targetShape))
                {
                    param.copy_(transposed.to(device));
                    loadedKeys.Add(pyKey);
                }
                else
                {
                    skippedKeys.Add($"{pyKey}: shape {string.Join("x", tensorShape)} != expected {string.Join("x", targetShape)}");
                }
            }
        }

        Console.WriteLine($"[Safetensors] Loaded {loadedKeys.Count}/{index.Count - 1} tensors.");
        if (skippedKeys.Count > 0)
        {
            Console.WriteLine($"[Safetensors] Skipped ({skippedKeys.Count}):");
            foreach (var k in skippedKeys.Take(20))
                Console.WriteLine($"  {k}");
        }
    }

    // ================================================================
    // TorchSave (.pt) Loader (Uses Python for tensor extraction)
    // ================================================================

    /// <summary>
    /// Load weights from a torch.save (.pt) checkpoint file.
    /// Uses Python to extract tensors to a temporary directory, then loads them.
    /// </summary>
    public static void LoadTorchSave(Sam3Base model, string checkpointPath, Device? device = null, bool strict = false)
    {
        device ??= model.device;
        var tempDir = Path.Combine(Path.GetTempPath(), "sam3_extract_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);

        try
        {
            // Step 1: Use Python to extract tensors and save as .npy + manifest
            var manifestPath = ExtractTensorsToNpy(checkpointPath, tempDir, isSafetensors: false);

            // Step 2: Load from manifest
            LoadFromNpyManifest(model, manifestPath, device, strict, isSafetensors: false);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                try { Directory.Delete(tempDir, true); } catch { }
        }
    }

    // ================================================================
    // Key Mapping
    // ================================================================

    private static string StripSafetensorsPrefix(string key)
    {
        if (key.StartsWith("detector_model."))
            return key.Substring("detector_model.".Length);
        if (key.StartsWith("tracker_model."))
            return key.Substring("tracker_model.".Length);
        if (key.StartsWith("tracker_neck."))
            return key.Substring("tracker_neck.".Length);
        return key;
    }

    private static string StripTorchSavePrefix(string key)
    {
        if (key.StartsWith("detector."))
            return key.Substring("detector.".Length);
        if (key.StartsWith("tracker."))
            return "inst_interactive_predictor.model." + key.Substring("tracker.".Length);
        return key;
    }

    /// <summary>
    /// Map a PyTorch key (without prefix) to a TorchSharp module path within Sam3Base.
    /// </summary>
    private static string MapPyTorchKeyToTorchSharp(string key)
    {
        // Vision encoder: vision_encoder.xxx -> vl_backbone.vision_backbone.vision_backbone.xxx
        if (key.StartsWith("vision_encoder."))
            return $"vl_backbone.vision_backbone.vision_backbone.{key.Substring("vision_encoder.".Length)}";

        // Text encoder: text_encoder.xxx -> text_encoder.encoder.xxx
        if (key.StartsWith("text_encoder."))
            return $"text_encoder.encoder.{key.Substring("text_encoder.".Length)}";

        // Text projection: text_projection.xxx -> text_encoder.resizer.xxx
        if (key.StartsWith("text_projection."))
            return $"text_encoder.resizer.{key.Substring("text_projection.".Length)}";

        // DETR encoder: detr_encoder.xxx -> transformer_encoder.xxx
        if (key.StartsWith("detr_encoder."))
            return $"transformer_encoder.{key.Substring("detr_encoder.".Length)}";

        // DETR decoder: detr_decoder.xxx -> transformer_decoder.xxx
        if (key.StartsWith("detr_decoder."))
            return $"transformer_decoder.{key.Substring("detr_decoder.".Length)}";

        // Mask decoder: mask_decoder.xxx -> segmentation_head.mask_decoder.xxx
        if (key.StartsWith("mask_decoder."))
            return $"segmentation_head.mask_decoder.{key.Substring("mask_decoder.".Length)}";

        // Dot product scoring: dot_product_scoring.xxx -> dot_prod_scoring.xxx
        if (key.StartsWith("dot_product_scoring."))
            return $"dot_prod_scoring.{key.Substring("dot_product_scoring.".Length)}";

        // Geometry encoder: geometry_encoder.xxx -> geometry_encoder.xxx (direct)
        if (key.StartsWith("geometry_encoder."))
            return $"geometry_encoder.{key.Substring("geometry_encoder.".Length)}";

        // Neck (FPN) keys
        if (key.StartsWith("neck.") || key.StartsWith("fpn_layers."))
            return $"vl_backbone.vision_backbone.neck.{key}";

        // Instance query
        if (key.StartsWith("instance_query."))
            return $"transformer_decoder.{key.Substring("instance_query.".Length)}";

        return string.Empty;
    }

    // ================================================================
    // Utility Methods
    // ================================================================

    private static Dictionary<string, Parameter> CollectModelParameters(Module module)
    {
        var result = new Dictionary<string, Parameter>();
        CollectParamsRecursive(module, "", result);
        return result;
    }

    private static void CollectParamsRecursive(Module mod, string prefix, Dictionary<string, Parameter> result)
    {
        foreach (var (name, param) in mod.named_parameters(recurse: false))
        {
            var fullPath = string.IsNullOrEmpty(prefix) ? name : $"{prefix}.{name}";
            result[fullPath] = param;
        }

        foreach (var (childName, childMod) in mod.named_children())
        {
            var childPrefix = string.IsNullOrEmpty(prefix) ? childName : $"{prefix}.{childName}";
            CollectParamsRecursive(childMod, childPrefix, result);
        }
    }

    private static bool IsSafetensorsFile(string path)
    {
        try
        {
            using var stream = File.OpenRead(path);
            var lengthBytes = new byte[8];
            if (stream.Read(lengthBytes, 0, 8) != 8) return false;
            var jsonLength = BitConverter.ToUInt64(lengthBytes, 0);
            if (jsonLength == 0 || jsonLength > 100_000_000) return false;
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static void ReadExactly(Stream stream, byte[] buffer, int offset, int count)
    {
        int total = 0;
        while (total < count)
        {
            int read = stream.Read(buffer, offset + total, count - total);
            if (read == 0) throw new InvalidDataException($"Unexpected end of stream at byte {offset + total}");
            total += read;
        }
    }

    private static Tensor TryTranspose(Tensor t)
    {
        if (t.dim() == 2)
            return t.transpose(0, 1);
        return t;
    }

    private static ScalarType ConvertToTorchDType(string dataType)
    {
        return dataType switch
        {
            "F64" or "float64" => ScalarType.Float64,
            "F32" or "float32" => ScalarType.Float32,
            "F16" or "float16" => ScalarType.Float16,
            "BF16" or "bfloat16" => ScalarType.BFloat16,
            "I64" or "int64" => ScalarType.Int64,
            "I32" or "int32" => ScalarType.Int32,
            "I16" or "int16" => ScalarType.Int16,
            "I8" or "int8" => ScalarType.Int8,
            "U8" or "uint8" => ScalarType.Byte,
            "BOOL" or "bool" => ScalarType.Bool,
            _ => ScalarType.Float32
        };
    }

    // ================================================================
    // Bytes to Tensor Conversion
    // ================================================================

    /// <summary>
    /// Convert raw bytes to a TorchSharp Tensor.
    /// Uses Python numpy to handle dtype conversion reliably.
    /// </summary>
    private static Tensor BytesToTensor(byte[] raw, long[] shape, ScalarType dtype)
    {
        var dtypeStr = TorchDtypeToStr(dtype);
        var shapeStr = string.Join(",", shape.Select(s => s.ToString()));

        var tmpFile = Path.Combine(Path.GetTempPath(), "tensor_raw_" + Guid.NewGuid().ToString("N") + ".bin");
        File.WriteAllBytes(tmpFile, raw);

        var tmpPtFile = Path.Combine(Path.GetTempPath(), "tensor_tmp_" + Guid.NewGuid().ToString("N") + ".pt");

        try
        {
            var escapedTmpFile = tmpFile.Replace("\\", "\\\\");
            var escapedPtFile = tmpPtFile.Replace("\\", "\\\\");

            var script = $@"
import torch
import numpy as np
arr = np.frombuffer(open(r'{escapedTmpFile}', 'rb').read(), dtype=np.{dtypeStr})
t = torch.from_numpy(arr).reshape({shapeStr})
torch.save(t.cpu(), r'{escapedPtFile}')
";
            var tmpScript = Path.Combine(Path.GetTempPath(), "bytes_to_tensor.py");
            File.WriteAllText(tmpScript, script);

            using var proc = System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
            {
                FileName = "python",
                Arguments = tmpScript,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            });
            proc.WaitForExit();

            // Find the saved .pt file
            var savedFiles = Directory.GetFiles(Path.GetTempPath(), "tensor_tmp_*.pt");
            if (savedFiles.Length > 0)
            {
                var result = torch.load(savedFiles[0]);
                foreach (var f in savedFiles) File.Delete(f);
                return result as Tensor ?? torch.zeros(0);
            }

            return torch.zeros(0);
        }
        finally
        {
            if (File.Exists(tmpFile)) File.Delete(tmpFile);
            // Cleanup any leftover temp files
            foreach (var f in Directory.GetFiles(Path.GetTempPath(), "tensor_tmp_*.pt"))
                try { File.Delete(f); } catch { }
        }
    }

    private static string TorchDtypeToStr(ScalarType dtype)
    {
        return dtype switch
        {
            ScalarType.Float64 => "float64",
            ScalarType.Float32 => "float32",
            ScalarType.Float16 => "float16",
            ScalarType.BFloat16 => "float32",  // numpy doesn't have BF16, approximate with float32
            ScalarType.Int64 => "int64",
            ScalarType.Int32 => "int32",
            ScalarType.Int16 => "int16",
            ScalarType.Int8 => "int8",
            ScalarType.Byte => "uint8",
            ScalarType.Bool => "bool",
            _ => "float32"
        };
    }

    // ================================================================
    // Python Tensor Extraction (for torch.save files)
    // ================================================================

    /// <summary>
    /// Use Python to extract tensors from a checkpoint file and save as .npy + manifest.
    /// </summary>
    private static string ExtractTensorsToNpy(string sourcePath, string outputDir, bool isSafetensors)
    {
        var escapedPath = sourcePath.Replace("\\", "\\\\").Replace("'", "\\'");
        var escapedOutDir = outputDir.Replace("\\", "\\\\").Replace("'", "\\'");

        string script;
        if (isSafetensors)
        {
            script = $@"
import numpy as np
import json
import struct
import os

path = '{escapedPath}'
out_dir = '{escapedOutDir}'
os.makedirs(out_dir, exist_ok=True)

with open(path, 'rb') as f:
    length = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(length))

dtype_map = {{'F64':'float64','F32':'float32','F16':'float16','BF16':'bfloat16',
             'I64':'int64','I32':'int32','I16':'int16','I8':'int8','U8':'uint8','BOOL':'bool'}}

index = {{}}
for key, val in header.items():
    if key == '__metadata__':
        continue
    np_dtype = dtype_map.get(val['dtype'], 'float32')
    shape = val['shape']
    off0, off1 = val['data_offsets'][0], val['data_offsets'][1]
    f.seek(8 + length + off0)
    raw = f.read(off1 - off0)
    arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
    safe_key = key.replace('.', '_').replace('-', '_')
    npy_file = f'{{safe_key}}.npy'
    np.save(os.path.join(out_dir, npy_file), arr)
    index[key] = {{'file': npy_file, 'dtype': str(arr.dtype), 'shape': list(arr.shape)}}

with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
    json.dump(index, f)
print(f'Extracted {{len(index)}} tensors')
";
        }
        else
        {
            script = $@"
import torch
import numpy as np
import json
import os

path = '{escapedPath}'
out_dir = '{escapedOutDir}'
os.makedirs(out_dir, exist_ok=True)

ckpt = torch.load(path, map_location='cpu', weights_only=True)
if 'model' in ckpt and isinstance(ckpt['model'], dict):
    actual = ckpt['model']
else:
    actual = ckpt

index = {{}}
for key, tensor in actual.items():
    if isinstance(tensor, torch.Tensor):
        arr = tensor.cpu().numpy()
        safe_key = key.replace('.', '_').replace('-', '_')
        npy_file = f'{{safe_key}}.npy'
        np.save(os.path.join(out_dir, npy_file), arr)
        index[key] = {{'file': npy_file, 'dtype': str(arr.dtype), 'shape': list(arr.shape)}}

with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
    json.dump(index, f)
print(f'Extracted {{len(index)}} tensors')
";
        }

        var tmpScript = Path.Combine(Path.GetTempPath(), "sam3_extract_" + Guid.NewGuid().ToString("N") + ".py");
        File.WriteAllText(tmpScript, script);

        try
        {
            using var proc = System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
            {
                FileName = "python",
                Arguments = tmpScript,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            });

            proc.WaitForExit();
            var stdout = proc.StandardOutput.ReadToEnd();
            var stderr = proc.StandardError.ReadToEnd();

            if (proc.ExitCode != 0)
            {
                throw new InvalidOperationException($"Python extraction failed (exit {proc.ExitCode}):\n{stderr}");
            }

            Console.WriteLine(stdout.Trim());
            return Path.Combine(outputDir, "manifest.json");
        }
        finally
        {
            if (File.Exists(tmpScript)) File.Delete(tmpScript);
        }
    }

    /// <summary>
    /// Load weights from a JSON manifest of .npy files.
    /// </summary>
    private static void LoadFromNpyManifest(Sam3Base model, string manifestPath, Device device, bool strict, bool isSafetensors)
    {
        var manifest = JsonSerializer.Deserialize<Dictionary<string, NpyEntry>>(
                File.ReadAllText(manifestPath))
            ?? throw new InvalidOperationException("Failed to parse NPY manifest.");

        var modelParams = CollectModelParameters(model);
        var loadedKeys = new HashSet<string>();
        var skippedKeys = new List<string>();
        var baseDir = Path.GetDirectoryName(manifestPath)!;

        foreach (var (pyKey, npyData) in manifest)
        {
            var strippedKey = isSafetensors ? StripSafetensorsPrefix(pyKey) : StripTorchSavePrefix(pyKey);
            var torchPath = MapPyTorchKeyToTorchSharp(strippedKey);
            if (string.IsNullOrEmpty(torchPath))
                continue;

            if (!modelParams.TryGetValue(torchPath, out var param))
            {
                if (strict)
                    Console.WriteLine($"[NPY Manifest] STRICT: No matching param for {pyKey} -> {torchPath}");
                continue;
            }

            var targetShape = param.size().ToArray();
            var npyPath = Path.Combine(baseDir, npyData.File);

            if (!File.Exists(npyPath))
            {
                skippedKeys.Add($"{pyKey}: .npy file not found: {npyPath}");
                continue;
            }

            // Load .npy and convert to Tensor
            var tensor = LoadNpyToTensor(npyPath, device);

            if (tensor.size().ToArray().SequenceEqual(targetShape))
            {
                param.copy_(tensor);
                loadedKeys.Add(pyKey);
            }
            else
            {
                var transposed = TryTranspose(tensor);
                if (transposed.size().ToArray().SequenceEqual(targetShape))
                {
                    param.copy_(transposed);
                    loadedKeys.Add(pyKey);
                }
                else
                {
                    skippedKeys.Add($"{pyKey}: shape mismatch {string.Join("x", tensor.size())} vs {string.Join("x", targetShape)}");
                }
            }
        }

        Console.WriteLine($"[NPY Manifest] Loaded {loadedKeys.Count}/{manifest.Count} tensors.");
        if (skippedKeys.Count > 0)
        {
            Console.WriteLine($"[NPY Manifest] Skipped ({skippedKeys.Count}):");
            foreach (var k in skippedKeys.Take(10))
                Console.WriteLine($"  {k}");
        }
    }

    private static Tensor LoadNpyToTensor(string npyPath, Device device)
    {
        var tmpPtFile = Path.Combine(Path.GetTempPath(), "npy_tmp_" + Guid.NewGuid().ToString("N") + ".pt");
        var escapedNpyPath = npyPath.Replace("\\", "\\\\");
        var escapedPtFile = tmpPtFile.Replace("\\", "\\\\");

        var script = $@"
import torch
import numpy as np
arr = np.load(r'{escapedNpyPath}')
t = torch.from_numpy(arr).contiguous()
torch.save(t.cpu(), r'{escapedPtFile}')
";
        var tmpScript = Path.Combine(Path.GetTempPath(), "load_npy.py");
        File.WriteAllText(tmpScript, script);

        try
        {
            using var proc = System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
            {
                FileName = "python",
                Arguments = tmpScript,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            });
            proc.WaitForExit();

            var savedFiles = Directory.GetFiles(Path.GetTempPath(), "npy_tmp_*.pt");
            if (savedFiles.Length > 0)
            {
                var result = torch.load(savedFiles[0]);
                foreach (var f in savedFiles) File.Delete(f);
                return result as Tensor ?? torch.zeros(0, device: device);
            }
            return torch.zeros(0, device: device);
        }
        finally
        {
            if (File.Exists(tmpScript)) File.Delete(tmpScript);
            foreach (var f in Directory.GetFiles(Path.GetTempPath(), "npy_tmp_*.pt"))
                try { File.Delete(f); } catch { }
        }
    }

    // ================================================================
    // Data Classes
    // ================================================================

    private class SafetensorsEntryData
    {
        public string Dtype { get; init; } = "";
        public long[] Shape { get; init; } = Array.Empty<long>();
        public long[] DataOffsets { get; init; } = Array.Empty<long>();
    }

    private class NpyEntry
    {
        public string File { get; init; } = "";
        public string Dtype { get; init; } = "";
        public long[] Shape { get; init; } = Array.Empty<long>();
    }
}
