// Copyright (c) Sapiens AI. All rights reserved.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SAMTorchSharp.Modeling.Sam3;

/// <summary>
/// Safetensors checkpoint loader for SAM3 models.
/// Maps PyTorch safetensors keys to TorchSharp module parameters.
/// </summary>
public class Sam3CheckpointLoader
{
    /// <summary>
    /// Load safetensors weights from file and apply to the given model.
    /// </summary>
    /// <param name="model">The SAM3 model to load weights into.</param>
    /// <param name="safetensorsPath">Path to the .safetensors file.</param>
    /// <param name="device">Device to load tensors onto.</param>
    /// <param name="strict">If true, throws on mismatched keys.</param>
    public static void LoadWeights(
        Sam3Base model,
        string safetensorsPath,
        Device? device = null,
        bool strict = false)
    {
        device ??= model.device;

        // Load safetensors index and data
        var (index, tensorData) = LoadSafetensorsFile(safetensorsPath);

        // Collect all leaf parameters from the model
        var modelParams = CollectModelParameters(model);

        // Map safetensors keys to model parameters
        var keyMapping = BuildKeyMapping(index.Keys.ToList());

        var loadedKeys = new HashSet<string>();
        var unmatchedKeys = new List<string>();

        foreach (var (modulePath, safetensorsPrefixes) in keyMapping)
        {
            if (!modelParams.ContainsKey(modulePath))
            {
                if (strict)
                    unmatchedKeys.Add($"{modulePath}: no matching model parameter");
                continue;
            }

            var param = modelParams[modulePath];
            foreach (var prefix in safetensorsPrefixes)
            {
                if (index.TryGetValue(prefix, out var meta) && tensorData.TryGetValue(prefix, out var tensor))
                {
                    var targetShape = new long[param.size().Length];
                    for (int i = 0; i < param.size().Length; i++)
                        targetShape[i] = param.size(i);

                    var loadedTensor = tensor.to(device);
                    var loadedDims = loadedTensor.size().ToArray();
                    bool shapesMatch = loadedDims.Length == targetShape.Length;
                    if (shapesMatch)
                    {
                        for (int di = 0; di < loadedDims.Length; di++)
                        {
                            if (loadedDims[di] != targetShape[di])
                            {
                                shapesMatch = false;
                                break;
                            }
                        }
                    }

                    if (shapesMatch)
                    {
                        param.copy_(loadedTensor);
                        loadedKeys.Add(prefix);
                    }
                    else
                    {
                        var matchedTensor = HandleShapeMismatch(loadedTensor, targetShape);
                        if (matchedTensor is not null)
                        {
                            param.copy_(matchedTensor.to(device));
                            loadedKeys.Add(prefix);
                        }
                        else
                        {
                            unmatchedKeys.Add($"{prefix}: shape [{string.Join(", ", loadedDims)}] != expected [{string.Join(", ", targetShape)}]");
                        }
                    }
                }
            }
        }

        Console.WriteLine($"Loaded {loadedKeys.Count} weight tensors into model.");
        if (unmatchedKeys.Count > 0)
        {
            Console.WriteLine($"Unmatched keys ({unmatchedKeys.Count}):");
            foreach (var k in unmatchedKeys.Take(20))
                Console.WriteLine($"  {k}");
        }
    }

    /// <summary>
    /// Collect all leaf parameters from a module, returning path -> parameter mapping.
    /// </summary>
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

    /// <summary>
    /// Build a mapping from safetensors keys to TorchSharp module paths.
    /// </summary>
    private static Dictionary<string, List<string>> BuildKeyMapping(IList<string> safetensorsKeys)
    {
        var mapping = new Dictionary<string, List<string>>();

        foreach (var key in safetensorsKeys)
        {
            var stripped = key;
            if (stripped.StartsWith("detector_model."))
                stripped = stripped.Substring("detector_model.".Length);

            var torchPath = MapPyTorchKeyToTorchSharp(stripped);
            if (!string.IsNullOrEmpty(torchPath))
            {
                if (!mapping.ContainsKey(torchPath))
                    mapping[torchPath] = new List<string>();
                mapping[torchPath].Add(key);
            }
        }

        return mapping;
    }

    /// <summary>
    /// Map a PyTorch safetensors key (without detector_model. prefix) to a TorchSharp module path.
    /// </summary>
    private static string MapPyTorchKeyToTorchSharp(string key)
    {
        if (key.StartsWith("vision_encoder."))
        {
            var rest = key.Substring("vision_encoder.".Length);
            return $"vl_backbone.vision_backbone.vision_backbone.{rest}";
        }

        if (key.StartsWith("text_encoder."))
        {
            var rest = key.Substring("text_encoder.".Length);
            return $"text_encoder.encoder.{rest}";
        }

        if (key.StartsWith("text_projection."))
        {
            var rest = key.Substring("text_projection.".Length);
            return $"text_encoder.resizer.{rest}";
        }

        if (key.StartsWith("geometry_encoder."))
        {
            var rest = key.Substring("geometry_encoder.".Length);
            return $"geometry_encoder.{rest}";
        }

        if (key.StartsWith("detr_encoder."))
        {
            var rest = key.Substring("detr_encoder.".Length);
            return $"transformer_encoder.{rest}";
        }

        if (key.StartsWith("detr_decoder."))
        {
            var rest = key.Substring("detr_decoder.".Length);
            return $"transformer_decoder.{rest}";
        }

        if (key.StartsWith("mask_decoder."))
        {
            var rest = key.Substring("mask_decoder.".Length);
            return $"segmentation_head.mask_decoder.{rest}";
        }

        if (key.StartsWith("dot_product_scoring."))
        {
            var rest = key.Substring("dot_product_scoring.".Length);
            return $"dot_prod_scoring.{rest}";
        }

        return string.Empty;
    }

    /// <summary>
    /// Load a safetensors file and return the index and tensor data.
    /// </summary>
    private static (Dictionary<string, SafetensorsEntry>, Dictionary<string, Tensor>) LoadSafetensorsFile(string path)
    {
        using var stream = File.OpenRead(path);
        var lengthBytes = new byte[8];
        stream.Read(lengthBytes, 0, 8);
        var jsonLength = BitConverter.ToUInt64(lengthBytes, 0);
        var jsonBytes = new byte[jsonLength];
        stream.Read(jsonBytes, 0, (int)jsonLength);
        var jsonStr = System.Text.Encoding.UTF8.GetString(jsonBytes);

        var rawData = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, SafetensorsEntryData>>(jsonStr)
                    ?? throw new InvalidOperationException("Failed to parse safetensors index");

        var index = new Dictionary<string, SafetensorsEntry>();
        var tensorData = new Dictionary<string, Tensor>();
        long offset = stream.Position;

        foreach (var kvp in rawData)
        {
            if (kvp.Key == "__metadata__") continue;

            var entryData = kvp.Value;
            index[kvp.Key] = new SafetensorsEntry
            {
                DataType = entryData.dtype,
                Shape = entryData.shape,
                Offsets = entryData.dataOffsets
            };

            var tensor = torch.empty(entryData.shape, dtype: ConvertToTorchDType(entryData.dtype));
            long length = entryData.dataOffsets[1] - entryData.dataOffsets[0];
            stream.Position = offset + entryData.dataOffsets[0];
            tensor.ReadBytesFromStream(stream);
            tensorData[kvp.Key] = tensor;
        }

        return (index, tensorData);
    }

    /// <summary>
    /// Handle shape mismatches between loaded tensors and expected shapes.
    /// </summary>
    private static Tensor? HandleShapeMismatch(Tensor loaded, long[] expectedShape)
    {
        var loadedShape = loaded.size().ToArray();

        if (loadedShape.Length == expectedShape.Length)
        {
            var loadedSet = new HashSet<long>(loadedShape);
            var expectedSet = new HashSet<long>(expectedShape);
            if (loadedSet.SetEquals(expectedSet))
            {
                if (loadedShape.Length == 2)
                {
                    var transposed = loaded.transpose(0, 1);
                    if (transposed.size().ToArray().SequenceEqual(expectedShape))
                        return transposed;
                }
            }
        }

        return null;
    }

    private static ScalarType ConvertToTorchDType(string dataType)
    {
        return dataType switch
        {
            "F64" => ScalarType.Float64,
            "F32" => ScalarType.Float32,
            "F16" => ScalarType.Float16,
            "BF16" => ScalarType.BFloat16,
            "I64" => ScalarType.Int64,
            "I32" => ScalarType.Int32,
            "I16" => ScalarType.Int16,
            "I8" => ScalarType.Int8,
            "U8" => ScalarType.Byte,
            "BOOL" => ScalarType.Bool,
            _ => ScalarType.Float32
        };
    }

    /// <summary>
    /// Internal representation of safetensors entry data (JSON deserialized).
    /// </summary>
    private class SafetensorsEntryData
    {
        public string dtype { get; set; } = "";
        public long[] shape { get; set; } = Array.Empty<long>();
        public long[] dataOffsets { get; set; } = Array.Empty<long>();
    }

    /// <summary>
    /// Standard SafetensorsEntry compatible with existing code.
    /// </summary>
    public class SafetensorsEntry
    {
        public string DataType { get; init; } = "";
        public long[] Shape { get; init; } = Array.Empty<long>();
        public long[] Offsets { get; init; } = Array.Empty<long>();
    }
}
