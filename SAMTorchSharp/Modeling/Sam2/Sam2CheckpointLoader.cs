using System.Collections;
using TorchSharp;
using TorchSharp.Modules;
using TorchSharp.PyBridge;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SAMTorchSharp.Modeling.Sam2;

public enum Sam2CheckpointFormat
{
    PyTorch,
    Safetensors,
}

public sealed record Sam2CheckpointLoadReport(
    string Path,
    Sam2CheckpointFormat Format,
    int ModelTensorCount,
    int CheckpointTensorCount,
    IReadOnlyList<string> LoadedKeys,
    IReadOnlyList<string> MissingKeys,
    IReadOnlyList<string> UnexpectedKeys,
    IReadOnlyList<string> ShapeMismatches)
{
    public double Coverage => ModelTensorCount == 0 ? 0 : LoadedKeys.Count * 100.0 / ModelTensorCount;
    public bool IsComplete => MissingKeys.Count == 0 && UnexpectedKeys.Count == 0 && ShapeMismatches.Count == 0;
}

public static class Sam2CheckpointLoader
{
    public static Sam2CheckpointLoadReport Load(Module model, string checkpointPath, bool strict = true)
    {
        ArgumentNullException.ThrowIfNull(model);
        var path = Path.GetFullPath(checkpointPath);
        if (!File.Exists(path))
            throw new FileNotFoundException("SAM2 checkpoint was not found.", path);

        var format = Path.GetExtension(path).ToLowerInvariant() switch
        {
            ".pt" or ".pth" => Sam2CheckpointFormat.PyTorch,
            ".safetensors" => Sam2CheckpointFormat.Safetensors,
            var extension => throw new NotSupportedException(
                $"Unsupported SAM2 checkpoint extension '{extension}'. Use .pt, .pth, or .safetensors."),
        };

        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        var checkpoint = format == Sam2CheckpointFormat.PyTorch
            ? LoadPyTorchStateDict(path)
            : Safetensors.LoadStateDict(path);
        var current = model.state_dict();

        var missing = current.Keys.Except(checkpoint.Keys).Order().ToArray();
        var unexpected = checkpoint.Keys.Except(current.Keys).Order().ToArray();
        var shapeMismatches = checkpoint.Keys.Intersect(current.Keys)
            .Where(key => !checkpoint[key].shape.SequenceEqual(current[key].shape))
            .Order()
            .Select(key => $"{key}: checkpoint=[{string.Join(",", checkpoint[key].shape)}], model=[{string.Join(",", current[key].shape)}]")
            .ToArray();
        var mismatchedKeys = shapeMismatches.Select(value => value[..value.IndexOf(':')]).ToHashSet();
        var loaded = checkpoint.Keys.Intersect(current.Keys)
            .Where(key => !mismatchedKeys.Contains(key))
            .Order()
            .ToArray();

        var report = new Sam2CheckpointLoadReport(
            path, format, current.Count, checkpoint.Count, loaded, missing, unexpected, shapeMismatches);
        if (strict && !report.IsComplete)
            throw new InvalidDataException(
                $"SAM2 checkpoint is incompatible: loaded={loaded.Length}, missing={missing.Length}, " +
                $"unexpected={unexpected.Length}, shape_mismatch={shapeMismatches.Length}.");

        var compatible = loaded.ToDictionary(key => key, key => checkpoint[key]);
        model.load_state_dict(compatible, strict: false);
        return report;
    }

    private static Dictionary<string, Tensor> LoadPyTorchStateDict(string path)
    {
        var root = PyTorchUnpickler.UnpickleStateDict(path);
        object state = root.ContainsKey("model") ? root["model"]! : root;
        if (state is not IDictionary dictionary)
            throw new InvalidDataException("PyTorch checkpoint must contain a state dictionary or a top-level 'model' state dictionary.");

        var result = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (DictionaryEntry entry in dictionary)
        {
            if (entry.Key is not string key || entry.Value is not Tensor tensor)
                throw new InvalidDataException("SAM2 state dictionary must contain only string-to-tensor entries.");
            result.Add(key, tensor);
        }
        return result;
    }
}