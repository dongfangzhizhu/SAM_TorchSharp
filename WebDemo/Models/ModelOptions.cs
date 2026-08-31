using SAMTorchSharp.Modeling.Sam2;

namespace WebDemo.Models;

public sealed class ModelOptions
{
    public const string SectionName = "Models";

    public ModelDefinition Sam { get; set; } = new();
    public ModelDefinition Sam2 { get; set; } = new();
    public ModelDefinition Sam21 { get; set; } = new();
    public ModelDefinition Sam3 { get; set; } = new();
    public ModelDefinition Sam31 { get; set; } = new();
}


public sealed class ModelDefinition
{
    public string Directory { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
}

public sealed record ModelAvailability(
    string Id, string Name, string CheckpointPath, bool Available, string PromptType, string Status);

public sealed record ResolvedModel(string Id, string DisplayName, string PromptType, string Name, string CheckpointPath)
{
    public bool Configured => !string.IsNullOrWhiteSpace(Name) && !string.IsNullOrWhiteSpace(CheckpointPath);
    public bool Available => Configured && File.Exists(CheckpointPath);
}

public sealed class ModelPathResolver
{
    private readonly string _contentRoot;

    public ModelPathResolver(ModelOptions options, string contentRoot)
    {
        ArgumentNullException.ThrowIfNull(options);
        _contentRoot = Path.GetFullPath(contentRoot);
        Sam = Resolve("sam", "SAM 1", "点或矩形", options.Sam, [".pt", ".pth", ".safetensors"]);
        Sam2 = Resolve("sam2", "SAM 2", "点或矩形", options.Sam2, [".pt", ".pth", ".safetensors"]);
        Sam21 = Resolve("sam2.1", "SAM 2.1", "点或矩形", options.Sam21, [".pt", ".pth", ".safetensors"]);
        Sam3 = Resolve("sam3", "SAM 3 detector-only", "文本", options.Sam3, [".safetensors", ".bin", ".pt"]);
        Sam31 = Resolve("sam3.1", "SAM 3.1 detector-only", "文本", options.Sam31, [".safetensors", ".bin", ".pt"]);
    }

    public ResolvedModel Sam { get; }
    public ResolvedModel Sam2 { get; }
    public ResolvedModel Sam21 { get; }
    public ResolvedModel Sam3 { get; }
    public ResolvedModel Sam31 { get; }
    public IReadOnlyList<ResolvedModel> All => [Sam, Sam2, Sam21, Sam3, Sam31];

    public ResolvedModel Get(string id) => id switch
    {
        "sam" or "sam1" => Sam,
        "sam2" => Sam2,
        "sam2.1" or "sam21" => Sam21,
        "sam3" => Sam3,
        "sam3.1" or "sam31" => Sam31,
        _ => throw new ArgumentException("Model must be sam, sam2, sam2.1, sam3, or sam3.1."),
    };

    public static string ParseSamModel(string name) => NormalizeName(name) switch
    {
        "mobile_sam" or "sam_vit_t" or "vit_t" => "vit_t",
        "sam_vit_b" or "vit_b" => "vit_b",
        "sam_vit_l" or "vit_l" => "vit_l",
        "sam_vit_h" or "vit_h" or "sam" => "vit_h",
        _ => throw new ArgumentException("SAM model name must be mobile_sam/sam_vit_t, sam_vit_b, sam_vit_l, or sam_vit_h."),
    };

    public static Sam2ModelVariant ParseSam2Model(string name, bool requireSam21)
    {
        var variant = NormalizeName(name) switch
        {
            "sam2_hiera_tiny" or "sam2-tiny" => Sam2ModelVariant.Sam2Tiny,
            "sam2_hiera_small" or "sam2-small" => Sam2ModelVariant.Sam2Small,
            "sam2.1_hiera_tiny" or "sam2.1-tiny" => Sam2ModelVariant.Sam21Tiny,
            "sam2.1_hiera_small" or "sam2.1-small" => Sam2ModelVariant.Sam21Small,
            _ => throw new ArgumentException("SAM2 model name must be sam2_hiera_tiny, sam2_hiera_small, sam2.1_hiera_tiny, or sam2.1_hiera_small."),
        };
        var isSam21 = variant is Sam2ModelVariant.Sam21Tiny or Sam2ModelVariant.Sam21Small;
        if (isSam21 != requireSam21)
            throw new ArgumentException(requireSam21
                ? "Models:Sam21:Name must identify a SAM 2.1 model."
                : "Models:Sam2:Name must identify a SAM 2 model (not SAM 2.1)." );
        return variant;
    }

    private ResolvedModel Resolve(
        string id, string displayName, string promptType, ModelDefinition definition, IReadOnlyList<string> extensions)
    {
        var directory = definition.Directory?.Trim() ?? string.Empty;
        var name = definition.Name?.Trim() ?? string.Empty;
        if (directory.Length == 0 || name.Length == 0)
            return new ResolvedModel(id, displayName, promptType, name, string.Empty);

        var root = Path.IsPathRooted(directory) ? directory : Path.Combine(_contentRoot, directory);
        var candidate = Path.GetFullPath(Path.Combine(root, name));
        if (extensions.Any(extension => name.EndsWith(extension, StringComparison.OrdinalIgnoreCase)))
            return new ResolvedModel(id, displayName, promptType, name, candidate);

        var existing = extensions.Select(extension => candidate + extension).FirstOrDefault(File.Exists);
        return new ResolvedModel(id, displayName, promptType, name, existing ?? candidate + extensions[0]);
    }

    private static string NormalizeName(string value)
    {
        var name = Path.GetFileName(value.Trim()).ToLowerInvariant();
        foreach (var extension in new[] { ".safetensors", ".pth", ".pt", ".bin" })
        {
            if (name.EndsWith(extension, StringComparison.OrdinalIgnoreCase))
                return name[..^extension.Length];
        }
        return name;
    }
}
