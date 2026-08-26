namespace WebDemo.Models;

public sealed class ModelOptions
{
    public const string SectionName = "Models";

    public string WeightsDirectory { get; set; } = "weights";
    public string Sam1Checkpoint { get; set; } = "mobile_sam.pt";
    public string Sam2Checkpoint { get; set; } = "sam2.1_hiera_tiny.pt";
    public string Sam2Variant { get; set; } = "sam2.1-tiny";
    public string Sam3Checkpoint { get; set; } = "sam3.safetensors";
}

public sealed record ModelAvailability(string Id, string Name, string CheckpointPath, bool Available, string PromptType);

public sealed class ModelPathResolver
{
    private readonly ModelOptions _options;
    private readonly string _contentRoot;

    public ModelPathResolver(ModelOptions options, string contentRoot)
    {
        _options = options;
        _contentRoot = contentRoot;
    }

    public string Resolve(string checkpoint)
    {
        var root = Path.IsPathRooted(_options.WeightsDirectory)
            ? _options.WeightsDirectory
            : Path.Combine(_contentRoot, _options.WeightsDirectory);
        return Path.GetFullPath(Path.IsPathRooted(checkpoint) ? checkpoint : Path.Combine(root, checkpoint));
    }

    public string Sam1 => Resolve(_options.Sam1Checkpoint);
    public string Sam2 => Resolve(_options.Sam2Checkpoint);
    public string Sam3 => Resolve(_options.Sam3Checkpoint);
}