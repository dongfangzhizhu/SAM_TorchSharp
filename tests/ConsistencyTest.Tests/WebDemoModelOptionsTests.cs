using SAMTorchSharp.Modeling.Sam2;
using WebDemo.Models;

namespace ConsistencyTest.Tests;

public sealed class WebDemoModelOptionsTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"sam-webdemo-{Guid.NewGuid():N}");

    public WebDemoModelOptionsTests() => Directory.CreateDirectory(_root);

    [Fact]
    public void EmptyDirectoryOrNameMakesEachModelUnavailable()
    {
        var options = new ModelOptions
        {
            Sam = new() { Directory = string.Empty, Name = "mobile_sam" },
            Sam2 = new() { Directory = "weights", Name = string.Empty },
            Sam21 = new(),
            Sam3 = new() { Directory = "weights", Name = string.Empty },
            Sam31 = new() { Directory = string.Empty, Name = "sam3.1_multiplex" },
        };

        var models = new ModelPathResolver(options, _root).All;

        Assert.Equal(["sam", "sam2", "sam2.1", "sam3", "sam3.1"], models.Select(x => x.Id));
        Assert.All(models, model =>
        {
            Assert.False(model.Configured);
            Assert.False(model.Available);
            Assert.Equal(string.Empty, model.CheckpointPath);
        });
    }

    [Fact]
    public void ResolvesFiveIndependentDirectoriesAndExtensionlessNames()
    {
        var options = new ModelOptions
        {
            Sam = Definition("sam-weights", "mobile_sam"),
            Sam2 = Definition("sam2-weights", "sam2_hiera_tiny"),
            Sam21 = Definition("sam21-weights", "sam2.1_hiera_tiny"),
            Sam3 = Definition("sam3-weights", "sam3"),
            Sam31 = Definition("sam31-weights", "sam3.1_multiplex"),
        };
        CreateCheckpoint("sam-weights", "mobile_sam.pt");
        CreateCheckpoint("sam2-weights", "sam2_hiera_tiny.pt");
        CreateCheckpoint("sam21-weights", "sam2.1_hiera_tiny.safetensors");
        CreateCheckpoint("sam3-weights", "sam3.bin");
        CreateCheckpoint("sam31-weights", "sam3.1_multiplex.pt");

        var models = new ModelPathResolver(options, _root).All;

        Assert.All(models, model => Assert.True(model.Available, model.CheckpointPath));
        Assert.EndsWith("mobile_sam.pt", models[0].CheckpointPath);
        Assert.EndsWith("sam2_hiera_tiny.pt", models[1].CheckpointPath);
        Assert.EndsWith("sam2.1_hiera_tiny.safetensors", models[2].CheckpointPath);
        Assert.EndsWith("sam3.bin", models[3].CheckpointPath);
        Assert.EndsWith("sam3.1_multiplex.pt", models[4].CheckpointPath);
    }

    [Theory]
    [InlineData("mobile_sam", "vit_t")]
    [InlineData("sam_vit_b.pth", "vit_b")]
    [InlineData("sam_vit_l", "vit_l")]
    [InlineData("sam_vit_h.safetensors", "vit_h")]
    public void ParsesSamArchitectureFromModelName(string name, string expected) =>
        Assert.Equal(expected, ModelPathResolver.ParseSamModel(name));

    [Theory]
    [InlineData("sam2_hiera_tiny.pt", false, Sam2ModelVariant.Sam2Tiny)]
    [InlineData("sam2_hiera_small", false, Sam2ModelVariant.Sam2Small)]
    [InlineData("sam2.1_hiera_tiny.pt", true, Sam2ModelVariant.Sam21Tiny)]
    [InlineData("sam2.1_hiera_small", true, Sam2ModelVariant.Sam21Small)]
    public void ParsesSam2ArchitectureFromModelName(string name, bool sam21, Sam2ModelVariant expected) =>
        Assert.Equal(expected, ModelPathResolver.ParseSam2Model(name, sam21));

    [Fact]
    public void RejectsSam2VersionInWrongConfigurationSection()
    {
        Assert.Throws<ArgumentException>(() => ModelPathResolver.ParseSam2Model("sam2.1_hiera_tiny", false));
        Assert.Throws<ArgumentException>(() => ModelPathResolver.ParseSam2Model("sam2_hiera_tiny", true));
    }

    private static ModelDefinition Definition(string directory, string name) => new() { Directory = directory, Name = name };

    private void CreateCheckpoint(string directory, string name)
    {
        var path = Path.Combine(_root, directory);
        Directory.CreateDirectory(path);
        File.WriteAllText(Path.Combine(path, name), string.Empty);
    }

    public void Dispose() => Directory.Delete(_root, recursive: true);
}