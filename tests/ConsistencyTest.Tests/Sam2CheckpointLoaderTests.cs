using SAMTorchSharp.Modeling.Sam2;
using static TorchSharp.torch.nn;

namespace ConsistencyTest.Tests;

public sealed class Sam2CheckpointLoaderTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), $"sam2-loader-tests-{Guid.NewGuid():N}");

    public Sam2CheckpointLoaderTests() => Directory.CreateDirectory(_directory);

    [Fact]
    public void LoadsCompatibleSafetensorsAndReportsFullCoverage()
    {
        using var source = Linear(2, 3);
        using var target = Linear(2, 3);
        var path = Path.Combine(_directory, "model.safetensors");
        TorchSharp.PyBridge.Safetensors.SaveStateDict(path, source.state_dict());

        var report = Sam2CheckpointLoader.Load(target, path);

        Assert.True(report.IsComplete);
        Assert.Equal(Sam2CheckpointFormat.Safetensors, report.Format);
        Assert.Equal(100, report.Coverage);
        Assert.Equal(2, report.LoadedKeys.Count);
    }

    [Fact]
    public void StrictLoadingRejectsShapeMismatch()
    {
        using var source = Linear(2, 4);
        using var target = Linear(2, 3);
        var path = Path.Combine(_directory, "mismatch.safetensors");
        TorchSharp.PyBridge.Safetensors.SaveStateDict(path, source.state_dict());

        var error = Assert.Throws<InvalidDataException>(() => Sam2CheckpointLoader.Load(target, path));

        Assert.Contains("shape_mismatch=2", error.Message);
    }

    [Fact]
    public void NonStrictLoadingReturnsMismatchReport()
    {
        using var source = Linear(2, 4);
        using var target = Linear(2, 3);
        var path = Path.Combine(_directory, "partial.safetensors");
        TorchSharp.PyBridge.Safetensors.SaveStateDict(path, source.state_dict());

        var report = Sam2CheckpointLoader.Load(target, path, strict: false);

        Assert.False(report.IsComplete);
        Assert.Empty(report.LoadedKeys);
        Assert.Equal(2, report.ShapeMismatches.Count);
    }

    [Fact]
    public void RejectsUnknownCheckpointFormat()
    {
        using var model = Linear(2, 3);
        var path = Path.Combine(_directory, "model.bin");
        File.WriteAllBytes(path, []);

        Assert.Throws<NotSupportedException>(() => Sam2CheckpointLoader.Load(model, path));
    }

    public void Dispose() => Directory.Delete(_directory, recursive: true);
}