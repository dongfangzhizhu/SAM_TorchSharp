namespace ConsistencyTest.Tests;

using SAMTorchSharp.Modeling.Sam3;

public sealed class CliOptionsTests
{
    [Theory]
    [InlineData("true", true)]
    [InlineData("false", false)]
    [InlineData("TRUE", true)]
    public void ParsesBooleanOptions(string value, bool expected)
    {
        var options = CliOptions.Parse(["--strict", value]);

        Assert.Equal(expected, options.GetBool("strict", !expected));
        options.EnsureNoUnused();
    }

    [Fact]
    public void RejectsInvalidBooleanOption()
    {
        var options = CliOptions.Parse(["--strict", "yes"]);

        var error = Assert.Throws<CliException>(() => options.GetBool("strict", true));

        Assert.Contains("must be true or false", error.Message);
    }

    [Fact]
    public void OptionalPathReturnsNullWhenOmitted()
    {
        var options = CliOptions.Parse([]);

        Assert.Null(options.GetOptionalPath("points"));
        options.EnsureNoUnused();
    }

    [Fact]
    public void OptionalPathRejectsMissingFile()
    {
        var options = CliOptions.Parse(["--points", Path.Combine(Path.GetTempPath(), $"missing-{Guid.NewGuid():N}.npy")]);

        var error = Assert.Throws<CliException>(() => options.GetOptionalPath("points"));

        Assert.Contains("does not exist", error.Message);
    }

    [Theory]
    [InlineData("model.safetensors", Sam3CheckpointFormat.OfficialSafetensors)]
    [InlineData("model.bin", Sam3CheckpointFormat.ConvertedBinary)]
    [InlineData("model.pt", Sam3CheckpointFormat.ConvertedBinary)]
    public void Sam3RunAcceptsSupportedCheckpointRoutes(string path, Sam3CheckpointFormat expected)
    {
        Assert.Equal(expected, Program.GetSam3CheckpointFormat(path));
    }

    [Fact]
    public void Sam3RunRejectsUnsupportedCheckpointRoute()
    {
        var exception = Assert.Throws<CliException>(() => Program.GetSam3CheckpointFormat("model.ckpt"));
        Assert.Contains("sibling .bin", exception.Message);
    }
}