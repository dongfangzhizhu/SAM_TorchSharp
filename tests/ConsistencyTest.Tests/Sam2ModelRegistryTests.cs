using SAMTorchSharp.Modeling.Sam2;

namespace ConsistencyTest.Tests;

public sealed class Sam2ModelRegistryTests
{
    [Fact]
    public void RegistryContainsFourSupportedInferenceVariants()
    {
        Assert.Equal(4, Sam2ModelRegistry.SupportedVariants.Count);
        Assert.Contains(Sam2ModelVariant.Sam2Tiny, Sam2ModelRegistry.SupportedVariants);
        Assert.Contains(Sam2ModelVariant.Sam2Small, Sam2ModelRegistry.SupportedVariants);
        Assert.Contains(Sam2ModelVariant.Sam21Tiny, Sam2ModelRegistry.SupportedVariants);
        Assert.Contains(Sam2ModelVariant.Sam21Small, Sam2ModelRegistry.SupportedVariants);
    }

    [Theory]
    [InlineData(Sam2ModelVariant.Sam2Tiny, "sam2_hiera_tiny", 7, false)]
    [InlineData(Sam2ModelVariant.Sam2Small, "sam2_hiera_small", 11, false)]
    [InlineData(Sam2ModelVariant.Sam21Tiny, "sam2.1_hiera_tiny", 7, true)]
    [InlineData(Sam2ModelVariant.Sam21Small, "sam2.1_hiera_small", 11, true)]
    public void RegistryMapsOfficialTinyAndSmallConfigurations(
        Sam2ModelVariant variant, string name, int stageDepth, bool isSam21)
    {
        var options = Sam2ModelRegistry.Get(variant);

        Assert.Equal(name, options.Name);
        Assert.Equal(1024, options.ImageSize);
        Assert.Equal(stageDepth, options.Stages[2]);
        Assert.Equal(isSam21, options.AddTemporalPositionEncodingToObjectPointers);
        Assert.Equal(isSam21, options.ProjectTemporalPositionEncodingInObjectPointers);
        Assert.Equal(isSam21, options.UseSignedTemporalPositionEncodingForObjectPointers);
        Assert.Equal(isSam21, options.UseNoObjectSpatialEmbedding);
    }

    [Fact]
    public void RegistryRejectsUnknownVariant()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Sam2ModelRegistry.Get((Sam2ModelVariant)99));
    }
}