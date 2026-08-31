using SAMTorchSharp.Modeling.Sam3;
using static TorchSharp.torch;

namespace ConsistencyTest.Tests;

public sealed class Sam3SegmentationHeadTests
{
    [Fact]
    public void ProducesPerQueryMaskLogitsAtHighestFpnResolution()
    {
        manual_seed(17);
        using var head = new Sam3MaskDecoder(d_model: 8);
        using var objectQueries = randn(1, 3, 8);
        using var encoderHiddenStates = randn(1, 1, 8);
        using var prompt = randn(2, 1, 8);
        using var promptMask = ones(1, 2, dtype: ScalarType.Bool);
        using var level0 = randn(1, 8, 8, 8);
        using var level1 = randn(1, 8, 4, 4);
        using var level2 = randn(1, 8, 2, 2);
        using var level3 = randn(1, 8, 1, 1);

        var (masks, semantic) = head.forward(
            objectQueries,
            [level0, level1, level2, level3],
            encoderHiddenStates,
            prompt,
            promptMask);
        using (masks)
        using (semantic)
        {
            Assert.Equal([1L, 3L, 8L, 8L], masks.shape);
            Assert.Equal([1L, 1L, 8L, 8L], semantic.shape);
            Assert.True(isfinite(masks).all().item<bool>());
        }
    }

    [Fact]
    public void RequiresFourFpnLevels()
    {
        using var head = new Sam3MaskDecoder(d_model: 8);
        using var objectQueries = zeros(1, 1, 8);
        using var encoderHiddenStates = zeros(1, 1, 8);
        using var prompt = zeros(1, 1, 8);
        using var feature = zeros(1, 8, 1, 1);

        var exception = Assert.Throws<ArgumentException>(() => head.forward(
            objectQueries,
            [feature],
            encoderHiddenStates,
            prompt,
            null));

        Assert.Contains("four FPN feature levels", exception.Message);
    }
}