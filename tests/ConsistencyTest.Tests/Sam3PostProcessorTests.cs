using TorchSharp;
using WebDemo.Utility;
using static TorchSharp.torch;

namespace ConsistencyTest.Tests;

public sealed class Sam3PostProcessorTests
{
    [Fact]
    public void FiltersSortsScalesBoxesAndResizesMasks()
    {
        using var boxes = tensor(new float[,,] { { { 0.5f, 0.5f, 0.4f, 0.2f }, { 0.2f, 0.3f, 0.2f, 0.2f }, { 0.8f, 0.8f, 0.1f, 0.1f } } });
        using var logits = tensor(new float[,,] { { { 2f }, { -2f }, { 1f } } });
        using var masks = tensor(new float[,,,]
        {
            { { { 10f, 10f }, { -10f, -10f } }, { { -10f, -10f }, { -10f, -10f } }, { { -10f, 10f }, { -10f, 10f } } }
        });

        using var result = Sam3PostProcessor.Process(boxes, logits, masks, imageHeight: 20, imageWidth: 100);

        Assert.Equal(2, result.Detections.Count);
        Assert.True(result.Detections[0].Score > result.Detections[1].Score);
        Assert.Equal([30f, 8f, 70f, 12f], result.Detections[0].Box, new FloatComparer(0.001f));
        Assert.Equal([2L, 20L, 100L], result.Masks.shape);
        Assert.True(result.Masks.any().item<bool>());
    }

    [Fact]
    public void ReturnsEmptyMasksWhenNoQueryPassesThreshold()
    {
        using var boxes = zeros(1, 2, 4);
        using var logits = full([1, 2, 1], -10f);
        using var masks = zeros(1, 2, 2, 2);

        using var result = Sam3PostProcessor.Process(boxes, logits, masks, 12, 16);

        Assert.Empty(result.Detections);
        Assert.Equal([0L, 12L, 16L], result.Masks.shape);
    }

    [Theory]
    [InlineData("boxes")]
    [InlineData("logits")]
    [InlineData("masks")]
    public void RejectsNonFiniteModelOutputs(string outputName)
    {
        using var boxes = zeros(1, 1, 4);
        using var logits = zeros(1, 1, 1);
        using var masks = zeros(1, 1, 2, 2);
        using var nonFinite = full([1], float.NaN);
        var target = outputName switch
        {
            "boxes" => boxes,
            "logits" => logits,
            _ => masks,
        };
        target.flatten()[0] = nonFinite[0];

        var exception = Assert.Throws<InvalidOperationException>(() =>
            Sam3PostProcessor.Process(boxes, logits, masks, 12, 16));

        Assert.Contains("non-finite", exception.Message);
    }

    private sealed class FloatComparer(float tolerance) : IEqualityComparer<float>
    {
        public bool Equals(float x, float y) => Math.Abs(x - y) <= tolerance;
        public int GetHashCode(float value) => value.GetHashCode();
    }
}