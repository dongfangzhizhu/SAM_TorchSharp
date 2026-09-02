using SAMTorchSharp.Modeling.Sam3;
using static TorchSharp.torch;

namespace ConsistencyTest.Tests;

public sealed class Sam3GeometryPoolingTests
{
    [Fact]
    public void EncodesPointAndBoxPoolingForMultipleBatches()
    {
        using var encoder = new Sam3GeometryEncoderNew(d_model: 8, num_geo_layers: 0);
        using var image = rand([2, 8, 8, 8]);
        using var points = tensor(new float[,,]
        {
            { { 0.25f, 0.25f }, { 0.75f, 0.75f } },
            { { 0.50f, 0.50f }, { 0.10f, 0.90f } },
        });
        using var boxes = tensor(new float[,,]
        {
            { { 0.50f, 0.50f, 0.50f, 0.50f }, { 0.50f, 0.50f, 0.25f, 0.25f } },
        });
        using var pointLabels = zeros([2, 2], dtype: ScalarType.Int64);
        using var boxLabels = zeros([1, 2], dtype: ScalarType.Int64);
        var prompt = new Sam3Prompt(
            point_embeddings: points,
            point_labels: pointLabels,
            box_embeddings: boxes,
            box_labels: boxLabels);

        var result = encoder.forward(prompt, [image], [[8L, 8L]]);

        try
        {
            Assert.Equal([3L, 2L, 8L], result.Item1.shape);
            Assert.Equal([2L, 3L], result.Item2.shape);
            Assert.True(result.Item1.isfinite().all().item<bool>());
        }
        finally
        {
            result.Item1.Dispose();
            result.Item2.Dispose();
        }
    }
}