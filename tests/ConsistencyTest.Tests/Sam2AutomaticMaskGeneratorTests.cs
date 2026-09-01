using SAMTorchSharp;
using static TorchSharp.torch;

namespace ConsistencyTest.Tests;

public sealed class Sam2AutomaticMaskGeneratorTests
{
    [Fact]
    public void GeneratesIndependentMasksForEveryGridPoint()
    {
        using var model = BuildSam2.BuildSam2HieraTiny(imageSize: 64);
        using var generator = new SAM2AutomaticMaskGenerator(
            model,
            pointsPerSide: 2,
            pointsPerBatch: 4,
            predIouThresh: -1,
            stabilityScoreThreshold: -1,
            boxNmsThresh: float.PositiveInfinity,
            multimaskOutput: true);
        using var image = rand(32, 48, 3);

        var records = generator.Generate(image);

        Assert.InRange(records.Count, 4, 12);
        Assert.All(records, record =>
        {
            Assert.NotNull(record.Segmentation);
            Assert.NotNull(record.BinaryMask);
            Assert.Equal(32, record.BinaryMask!.GetLength(0));
            Assert.Equal(48, record.BinaryMask.GetLength(1));
            Assert.Equal([32L, 48L], record.Segmentation!.Size);
            Assert.Equal(32 * 48, record.Segmentation.Counts.Sum());
            Assert.InRange(record.Area, 0, 32 * 48);
            Assert.Equal(4, record.BBox.Length);
            Assert.Equal(2, record.PointCoords.Length);
            Assert.Equal([0f, 0f, 48f, 32f], record.CropBox);
        });

        var distinctPoints = records
            .Select(record => (X: record.PointCoords[0], Y: record.PointCoords[1]))
            .Distinct()
            .OrderBy(point => point.X)
            .ThenBy(point => point.Y)
            .ToArray();
        Assert.Equal([(12f, 8f), (12f, 24f), (36f, 8f), (36f, 24f)], distinctPoints);
    }

    [Fact]
    public void RleRoundTripPreservesBinaryMaskLayout()
    {
        using var mask = zeros(1, 3, 4, dtype: ScalarType.Bool);
        mask[0, 1, 2] = true;
        mask[0, 2, 0] = true;

        var decoded = AMGUtiities.RleToMask(AMGUtiities.MaskToRle(mask)[0]);

        Assert.True(decoded[1, 2]);
        Assert.True(decoded[2, 0]);
        Assert.Equal(2, decoded.Cast<bool>().Count(value => value));
    }

    [Fact]
    public void MaskDataFiltersRlesByNmsIndicesWithoutTreatingIndicesAsBooleans()
    {
        var data = new MaskData();
        data.Set("scores", tensor(new float[] { 10, 20, 30 }));
        data.Set("rles", new List<RleElement>
        {
            new() { Size = [1, 1], Counts = [1] },
            new() { Size = [1, 1], Counts = [0, 1] },
            new() { Size = [1, 1], Counts = [1] },
        });
        using var keep = tensor(new long[] { 2, 0 });

        data.Filter(keep);

        Assert.Equal([30f, 10f], data.GetTensor("scores").data<float>().ToArray());
        Assert.Equal(2, data.GetRles("rles").Count);
        Assert.Equal([1], data.GetRles("rles")[0].Counts);
        Assert.Equal([1], data.GetRles("rles")[1].Counts);
    }
}