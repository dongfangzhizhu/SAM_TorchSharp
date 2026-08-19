using SAMTorchSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace ConsistencyTest.Tests;

public sealed class Sam2TransformsTests
{
    [Fact]
    public void UsesOfficialImageNetNormalization()
    {
        var transforms = new SAM2Transforms(1);
        using var image = tensor(new byte[] { 255, 255, 255 }).reshape(1, 1, 3);

        using var result = transforms.__call(image);
        var values = result.flatten().data<float>().ToArray();

        Assert.Equal((1 - 0.485) / 0.229, values[0], 5);
        Assert.Equal((1 - 0.456) / 0.224, values[1], 5);
        Assert.Equal((1 - 0.406) / 0.225, values[2], 5);
    }

    [Fact]
    public void ConvertsOriginalPixelCoordinatesToModelCoordinates()
    {
        var transforms = new SAM2Transforms(1024);
        using var coordinates = tensor(new float[] { 320, 180 }).reshape(1, 2);

        using var result = transforms.TransformCoordinates(coordinates, 360, 640);

        Assert.Equal(new[] { 512f, 512f }, result.data<float>().ToArray());
    }

    [Fact]
    public void RestoresMasksToOriginalImageShape()
    {
        var transforms = new SAM2Transforms(1024);
        using var masks = zeros(1, 3, 256, 256);

        using var result = transforms.PostprocessMasks(masks, 360, 640);

        Assert.Equal(new long[] { 1, 3, 360, 640 }, result.shape);
    }

    [Fact]
    public void TreatsAmbiguousThreeByWidthByThreeInputAsHwc()
    {
        using var image = tensor(Enumerable.Range(0, 90).Select(value => value / 89f).ToArray())
            .reshape(3, 10, 3);
        using var explicitChw = image.permute(2, 0, 1);
        var transforms = new SAM2Transforms(4);

        using var actual = transforms.__call(image);
        using var expected = transforms.__call(explicitChw);
        using var difference = (actual - expected).abs();

        Assert.Equal([3L, 4L, 4L], actual.shape);
        Assert.True(difference.max().ToSingle() < 1e-6f);
    }

    [Fact]
    public void ProducesEquivalentResultsForByteAndNormalizedFloatImages()
    {
        var bytes = Enumerable.Range(0, 60).Select(value => (byte)(value * 4)).ToArray();
        using var byteImage = tensor(bytes, dtype: ScalarType.Byte).reshape(4, 5, 3);
        using var floatImage = tensor(bytes.Select(value => value / 255f).ToArray()).reshape(4, 5, 3);
        var transforms = new SAM2Transforms(4);

        using var fromBytes = transforms.__call(byteImage);
        using var fromFloats = transforms.__call(floatImage);
        using var difference = (fromBytes - fromFloats).abs();

        Assert.True(difference.max().ToSingle() < 1e-6f);
    }

    [Fact]
    public void AcceptsChwInputForCompatibility()
    {
        using var image = zeros([3, 4, 5], dtype: ScalarType.Float32);
        var transforms = new SAM2Transforms(4);

        using var actual = transforms.__call(image);

        Assert.Equal([3L, 4L, 4L], actual.shape);
    }

    [Theory]
    [InlineData(2, 4, 5)]
    [InlineData(4, 5, 2)]
    public void RejectsImagesWithoutThreeChannels(long first, long second, long third)
    {
        using var image = zeros([first, second, third], dtype: ScalarType.Float32);
        var transforms = new SAM2Transforms(4);

        var error = Assert.Throws<ArgumentException>(() => transforms.__call(image));

        Assert.Contains("exactly 3 RGB channels", error.Message);
    }

    [Fact]
    public void RejectsNonThreeDimensionalInput()
    {
        using var image = zeros([1, 3, 4, 5], dtype: ScalarType.Float32);
        var transforms = new SAM2Transforms(4);

        var error = Assert.Throws<ArgumentException>(() => transforms.__call(image));

        Assert.Contains("3D RGB tensor", error.Message);
    }
}
