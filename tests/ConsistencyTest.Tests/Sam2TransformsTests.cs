using SAMTorchSharp;
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
}