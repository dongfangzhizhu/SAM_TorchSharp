using TorchSharp;
using static TorchSharp.torch;

namespace ConsistencyTest;

internal static class Sam3ImageCommand
{
    public static NpyArray LoadImage(string path)
    {
        var image = NpyFile.ReadFloat32(path);
        ValidateImage(image);
        return image;
    }

    public static void ValidateImage(NpyArray image)
    {
        if (!image.Shape.SequenceEqual([1, 3, 1008, 1008]))
            throw new CliException("--image must be a float32 NCHW array with shape [1,3,1008,1008].");
        if (image.Values.Any(value => !float.IsFinite(value)))
            throw new CliException("--image values must be finite.");
    }

    public static Tensor ToTensor(NpyArray image) =>
        tensor(image.Values, dtype: ScalarType.Float32, device: CPU).reshape(1, 3, 1008, 1008);
}