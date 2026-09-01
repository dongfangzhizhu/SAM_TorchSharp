using TorchSharp;
using static TorchSharp.torch;

namespace ConsistencyTest;

internal static class Sam3ImageCommand
{
    private static readonly IReadOnlyDictionary<string, int> RequiredOutputRanks = new Dictionary<string, int>
    {
        ["pred_boxes"] = 3,
        ["pred_logits"] = 3,
        ["pred_masks"] = 4,
        ["semantic_seg"] = 4,
    };

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

    public static void ValidateOutputs(IReadOnlyDictionary<string, Tensor> outputs)
    {
        ArgumentNullException.ThrowIfNull(outputs);
        foreach (var (name, rank) in RequiredOutputRanks)
        {
            if (!outputs.TryGetValue(name, out var output))
                throw new InvalidOperationException($"SAM 3 inference did not produce required output '{name}'.");
            if (output.dim() != rank)
                throw new InvalidOperationException($"SAM 3 output '{name}' must have rank {rank}.");
            if (!output.isfinite().all().item<bool>())
                throw new InvalidOperationException($"SAM 3 output '{name}' contains non-finite values.");
        }

        var queryCount = outputs["pred_boxes"].size(1);
        if (outputs["pred_boxes"].shape is not [1, _, 4] ||
            outputs["pred_logits"].shape is not [1, _, 1] ||
            outputs["pred_masks"].size(0) != 1 ||
            outputs["pred_masks"].size(1) != queryCount ||
            outputs["pred_logits"].size(1) != queryCount ||
            outputs["semantic_seg"].shape is not [1, 1, _, _])
            throw new InvalidOperationException("SAM 3 output shapes or query counts do not match the inference contract.");
    }
}