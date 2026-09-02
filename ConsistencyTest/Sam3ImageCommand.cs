using TorchSharp;
using static TorchSharp.torch;

namespace ConsistencyTest;

internal sealed record Sam3GeometryInputs(NpyArray? Points, NpyArray? Labels, NpyArray? Box);

internal sealed class Sam3InferenceResult : IDisposable
{
    public Sam3InferenceResult(Tensor boxes, Tensor scores, Tensor masks)
    {
        Boxes = boxes;
        Scores = scores;
        Masks = masks;
    }

    public Tensor Boxes { get; }
    public Tensor Scores { get; }
    public Tensor Masks { get; }

    public void Dispose()
    {
        Boxes.Dispose();
        Scores.Dispose();
        Masks.Dispose();
    }
}

internal static class Sam3ImageCommand
{
    public const int ModelInputSize = 1008;
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

    public static Sam3GeometryInputs LoadGeometryInputs(
        string? pointsPath,
        string? labelsPath,
        string? boxPath)
    {
        var inputs = new Sam3GeometryInputs(
            pointsPath is null ? null : NpyFile.ReadFloat32(pointsPath),
            labelsPath is null ? null : NpyFile.ReadFloat32(labelsPath),
            boxPath is null ? null : NpyFile.ReadFloat32(boxPath));
        ValidateGeometryInputs(inputs);
        return inputs;
    }

    public static void ValidateGeometryInputs(Sam3GeometryInputs inputs)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        if ((inputs.Points is null) != (inputs.Labels is null))
            throw new CliException("--points and --labels must be supplied together.");

        if (inputs.Points is not null)
        {
            if (inputs.Points.Shape.Length != 2 || inputs.Points.Shape[0] <= 0 || inputs.Points.Shape[1] != 2)
                throw new CliException("--points must have shape [N,2].");
            if (inputs.Labels!.Shape.Length != 1 || inputs.Labels.Shape[0] != inputs.Points.Shape[0])
                throw new CliException("--labels must have shape [N] matching --points.");
            if (inputs.Labels.Values.Any(value => value is not 0f and not 1f))
                throw new CliException("--labels values must be 0 (negative) or 1 (positive).");
            ValidateCoordinates(inputs.Points.Values, "--points");
        }

        if (inputs.Box is not null)
        {
            if (!inputs.Box.Shape.SequenceEqual([4]))
                throw new CliException("--box must have shape [4] in x0,y0,x1,y1 order.");
            ValidateCoordinates(inputs.Box.Values, "--box");
            if (inputs.Box.Values[0] > inputs.Box.Values[2] || inputs.Box.Values[1] > inputs.Box.Values[3])
                throw new CliException("--box must satisfy x0 <= x1 and y0 <= y1.");
        }
    }

    public static SAMTorchSharp.Modeling.Sam3.Sam3Prompt CreateGeometryPrompt(Sam3GeometryInputs inputs)
    {
        ValidateGeometryInputs(inputs);
        Tensor? points = null;
        Tensor? labels = null;
        Tensor? boxes = null;

        if (inputs.Points is not null)
        {
            points = tensor(inputs.Points.Values, dtype: ScalarType.Float32, device: CPU)
                .reshape(inputs.Points.Shape[0], 1, 2) / ModelInputSize;
            labels = tensor(inputs.Labels!.Values, dtype: ScalarType.Float32, device: CPU)
                .to_type(ScalarType.Int64).reshape(inputs.Labels.Shape[0], 1);
        }

        if (inputs.Box is not null)
        {
            var values = inputs.Box.Values;
            var cx = (values[0] + values[2]) / (2 * ModelInputSize);
            var cy = (values[1] + values[3]) / (2 * ModelInputSize);
            var width = (values[2] - values[0]) / ModelInputSize;
            var height = (values[3] - values[1]) / ModelInputSize;
            boxes = tensor(new[] { cx, cy, width, height }, dtype: ScalarType.Float32, device: CPU)
                .reshape(1, 1, 4);
        }

        return new SAMTorchSharp.Modeling.Sam3.Sam3Prompt(
            box_embeddings: boxes,
            point_embeddings: points,
            point_labels: labels);
    }

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

    public static Sam3InferenceResult PostProcess(
        IReadOnlyDictionary<string, Tensor> outputs,
        long imageHeight,
        long imageWidth,
        float confidenceThreshold = 0.5f,
        int maxDetections = 20)
    {
        ValidateOutputs(outputs);
        if (imageHeight <= 0 || imageWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(imageHeight));
        if (confidenceThreshold is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(confidenceThreshold));
        if (maxDetections <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxDetections));

        using var scores = outputs["pred_logits"].squeeze(0).squeeze(-1).sigmoid();
        using var candidates = scores.ge(confidenceThreshold).nonzero().flatten();
        if (candidates.numel() == 0)
        {
            return new Sam3InferenceResult(
                zeros([0, 4], dtype: ScalarType.Float32, device: CPU),
                zeros([0], dtype: ScalarType.Float32, device: CPU),
                zeros([0, imageHeight, imageWidth], dtype: ScalarType.Float32, device: CPU));
        }

        using var candidateScores = scores.index_select(0, candidates);
        var count = Math.Min(maxDetections, checked((int)candidates.numel()));
        var (topValues, topIndices) = candidateScores.topk(count);
        using (topValues)
        using (topIndices)
        using (var indices = candidates.index_select(0, topIndices))
        using (var selectedScores = scores.index_select(0, indices))
        using (var selectedBoxes = outputs["pred_boxes"].squeeze(0).index_select(0, indices))
        using (var cx = selectedBoxes.select(1, 0))
        using (var cy = selectedBoxes.select(1, 1))
        using (var width = selectedBoxes.select(1, 2))
        using (var height = selectedBoxes.select(1, 3))
        using (var xyxy = stack([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2], dim: 1))
        using (var scale = tensor(new[] { (float)imageWidth, (float)imageHeight, (float)imageWidth, (float)imageHeight }))
        using (var scaledBoxes = xyxy * scale)
        using (var minimum = zeros_like(scaledBoxes))
        using (var maximum = tensor(new[] { (float)imageWidth, (float)imageHeight, (float)imageWidth, (float)imageHeight }).expand_as(scaledBoxes))
        using (var pixelBoxes = maximum.minimum(scaledBoxes.maximum(minimum)))
        using (var selectedMaskLogits = outputs["pred_masks"].squeeze(0).index_select(0, indices).unsqueeze(1))
        using (var resized = nn.functional.interpolate(selectedMaskLogits, size: [imageHeight, imageWidth], mode: InterpolationMode.Bilinear, align_corners: false))
        {
            return new Sam3InferenceResult(
                pixelBoxes.detach().cpu().clone(),
                selectedScores.detach().cpu().clone(),
                resized.squeeze(1).sigmoid().gt(0.5).to_type(ScalarType.Float32).detach().cpu().clone());
        }
    }

    private static void ValidateCoordinates(float[] values, string option)
    {
        for (var index = 0; index < values.Length; index++)
        {
            if (!float.IsFinite(values[index]) || values[index] < 0 || values[index] > ModelInputSize)
                throw new CliException($"{option} coordinates must be finite and within the 1008x1008 model input.");
        }
    }
}