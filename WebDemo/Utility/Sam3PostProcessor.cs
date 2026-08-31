using TorchSharp;
using static TorchSharp.torch;

namespace WebDemo.Utility;

public sealed record Sam3Detection(float Score, float[] Box);

public sealed class Sam3PostProcessResult : IDisposable
{
    public Sam3PostProcessResult(Tensor masks, IReadOnlyList<Sam3Detection> detections)
    {
        Masks = masks;
        Detections = detections;
    }

    public Tensor Masks { get; }
    public IReadOnlyList<Sam3Detection> Detections { get; }
    public void Dispose() => Masks.Dispose();
}

public static class Sam3PostProcessor
{
    public static Sam3PostProcessResult Process(
        Tensor boxes,
        Tensor logits,
        Tensor maskLogits,
        long imageHeight,
        long imageWidth,
        float confidenceThreshold = 0.5f,
        int maxDetections = 20)
    {
        if (boxes.shape is not [1, _, 4] || logits.shape is not [1, _, 1] || maskLogits.dim() != 4 || maskLogits.size(0) != 1)
            throw new ArgumentException("SAM 3 outputs must have shapes [1,Q,4], [1,Q,1], and [1,Q,H,W].");
        if (boxes.size(1) != logits.size(1) || boxes.size(1) != maskLogits.size(1))
            throw new ArgumentException("SAM 3 output query counts must match.");
        if (imageHeight <= 0 || imageWidth <= 0) throw new ArgumentOutOfRangeException(nameof(imageHeight));
        if (confidenceThreshold is < 0 or > 1) throw new ArgumentOutOfRangeException(nameof(confidenceThreshold));
        if (maxDetections <= 0) throw new ArgumentOutOfRangeException(nameof(maxDetections));

        using var scores = logits.squeeze(0).squeeze(-1).sigmoid();
        using var candidates = scores.ge(confidenceThreshold).nonzero().flatten();
        if (candidates.numel() == 0)
            return new Sam3PostProcessResult(zeros([0, imageHeight, imageWidth], dtype: ScalarType.Bool), []);

        using var candidateScores = scores.index_select(0, candidates);
        var count = Math.Min(maxDetections, checked((int)candidates.numel()));
        var (topValues, topIndices) = candidateScores.topk(count);
        using var topValuesScope = topValues;
        using var topIndicesScope = topIndices;
        using var indices = candidates.index_select(0, topIndices);
        using var selectedScores = scores.index_select(0, indices);
        using var selectedBoxes = boxes.squeeze(0).index_select(0, indices);
        using var cx = selectedBoxes.select(1, 0);
        using var cy = selectedBoxes.select(1, 1);
        using var width = selectedBoxes.select(1, 2);
        using var height = selectedBoxes.select(1, 3);
        using var halfWidth = width / 2;
        using var halfHeight = height / 2;
        using var xyxy = stack([cx - halfWidth, cy - halfHeight, cx + halfWidth, cy + halfHeight], dim: 1);
        using var scale = tensor(new[] { (float)imageWidth, (float)imageHeight, (float)imageWidth, (float)imageHeight });
        using var scaledBoxes = xyxy * scale;
        using var minimum = zeros_like(scaledBoxes);
        using var maximum = tensor(new[] { (float)imageWidth, (float)imageHeight, (float)imageWidth, (float)imageHeight }).expand_as(scaledBoxes);
        using var pixelBoxes = maximum.minimum(scaledBoxes.maximum(minimum));

        var detections = new List<Sam3Detection>(count);
        for (var i = 0; i < count; i++)
        {
            var box = new float[4];
            for (var coordinate = 0; coordinate < 4; coordinate++)
            {
                using var value = pixelBoxes[i, coordinate];
                box[coordinate] = value.item<float>();
            }
            using var score = selectedScores[i];
            detections.Add(new Sam3Detection(score.item<float>(), box));
        }

        using var selectedMaskLogits = maskLogits.squeeze(0).index_select(0, indices).unsqueeze(1);
        using var resized = nn.functional.interpolate(
            selectedMaskLogits,
            size: [imageHeight, imageWidth],
            mode: InterpolationMode.Bilinear,
            align_corners: false);
        return new Sam3PostProcessResult(resized.squeeze(1).sigmoid().gt(0.5), detections);
    }
}