using System.Diagnostics;
using System.Text.Json;
using SAMTorchSharp;
using SAMTorchSharp.Modeling.Sam2;
using TorchSharp;
using static TorchSharp.torch;

namespace ConsistencyTest;

internal sealed record Sam2ImageInputs(
    NpyArray Image,
    NpyArray? Points,
    NpyArray? Labels,
    NpyArray? Box,
    NpyArray? MaskInput = null);

internal static class Sam2ImageCommand
{
    public static Sam2ModelVariant ParseVariant(string name) => name.ToLowerInvariant() switch
    {
        "sam2-tiny" => Sam2ModelVariant.Sam2Tiny,
        "sam2-small" => Sam2ModelVariant.Sam2Small,
        "sam2.1-tiny" => Sam2ModelVariant.Sam21Tiny,
        "sam2.1-small" => Sam2ModelVariant.Sam21Small,
        _ => throw new CliException("--variant must be sam2-tiny, sam2-small, sam2.1-tiny, or sam2.1-small."),
    };

    public static Sam2ImageInputs LoadInputs(
        string imagePath,
        string? pointsPath,
        string? labelsPath,
        string? boxPath,
        string? maskInputPath)
    {
        var inputs = new Sam2ImageInputs(
            NpyFile.ReadFloat32(imagePath),
            pointsPath is null ? null : NpyFile.ReadFloat32(pointsPath),
            labelsPath is null ? null : NpyFile.ReadFloat32(labelsPath),
            boxPath is null ? null : NpyFile.ReadFloat32(boxPath),
            maskInputPath is null ? null : NpyFile.ReadFloat32(maskInputPath));
        ValidateInputs(inputs);
        return inputs;
    }

    public static void ValidateInputs(Sam2ImageInputs inputs)
    {
        if (inputs.Image.Shape.Length != 3 || inputs.Image.Shape[2] != 3 ||
            inputs.Image.Shape[0] <= 0 || inputs.Image.Shape[1] <= 0)
            throw new CliException("--image must be a non-empty float32 HWC array with shape [H,W,3].");
        if (inputs.Image.Values.Any(value => !float.IsFinite(value) || value < 0 || value > 1))
            throw new CliException("--image values must be finite and in the [0,1] range.");

        if ((inputs.Points is null) != (inputs.Labels is null))
            throw new CliException("--points and --labels must be supplied together.");
        if (inputs.Points is null && inputs.Box is null)
            throw new CliException("At least one point prompt (--points and --labels) or --box is required.");

        if (inputs.Points is not null)
        {
            if (inputs.Points.Shape.Length != 2 || inputs.Points.Shape[1] != 2 || inputs.Points.Shape[0] <= 0)
                throw new CliException("--points must have shape [N,2].");
            if (inputs.Labels!.Shape.Length != 1 || inputs.Labels.Shape[0] != inputs.Points.Shape[0])
                throw new CliException("--labels must have shape [N] matching --points.");
            if (inputs.Labels.Values.Any(value => value is not 0f and not 1f))
                throw new CliException("--labels values must be 0 (negative) or 1 (positive).");
            ValidateCoordinates(inputs.Points.Values, inputs.Image.Shape[0], inputs.Image.Shape[1], "--points");
        }

        if (inputs.Box is not null)
        {
            var validShape = inputs.Box.Shape.SequenceEqual([4]) || inputs.Box.Shape.SequenceEqual([2, 2]);
            if (!validShape)
                throw new CliException("--box must have shape [4] or [2,2] in x0,y0,x1,y1 order.");
            ValidateCoordinates(inputs.Box.Values, inputs.Image.Shape[0], inputs.Image.Shape[1], "--box");
            if (inputs.Box.Values[0] > inputs.Box.Values[2] || inputs.Box.Values[1] > inputs.Box.Values[3])
                throw new CliException("--box must satisfy x0 <= x1 and y0 <= y1.");
        }

        if (inputs.MaskInput is not null)
        {
            if (!inputs.MaskInput.Shape.SequenceEqual([1, 256, 256]))
                throw new CliException("--mask-input must have shape [1,256,256].");
            if (inputs.MaskInput.Values.Any(value => !float.IsFinite(value)))
                throw new CliException("--mask-input values must be finite.");
        }
    }

    public static int Run(
        Sam2ModelVariant variant,
        string variantName,
        string checkpointPath,
        string outputDirectory,
        Sam2ImageInputs inputs,
        bool multimask,
        bool returnLogits)
    {
        Directory.CreateDirectory(outputDirectory);
        Console.WriteLine($"Building {variantName} image predictor...");
        var model = Sam2ModelBuilder.Build(variant);
        using var predictor = new SAM2ImagePredictor(model);
        var report = Sam2CheckpointLoader.Load(model, checkpointPath, strict: true);

        using var image = ToTensor(inputs.Image);
        using var points = inputs.Points is null ? null : ToTensor(inputs.Points);
        using var labels = inputs.Labels is null ? null : ToTensor(inputs.Labels);
        using var box = inputs.Box is null ? null : ToTensor(inputs.Box);
        using var maskInput = inputs.MaskInput is null ? null : ToTensor(inputs.MaskInput);

        var stopwatch = Stopwatch.StartNew();
        predictor.SetImage(image);
        var (masksBatch, scoresBatch, logitsBatch) = predictor.Predict(
            points, labels, box, maskInput,
            multimaskOutput: multimask,
            returnLogits: returnLogits,
            normalizeCoordinates: true);
        stopwatch.Stop();

        using (masksBatch)
        using (scoresBatch)
        using (logitsBatch)
        using (var masks = masksBatch.squeeze(0))
        using (var scores = scoresBatch.squeeze(0))
        using (var logits = logitsBatch.squeeze(0))
        {
            var masksPath = Path.Combine(outputDirectory, "masks.npy");
            var scoresPath = Path.Combine(outputDirectory, "scores.npy");
            var logitsPath = Path.Combine(outputDirectory, "low_res_logits.npy");
            NpyFile.WriteFloat32(masksPath, masks);
            NpyFile.WriteFloat32(scoresPath, scores);
            NpyFile.WriteFloat32(logitsPath, logits);

            var summary = new
            {
                variant = variantName,
                checkpoint = checkpointPath,
                device = "cpu",
                imageShape = inputs.Image.Shape,
                pointCount = inputs.Points?.Shape[0] ?? 0,
                hasBox = inputs.Box is not null,
                hasMaskInput = inputs.MaskInput is not null,
                multimask,
                returnLogits,
                inferenceMilliseconds = stopwatch.ElapsedMilliseconds,
                outputs = new Dictionary<string, object>
                {
                    ["masks.npy"] = masks.shape,
                    ["scores.npy"] = scores.shape,
                    ["low_res_logits.npy"] = logits.shape,
                },
            };
            File.WriteAllText(
                Path.Combine(outputDirectory, "summary.json"),
                JsonSerializer.Serialize(summary, new JsonSerializerOptions { WriteIndented = true }));

            Console.WriteLine($"masks: [{string.Join(", ", masks.shape)}] -> {masksPath}");
            Console.WriteLine($"scores: [{string.Join(", ", scores.shape)}] -> {scoresPath}");
            Console.WriteLine($"low_res_logits: [{string.Join(", ", logits.shape)}] -> {logitsPath}");
        }

        Console.WriteLine($"Inference completed in {stopwatch.ElapsedMilliseconds} ms.");
        return 0;
    }

    private static Tensor ToTensor(NpyArray array) =>
        tensor(array.Values, dtype: ScalarType.Float32).reshape(array.Shape.Select(value => (long)value).ToArray());

    private static void ValidateCoordinates(float[] values, int height, int width, string option)
    {
        for (var index = 0; index < values.Length; index += 2)
        {
            var x = values[index];
            var y = values[index + 1];
            if (!float.IsFinite(x) || !float.IsFinite(y) || x < 0 || x > width || y < 0 || y > height)
                throw new CliException($"{option} coordinates must be finite and within the original image bounds.");
        }
    }
}