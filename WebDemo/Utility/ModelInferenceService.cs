using Microsoft.Extensions.Options;
using SAMTorchSharp;
using SAMTorchSharp.Modeling.Sam1;
using SAMTorchSharp.Modeling.Sam2;
using SAMTorchSharp.Modeling.Sam3;
using TorchSharp;
using WebDemo.Models;
using static TorchSharp.torch;
using static TorchSharp.torchvision;

namespace WebDemo.Utility;

public sealed class ModelInferenceService : IDisposable
{
    private readonly ModelOptions _options;
    private readonly ModelPathResolver _paths;
    private readonly ILogger<ModelInferenceService> _logger;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly io.SkiaImager _imager = new();
    private SamPredictor? _sam1;
    private Sam? _sam1Model;
    private SAM2ImagePredictor? _sam2;
    private Sam3BaseNew? _sam3;

    public ModelInferenceService(
        IOptions<ModelOptions> options,
        IWebHostEnvironment environment,
        ILogger<ModelInferenceService> logger)
    {
        _options = options.Value;
        _paths = new ModelPathResolver(_options, environment.ContentRootPath);
        _logger = logger;
        torchvision.io.DefaultImager = new io.SkiaImager(100);
    }

    public IReadOnlyList<ModelAvailability> GetAvailability() =>
    [
        Availability("sam1", "SAM 1 / MobileSAM", _paths.Sam1, "点或矩形"),
        Availability("sam2", $"SAM 2 ({_options.Sam2Variant})", _paths.Sam2, "点或矩形"),
        Availability("sam3", "SAM 3 detector-only", _paths.Sam3, "文本"),
    ];

    public async Task<PredictResponse> PredictAsync(ImageDataRequest request, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        var model = request.Model.Trim().ToLowerInvariant();
        if (model is not ("sam1" or "sam2" or "sam3"))
            throw new ArgumentException("Model must be sam1, sam2, or sam3.");
        if (string.IsNullOrWhiteSpace(request.Image))
            throw new ArgumentException("Please upload an image.");

        await _gate.WaitAsync(cancellationToken);
        try
        {
            using var image = DecodeImage(request.Image);
            return model switch
            {
                "sam1" => PredictSam1(image, request.Annotations),
                "sam2" => PredictSam2(image, request.Annotations),
                "sam3" => PredictSam3(image, request.Caption),
                _ => throw new InvalidOperationException("Unreachable model selection."),
            };
        }
        finally
        {
            _gate.Release();
        }
    }

    private PredictResponse PredictSam1(Tensor image, IReadOnlyList<Annotation> annotations)
    {
        EnsurePrompts(annotations);
        _sam1 ??= CreateSam1();
        _sam1.SetImage(image.unsqueeze(0));
        using var points = CreatePoints(annotations);
        using var labels = CreateLabels(annotations);
        using var box = CreateBox(annotations);
        // The legacy SAM 1 API uses null to represent an omitted prompt although its annotations are non-nullable.
        var (masks, scores, logits) = _sam1.Predict(box: box!, pointCoords: points!, pointLabels: labels!, multimaskOutput: true);
        using (masks)
        using (scores)
        using (logits)
        {
            return MaskResponse("sam1", image, masks, scores);
        }
    }

    private PredictResponse PredictSam2(Tensor image, IReadOnlyList<Annotation> annotations)
    {
        EnsurePrompts(annotations);
        _sam2 ??= CreateSam2();
        using var hwc = image.permute(1, 2, 0).contiguous();
        _sam2.SetImage(hwc);
        using var points = CreatePoints(annotations, ScalarType.Float32);
        using var labels = CreateLabels(annotations, ScalarType.Float32);
        using var box = CreateBox(annotations, ScalarType.Float32);
        var (masks, scores, logits) = _sam2.Predict(
            points, labels, box, multimaskOutput: true, returnLogits: false, normalizeCoordinates: true);
        using (masks)
        using (scores)
        using (logits)
        {
            return MaskResponse("sam2", image, masks, scores);
        }
    }

    private PredictResponse PredictSam3(Tensor image, string? caption)
    {
        if (string.IsNullOrWhiteSpace(caption))
            throw new ArgumentException("SAM 3 requires a text prompt.");
        _sam3 ??= CreateSam3();
        using var transformed = new SAM2Transforms(1008).__call(image).unsqueeze(0);
        var output = _sam3.Forward(transformed, [caption.Trim()]);
        try
        {
            var boxes = output["pred_boxes"];
            var logits = output["pred_logits"];
            return new PredictResponse(
                "sam3",
                null,
                "SAM 3 当前为 detector-only：以下是原始检测张量统计，不是最终实例 mask。",
                new
                {
                    caption = caption.Trim(),
                    predBoxesShape = boxes.shape,
                    predLogitsShape = logits.shape,
                    boxRange = new[] { boxes.min().item<float>(), boxes.max().item<float>() },
                    logitRange = new[] { logits.min().item<float>(), logits.max().item<float>() },
                });
        }
        finally
        {
            foreach (var tensor in output.Values.Distinct()) tensor.Dispose();
        }
    }

    private SamPredictor CreateSam1()
    {
        RequireCheckpoint(_paths.Sam1, "SAM 1");
        _logger.LogInformation("Loading SAM 1 checkpoint {Checkpoint}", _paths.Sam1);
        _sam1Model = BuildSam.BuildSAMVitT(_paths.Sam1);
        _sam1Model.eval();
        return new SamPredictor(_sam1Model);
    }

    private SAM2ImagePredictor CreateSam2()
    {
        RequireCheckpoint(_paths.Sam2, "SAM 2");
        var variant = ParseSam2Variant(_options.Sam2Variant);
        _logger.LogInformation("Loading SAM 2 checkpoint {Checkpoint} as {Variant}", _paths.Sam2, variant);
        var model = Sam2ModelBuilder.Build(variant);
        Sam2CheckpointLoader.Load(model, _paths.Sam2, strict: true);
        return new SAM2ImagePredictor(model);
    }

    private Sam3BaseNew CreateSam3()
    {
        RequireCheckpoint(_paths.Sam3, "SAM 3");
        _logger.LogInformation("Loading SAM 3 checkpoint {Checkpoint}", _paths.Sam3);
        var model = new BuildSam3New().Build();
        new Sam3CheckpointLoaderUniversal().LoadModel(model, _paths.Sam3, CPU);
        model.eval();
        return model;
    }

    private PredictResponse MaskResponse(string model, Tensor image, Tensor masksBatch, Tensor scores)
    {
        using var masks = masksBatch.squeeze(0);
        using var flatScores = scores.flatten();
        using var bestIndexTensor = flatScores.argmax();
        var bestIndex = bestIndexTensor.item<long>();
        using var bestMask = masks.dim() == 2 ? masks : masks[bestIndex];
        using var inverse = bestMask.logical_not().unsqueeze(0).expand_as(image);
        using var color = tensor(new byte[] { 0, 180, 255 }, dtype: ScalarType.Byte).reshape(3, 1, 1).expand_as(image);
        using var overlay = where(inverse, image, color);
        var png = _imager.EncodeImage(overlay.to_type(ScalarType.Byte), ImageFormat.Png);
        return new PredictResponse(
            model,
            $"data:image/png;base64,{Convert.ToBase64String(png)}",
            $"完成 {model.ToUpperInvariant()} 多 mask 推理，显示最高分 mask。",
            new { maskCount = masks.dim() == 2 ? 1 : masks.size(0), bestScore = flatScores[bestIndex].item<float>() });
    }

    private Tensor DecodeImage(string dataUrl)
    {
        var comma = dataUrl.IndexOf(',');
        var payload = comma >= 0 ? dataUrl[(comma + 1)..] : dataUrl;
        byte[] bytes;
        try { bytes = Convert.FromBase64String(payload); }
        catch (FormatException exception) { throw new ArgumentException("Image is not valid base64 data.", exception); }
        if (bytes.Length > 20 * 1024 * 1024) throw new ArgumentException("Image must not exceed 20 MB.");
        return _imager.DecodeImage(bytes, io.ImageReadMode.RGB);
    }

    private static Tensor? CreatePoints(IReadOnlyList<Annotation> annotations, ScalarType type = ScalarType.Int16)
    {
        var values = annotations.Where(x => x.Type is "foreground" or "background")
            .SelectMany(x => new float[] { x.X, x.Y }).ToArray();
        return values.Length == 0 ? null : tensor(values, dtype: type).reshape(-1, 2);
    }

    private static Tensor? CreateLabels(IReadOnlyList<Annotation> annotations, ScalarType type = ScalarType.Int16)
    {
        var values = annotations.Where(x => x.Type is "foreground" or "background")
            .Select(x => x.Type == "foreground" ? 1f : 0f).ToArray();
        return values.Length == 0 ? null : tensor(values, dtype: type);
    }

    private static Tensor? CreateBox(IReadOnlyList<Annotation> annotations, ScalarType type = ScalarType.Int16)
    {
        var box = annotations.FirstOrDefault(x => x.Type == "rectangle");
        return box is null ? null : tensor(new float[] { box.X1!.Value, box.Y1!.Value, box.X2!.Value, box.Y2!.Value }, dtype: type);
    }

    private static void EnsurePrompts(IReadOnlyList<Annotation> annotations)
    {
        if (annotations.Count == 0) throw new ArgumentException("SAM 1 and SAM 2 require at least one point or rectangle prompt.");
        if (annotations.Any(x => x.Type is not ("foreground" or "background" or "rectangle")))
            throw new ArgumentException("Annotation type must be foreground, background, or rectangle.");
        if (annotations.Count(x => x.Type == "rectangle") > 1) throw new ArgumentException("Only one rectangle prompt is supported.");
        var box = annotations.FirstOrDefault(x => x.Type == "rectangle");
        if (box is not null && (box.X1 is null || box.Y1 is null || box.X2 is null || box.Y2 is null))
            throw new ArgumentException("Rectangle annotations require x1, y1, x2, and y2.");
    }

    internal static Sam2ModelVariant ParseSam2Variant(string value) => value.Trim().ToLowerInvariant() switch
    {
        "sam2-tiny" => Sam2ModelVariant.Sam2Tiny,
        "sam2-small" => Sam2ModelVariant.Sam2Small,
        "sam2.1-tiny" => Sam2ModelVariant.Sam21Tiny,
        "sam2.1-small" => Sam2ModelVariant.Sam21Small,
        _ => throw new ArgumentException("Models:Sam2Variant must be sam2-tiny, sam2-small, sam2.1-tiny, or sam2.1-small."),
    };

    private static void RequireCheckpoint(string path, string model)
    {
        if (!File.Exists(path)) throw new FileNotFoundException($"{model} checkpoint was not found. Configure Models:WeightsDirectory or its checkpoint filename.", path);
    }

    private static ModelAvailability Availability(string id, string name, string path, string promptType) =>
        new(id, name, path, File.Exists(path), promptType);

    public void Dispose()
    {
        _sam1Model?.Dispose();
        _sam2?.Dispose();
        _sam3?.Dispose();
        _gate.Dispose();
    }
}