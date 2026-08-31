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
    private readonly ModelPathResolver _paths;
    private readonly ILogger<ModelInferenceService> _logger;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly io.SkiaImager _imager = new();
    private SamPredictor? _sam;
    private Sam? _samModel;
    private SAM2ImagePredictor? _sam2;
    private SAM2ImagePredictor? _sam21;
    private Sam3BaseNew? _sam3;
    private Sam3BaseNew? _sam31;

    public ModelInferenceService(
        IOptions<ModelOptions> options,
        IWebHostEnvironment environment,
        ILogger<ModelInferenceService> logger)
    {
        _paths = new ModelPathResolver(options.Value, environment.ContentRootPath);
        _logger = logger;
        torchvision.io.DefaultImager = new io.SkiaImager(100);
    }

    public IReadOnlyList<ModelAvailability> GetAvailability() => _paths.All.Select(Availability).ToArray();

    public async Task<PredictResponse> PredictAsync(ImageDataRequest request, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        var model = request.Model.Trim().ToLowerInvariant();
        var configuredModel = _paths.Get(model);
        RequireCheckpoint(configuredModel);
        if (string.IsNullOrWhiteSpace(request.Image))
            throw new ArgumentException("Please upload an image.");

        await _gate.WaitAsync(cancellationToken);
        try
        {
            using var image = DecodeImage(request.Image);
            return model switch
            {
                "sam" or "sam1" => PredictSam(image, request.Annotations),
                "sam2" => PredictSam2(image, request.Annotations),
                "sam2.1" or "sam21" => PredictSam21(image, request.Annotations),
                "sam3" => PredictSam3(image, request.Caption, isSam31: false),
                "sam3.1" or "sam31" => PredictSam3(image, request.Caption, isSam31: true),
                _ => throw new InvalidOperationException("Unreachable model selection."),
            };
        }
        finally
        {
            _gate.Release();
        }
    }

    private PredictResponse PredictSam(Tensor image, IReadOnlyList<Annotation> annotations)
    {
        EnsurePrompts(annotations);
        _sam ??= CreateSam();
        _sam.SetImage(image.unsqueeze(0));
        using var points = CreatePoints(annotations);
        using var labels = CreateLabels(annotations);
        using var box = CreateBox(annotations);
        // The legacy SAM 1 API uses null to represent an omitted prompt although its annotations are non-nullable.
        var (masks, scores, logits) = _sam.Predict(box: box!, pointCoords: points!, pointLabels: labels!, multimaskOutput: true);
        using (masks)
        using (scores)
        using (logits)
        {
            return MaskResponse("sam", image, masks, scores);
        }
    }

    private PredictResponse PredictSam2(Tensor image, IReadOnlyList<Annotation> annotations)
        => PredictSam2(image, annotations, isSam21: false);

    private PredictResponse PredictSam21(Tensor image, IReadOnlyList<Annotation> annotations)
        => PredictSam2(image, annotations, isSam21: true);

    private PredictResponse PredictSam2(Tensor image, IReadOnlyList<Annotation> annotations, bool isSam21)
    {
        EnsurePrompts(annotations);
        var predictor = isSam21
            ? _sam21 ??= CreateSam2(_paths.Sam21, requireSam21: true)
            : _sam2 ??= CreateSam2(_paths.Sam2, requireSam21: false);
        using var hwc = image.permute(1, 2, 0).contiguous();
        predictor.SetImage(hwc);
        using var points = CreatePoints(annotations, ScalarType.Float32);
        using var labels = CreateLabels(annotations, ScalarType.Float32);
        using var box = CreateBox(annotations, ScalarType.Float32);
        var (masks, scores, logits) = predictor.Predict(
            points, labels, box, multimaskOutput: true, returnLogits: false, normalizeCoordinates: true);
        using (masks)
        using (scores)
        using (logits)
        {
            return MaskResponse(isSam21 ? "sam2.1" : "sam2", image, masks, scores);
        }
    }

    private PredictResponse PredictSam3(Tensor image, string? caption, bool isSam31)
    {
        if (string.IsNullOrWhiteSpace(caption))
            throw new ArgumentException("SAM 3 requires a text prompt.");
        var model = isSam31
            ? _sam31 ??= CreateSam3(_paths.Sam31)
            : _sam3 ??= CreateSam3(_paths.Sam3);
        using var transformed = new SAM2Transforms(1008).__call(image).unsqueeze(0);
        var output = model.Forward(transformed, [caption.Trim()]);
        try
        {
            var boxes = output["pred_boxes"];
            var logits = output["pred_logits"];
            var masks = output["pred_masks"];
            using var processed = Sam3PostProcessor.Process(boxes, logits, masks, image.size(1), image.size(2));
            using var overlay = RenderInstanceMasks(image, processed.Masks);
            var png = _imager.EncodeImage(overlay, ImageFormat.Png);
            return new PredictResponse(
                isSam31 ? "sam3.1" : "sam3",
                $"data:image/png;base64,{Convert.ToBase64String(png)}",
                $"{(isSam31 ? "SAM 3.1" : "SAM 3")} 文本分割完成，检测到 {processed.Detections.Count} 个实例。",
                new
                {
                    caption = caption.Trim(),
                    count = processed.Detections.Count,
                    detections = processed.Detections,
                });
        }
        finally
        {
            foreach (var tensor in output.Values.Distinct()) tensor.Dispose();
        }
    }

    private static Tensor RenderInstanceMasks(Tensor image, Tensor masks)
    {
        var overlay = image.clone();
        var colors = new byte[][]
        {
            [0, 180, 255], [255, 90, 90], [100, 220, 120], [255, 190, 60],
            [180, 100, 255], [255, 100, 200], [80, 210, 210], [220, 220, 80],
        };
        for (var i = 0L; i < masks.size(0); i++)
        {
            using var mask = masks[i].unsqueeze(0).expand_as(image);
            using var color = tensor(colors[i % colors.Length], dtype: ScalarType.Byte).reshape(3, 1, 1).expand_as(image);
            var next = where(mask, color, overlay);
            overlay.Dispose();
            overlay = next;
        }
        if (overlay.dtype == ScalarType.Byte) return overlay;
        var byteOverlay = overlay.to_type(ScalarType.Byte);
        overlay.Dispose();
        return byteOverlay;
    }

    private SamPredictor CreateSam()
    {
        var type = ModelPathResolver.ParseSamModel(_paths.Sam.Name);
        _logger.LogInformation("Loading SAM checkpoint {Checkpoint} as {Model}", _paths.Sam.CheckpointPath, type);
        _samModel = type switch
        {
            "vit_t" => BuildSam.BuildSAMVitT(_paths.Sam.CheckpointPath),
            "vit_b" => BuildSam.BuildSAMVitB(_paths.Sam.CheckpointPath),
            "vit_l" => BuildSam.BuildSAMVitL(_paths.Sam.CheckpointPath),
            "vit_h" => BuildSam.BuildSAMVitH(_paths.Sam.CheckpointPath),
            _ => throw new InvalidOperationException("Unreachable SAM model selection."),
        };
        _samModel.eval();
        return new SamPredictor(_samModel);
    }

    private SAM2ImagePredictor CreateSam2(ResolvedModel configuredModel, bool requireSam21)
    {
        var variant = ModelPathResolver.ParseSam2Model(configuredModel.Name, requireSam21);
        _logger.LogInformation("Loading {Model} checkpoint {Checkpoint} as {Variant}", configuredModel.DisplayName, configuredModel.CheckpointPath, variant);
        var model = Sam2ModelBuilder.Build(variant);
        Sam2CheckpointLoader.Load(model, configuredModel.CheckpointPath, strict: true);
        return new SAM2ImagePredictor(model);
    }

    private Sam3BaseNew CreateSam3(ResolvedModel configuredModel)
    {
        _logger.LogInformation("Loading {Model} checkpoint {Checkpoint}", configuredModel.DisplayName, configuredModel.CheckpointPath);
        var model = new BuildSam3New().Build();
        new Sam3CheckpointLoaderBinary().LoadModel(model, configuredModel.CheckpointPath, CPU);
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

    private static void RequireCheckpoint(ResolvedModel model)
    {
        if (!model.Configured)
            throw new ArgumentException($"{model.DisplayName} is not configured. Set Models:{ConfigurationKey(model.Id)}:Directory and Name.");
        if (!model.Available)
            throw new FileNotFoundException($"{model.DisplayName} checkpoint was not found. Check Models:{ConfigurationKey(model.Id)}:Directory and Name.", model.CheckpointPath);
    }

    private static ModelAvailability Availability(ResolvedModel model) => new(
        model.Id,
        string.IsNullOrWhiteSpace(model.Name) ? model.DisplayName : $"{model.DisplayName} ({model.Name})",
        model.CheckpointPath,
        model.Available,
        model.PromptType,
        !model.Configured ? "未配置" : model.Available ? "可用" : "权重不存在");

    private static string ConfigurationKey(string id) => id switch
    {
        "sam" => "Sam", "sam2" => "Sam2", "sam2.1" => "Sam21", "sam3" => "Sam3", "sam3.1" => "Sam31", _ => id,
    };

    public void Dispose()
    {
        _samModel?.Dispose();
        _sam2?.Dispose();
        _sam21?.Dispose();
        _sam3?.Dispose();
        _sam31?.Dispose();
        _gate.Dispose();
    }
}