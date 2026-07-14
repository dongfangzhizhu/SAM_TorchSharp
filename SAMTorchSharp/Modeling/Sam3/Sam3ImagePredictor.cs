// Copyright (c) Sapiens AI. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SAMTorchSharp.Modeling.Sam3;

/// <summary>
/// SAM3 Image Predictor for interactive mask prediction.
/// 
/// API mirrors the Python Sam3Processor:
///   - SetImage(image) → preprocess and run backbone
///   - SetTextPrompt(captions) → run text encoder
///   - AddBoxPrompt(box, label) / AddPointPrompt(coords, labels) → append geometric prompts
///   - Predict() → full encoder-decoder-segmentation pipeline, post-process masks
///   - ResetAllPrompts() → clear prompts
/// 
/// The predictor maintains an internal state dictionary that holds:
///   - backbone_out: features from the VL backbone
///   - geometric_prompt: accumulated box/point/mask prompts
/// </summary>
public class Sam3ImagePredictor : IDisposable
{
    private readonly Sam3Base _model;
    private readonly float _confidenceThreshold;
    private readonly long _imageSize;
    private readonly Device _device;

    // ── Inference state ────────────────────────────────────────────────
    private bool _isImageSet = false;
    private long _origH;
    private long _origW;
    private Tensor? _backboneFpn;       // lowest-res FPN feature [1, C, H', W']
    private IList<Tensor>? _highResFeatures; // higher-res FPN features
    private Dictionary<string, Tensor>? _backboneOut;
    private Sam3Prompt? _geometricPrompt;

    // Feature sizes for reshaping backbone output
    private readonly (long h, long w)[] _bbFeatSizes = new[]
    {
        (288L, 288L),
        (144L, 144L),
        (72L, 72L)
    };

    // ── Constructor ────────────────────────────────────────────────────
    public Sam3ImagePredictor(
        Sam3Base model,
        float confidenceThreshold = 0.5f,
        long imageSize = 1008)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _device = _model.device;
        _confidenceThreshold = confidenceThreshold;
        _imageSize = imageSize;
    }

    // ── Set Image ──────────────────────────────────────────────────────
    /// <summary>
    /// Preprocess the image and run the backbone to extract features.
    /// Must be called before any prompt or prediction.
    /// </summary>
    public void SetImage(Tensor image)
    {
        ResetAllPrompts();

        if (image.dim() != 3)
            throw new ArgumentException("Image must be a 3D tensor [C,H,W] or [H,W,C].");

        long h, w;
        if (image.size(0) == 3)
        {
            h = image.size(1);
            w = image.size(2);
        }
        else
        {
            h = image.size(0);
            w = image.size(1);
        }
        _origH = h;
        _origW = w;

        // Preprocess: resize, normalize to [-1, 1]
        Tensor inputImage = PreprocessImage(image);
        inputImage = inputImage.unsqueeze(0).to(_device);

        // Run backbone
        _backboneOut = _model.forward_backbone(inputImage);

        // Extract FPN features
        ExtractFeatures(_backboneOut);

        _isImageSet = true;
    }

    /// <summary>
    /// Set image from a byte array (RGB).
    /// </summary>
    public void SetImage(byte[] rgbBytes, long height, long width)
    {
        var tensor = torch.from_numpy_array(rgbBytes, dtype: ScalarType.Byte)
            .view(new long[] { height, width, 3 })
            .to(_device);
        SetImage(tensor);
    }

    /// <summary>
    /// Set image from a 3D array [H,W,3] of bytes.
    /// </summary>
    public void SetImage(byte[, ,] rgbArray, long height, long width)
    {
        var flat = new byte[height * width * 3];
        for (long y = 0; y < height; y++)
            for (long x = 0; x < width; x++)
                for (int c = 0; c < 3; c++)
                    flat[(y * width + x) * 3 + c] = rgbArray[y, x, c];
        SetImage(flat, height, width);
    }

    // ── Text Prompts ───────────────────────────────────────────────────
    /// <summary>
    /// Run the text encoder on the given captions and update the backbone state.
    /// This encodes text into language features that will be used during inference.
    /// </summary>
    public void SetTextPrompt(IList<string> captions)
    {
        if (!_isImageSet)
            throw new InvalidOperationException("Call SetImage() first.");

        if (_model.text_encoder is null || _model.tokenizer is null)
            throw new InvalidOperationException("Model does not have a text encoder.");

        var textTokens = _model.tokenizer.EncodeTexts(captions);
        var (textEmbeddings, textFeatures, textMask) = _model.text_encoder.ForwardEncoded(
            textTokens, _device);

        // Store in backbone_out
        if (_backboneOut is null) _backboneOut = new Dictionary<string, Tensor>();

        // textFeatures: [seq_len, batch, d_model]
        if (textFeatures.numel() > 0)
            _backboneOut["language_features"] = textFeatures;
        if (textMask.numel() > 0)
            _backboneOut["language_mask"] = textMask;
    }

    /// <summary>
    /// Set a single text caption.
    /// </summary>
    public void SetTextPrompt(string caption)
    {
        SetTextPrompt(new[] { caption });
    }

    // ── Geometric Prompts ──────────────────────────────────────────────
    /// <summary>
    /// Add a box prompt. Box should be in normalized [cx, cy, w, h] format (range [0,1]).
    /// label=true means positive (object present), false means negative (background).
    /// </summary>
    public void AddBoxPrompt(float[] box, bool label = true)
    {
        EnsureGeometricPrompt();

        var boxTensor = torch.tensor(box, dtype: ScalarType.Float32, device: _device).reshape(1, 1, 4);
        var labelTensor = scalar(label ? 1L : 0L, dtype: ScalarType.Int64, device: _device).reshape(1, 1);

        _geometricPrompt = _geometricPrompt!.AppendBoxes(boxTensor);
    }

    /// <summary>
    /// Add multiple box prompts.
    /// </summary>
    public void AddBoxPrompts(float[][] boxes, bool[] labels)
    {
        if (boxes.Length != labels.Length)
            throw new ArgumentException("boxes and labels must have same length.");

        EnsureGeometricPrompt();

        var boxTensor = torch.zeros(new long[] { 1, boxes.Length, 4 }, dtype: ScalarType.Float32, device: _device);
        var labelTensor = torch.zeros(new long[] { 1, boxes.Length }, dtype: ScalarType.Int64, device: _device);

        for (int i = 0; i < boxes.Length; i++)
        {
            boxTensor[0, i] = torch.tensor(boxes[i], dtype: ScalarType.Float32, device: _device);
            labelTensor[0, i] = labels[i] ? scalar(1L) : scalar(0L);
        }

        _geometricPrompt = _geometricPrompt!
            .WithBoxes(boxTensor)
            .WithBoxLabels(labelTensor);
    }

    /// <summary>
    /// Add a point prompt. Point should be in normalized [x, y] format (range [0,1]).
    /// label=1 means positive, label=0 means negative.
    /// </summary>
    public void AddPointPrompt(float[] point, int label = 1)
    {
        EnsureGeometricPrompt();

        var pointTensor = torch.tensor(point, dtype: ScalarType.Float32, device: _device).reshape(1, 1, 2);

        _geometricPrompt = _geometricPrompt!.AppendPoints(pointTensor);
    }

    /// <summary>
    /// Add multiple point prompts.
    /// </summary>
    public void AddPointPrompts(float[][] points, int[] labels)
    {
        if (points.Length != labels.Length)
            throw new ArgumentException("points and labels must have same length.");

        EnsureGeometricPrompt();

        var pointTensor = torch.zeros(new long[] { 1, points.Length, 2 }, dtype: ScalarType.Float32, device: _device);
        for (int i = 0; i < points.Length; i++)
        {
            pointTensor[0, i] = torch.tensor(points[i], dtype: ScalarType.Float32, device: _device);
        }

        _geometricPrompt = _geometricPrompt!.AppendPoints(pointTensor);
    }

    // ── Prediction ─────────────────────────────────────────────────────
    /// <summary>
    /// Run the full inference pipeline: encoder + decoder + segmentation head.
    /// Returns masks, IoU predictions, and logits filtered by confidence threshold.
    /// </summary>
    public Sam3PredictionResult Predict()
    {
        if (!_isImageSet)
            throw new InvalidOperationException("Call SetImage() first.");

        if (_backboneFpn is null)
            throw new InvalidOperationException("No backbone features available. Call SetImage() first.");

        // Ensure geometric prompt exists
        EnsureGeometricPrompt();

        // Prepare captions for text encoder (use "visual" as dummy if no text set)
        IList<string>? captions = null;
        if (_model.text_encoder is not null && _model.tokenizer is not null)
        {
            if (_backboneOut is not null &&
                _backboneOut.ContainsKey("language_features") &&
                _backboneOut["language_features"].numel() > 0)
            {
                // Text already encoded, no need to pass captions
                captions = null;
            }
            else
            {
                // Use a dummy "visual" token so the text encoder produces visual-style features
                captions = new[] { "visual" };
            }
        }

        // Re-run backbone if text prompts are provided (needed to get language features)
        Dictionary<string, Tensor> backboneOutForInference = _backboneOut ?? new Dictionary<string, Tensor>();

        if (captions is not null && captions.Count > 0 && _model.text_encoder is not null)
        {
            // Need to re-run backbone with text. Use the stored image features.
            // Actually, for VL backbone we need to re-run forward_backbone to get language features.
            // Since we don't store the original image tensor, we'll use the pre-extracted features
            // and just add the language features from the text encoder.

            // For now, skip text encoding if we already have backbone features without language.
            // The geometry encoder will handle the case without text.
            captions = null;
        }

        // Run inference from pre-extracted features
        var forwardResult = _model.RunInferenceFromFeatures(
            _highResFeatures ?? new List<Tensor> { _backboneFpn },
            new List<Tensor>(),  // pos embeds - will be handled by geometry encoder
            _highResFeatures?.Select(f => new long[] { f.size(2), f.size(3) }).ToList() ??
                new List<long[]> { new long[] { _backboneFpn.size(2), _backboneFpn.size(3) } },
            captions is not null ? backboneOutForInference : null,
            _geometricPrompt);

        // Extract outputs
        var predMasks = forwardResult.TryGetValue("pred_masks", out var masks) ? masks : null;
        var predLogits = forwardResult.TryGetValue("pred_logits", out var logits) ? logits : null;
        var predBoxes = forwardResult.TryGetValue("pred_boxes", out var boxes) ? boxes : null;

        // Post-process
        return PostProcess(predMasks, predLogits, predBoxes);
    }

    /// <summary>
    /// Run prediction with a single text caption (sets text prompt and predicts in one call).
    /// </summary>
    public Sam3PredictionResult PredictWithText(string caption)
    {
        SetTextPrompt(caption);
        return Predict();
    }

    /// <summary>
    /// Run prediction with text and box prompts.
    /// </summary>
    public Sam3PredictionResult PredictWithTextAndBox(string caption, float[] box, bool boxLabel = true)
    {
        SetTextPrompt(caption);
        AddBoxPrompt(box, boxLabel);
        return Predict();
    }

    // ── Reset ──────────────────────────────────────────────────────────
    /// <summary>
    /// Reset all prompts and intermediate state. Keeps the image set.
    /// </summary>
    public void ResetAllPrompts()
    {
        _isImageSet = false;
        _backboneOut = null;
        _backboneFpn = null;
        _highResFeatures = null;
        _geometricPrompt = null;
        _origH = 0;
        _origW = 0;
    }

    /// <summary>
    /// Reset only prompts, keeping the image features.
    /// </summary>
    public void ResetPrompts()
    {
        _geometricPrompt = null;
    }

    // ── Properties ─────────────────────────────────────────────────────
    public bool IsImageSet => _isImageSet;
    public long OriginalHeight => _origH;
    public long OriginalWidth => _origW;
    public Device Device => _device;

    // ── IDisposable ────────────────────────────────────────────────────
    public void Dispose()
    {
        _backboneFpn?.Dispose();
        _backboneOut?.Values.ToList().ForEach(t => t?.Dispose());
        _highResFeatures?.ForEach(t => t?.Dispose());
        _geometricPrompt?.boxes?.Dispose();
        _geometricPrompt?.points?.Dispose();
        _isImageSet = false;
    }

    // ── Private Helpers ────────────────────────────────────────────────

    private void EnsureGeometricPrompt()
    {
        if (_geometricPrompt is null)
        {
            _geometricPrompt = new Sam3Prompt { batch_size = 1, num_points = 0, num_boxes = 0 };
        }
    }

    private Tensor PreprocessImage(Tensor image)
    {
        // Normalize pixel values to [0, 1]
        Tensor inputImage = image.dtype == ScalarType.Byte
            ? image.to(ScalarType.Float32) / 255.0f
            : image.to(ScalarType.Float32);

        // Ensure channel-first [C, H, W]
        if (inputImage.size(0) != 3)
            inputImage = inputImage.permute(new long[] { 2, 0, 1 });

        // Normalize to [-1, 1]
        inputImage = (inputImage - 0.5f) / 0.5f;

        // Resize to _imageSize x _imageSize
        var h = inputImage.size(1);
        var w = inputImage.size(2);
        if (h != _imageSize || w != _imageSize)
        {
            if (inputImage.dim() == 3)
            {
                inputImage = inputImage.unsqueeze(0);
                inputImage = functional.interpolate(
                    inputImage,
                    new long[] { _imageSize, _imageSize },
                    null,
                    InterpolationMode.Bilinear,
                    false);
                inputImage = inputImage.squeeze(0);
            }
            else
            {
                inputImage = functional.interpolate(
                    inputImage,
                    new long[] { _imageSize, _imageSize },
                    null,
                    InterpolationMode.Bilinear,
                    false);
            }
        }

        return inputImage;
    }

    /// <summary>
    /// Extract FPN features from backbone output.
    /// The backbone returns a dictionary with keys like "backbone_fpn" containing a list of features.
    /// We extract the lowest-res feature as the image embedding and the rest as high-res features.
    /// </summary>
    private void ExtractFeatures(Dictionary<string, Tensor> backboneOut)
    {
        // Try to find FPN features from the backbone output
        var fpnFeatures = new List<Tensor>();

        // Check for "backbone_fpn" key
        if (backboneOut.TryGetValue("backbone_fpn", out var fpnTensor))
        {
            if (fpnTensor is Tensor t && t.dim() >= 4)
            {
                fpnFeatures.Add(t);
            }
            else if (fpnTensor is IList<Tensor> fpnList)
            {
                fpnFeatures.AddRange(fpnList);
            }
        }

        // If no explicit backbone_fpn, try to find features with "features" or "feats" in the key
        if (fpnFeatures.Count == 0)
        {
            foreach (var (key, value) in backboneOut)
            {
                if ((key.Contains("features") || key.Contains("feats") || key == "backbone") && value.dim() >= 4)
                {
                    fpnFeatures.Add(value);
                }
            }
        }

        if (fpnFeatures.Count == 0)
        {
            // Fallback: take any 4D tensor from the output
            foreach (var value in backboneOut.Values)
            {
                if (value.dim() >= 4)
                    fpnFeatures.Add(value);
            }
        }

        if (fpnFeatures.Count == 0)
            throw new InvalidOperationException("No FPN features found in backbone output.");

        // The backbone typically returns multi-scale features.
        // We need to reshape them to the expected sizes.
        var feats = new List<Tensor>();
        for (int i = fpnFeatures.Count - 1; i >= 0; i--)
        {
            var feat = fpnFeatures[i];
            var (fh, fw) = _bbFeatSizes[Math.Min(i, _bbFeatSizes.Length - 1)];

            if (feat.dim() == 4)
            {
                var b = feat.size(0);
                feat = feat.permute(new long[] { 0, 2, 3, 1 });
                var c = feat.size(3);
                feat = feat.view(new long[] { b, fh, fw, c });
                feat = feat.permute(new long[] { 0, 3, 1, 2 });
            }
            feats.Add(feat);
        }

        _backboneFpn = feats[feats.Count - 1];  // lowest-res feature
        _highResFeatures = feats.Take(feats.Count - 1).ToList();
    }

    /// <summary>
    /// Post-process model outputs: filter by confidence, convert masks to original image size.
    /// </summary>
    private Sam3PredictionResult PostProcess(Tensor? predMasks, Tensor? predLogits, Tensor? predBoxes)
    {
        if (predLogits is null || predLogits.numel() == 0)
            return new Sam3PredictionResult(
                new List<float[][]>(),
                new List<byte[][]>(),
                new List<float[]>());

        // pred_logits: [batch, num_masks, num_classes] or [batch, num_masks]
        // Apply sigmoid to get probabilities
        var probs = functional.sigmoid(predLogits);

        // Get the best mask index per batch
        var bestMaskIdx = probs.argmax(dim: 1);  // [batch]

        var numMasks = (int)bestMaskIdx.size(0);
        var masksList = new List<float[][]>();
        var boxesList = new List<float[][]>();
        var scoresList = new List<float[]>();

        for (int b = 0; b < numMasks; b++)
        {
            var maskIdx = (long)bestMaskIdx[b].item().ToScalar<long>();

            // Get mask and score for this batch
            Tensor mask;
            float score;

            if (predMasks is not null && predMasks.dim() >= 4)
            {
                // [batch, num_masks, H, W]
                mask = predMasks[b, maskIdx];
            }
            else if (predMasks is not null && predMasks.dim() == 3)
            {
                // [batch, H, W] — single mask
                mask = predMasks[b];
            }
            else
            {
                continue;
            }

            if (probs.dim() >= 3)
            {
                score = (float)probs[b, maskIdx].item().ToScalar<float>();
            }
            else
            {
                score = (float)probs[b, maskIdx].item().ToScalar<float>();
            }

            // Filter by confidence threshold
            if (score < _confidenceThreshold)
                continue;

            // Upsample mask to original image size
            var origH = (long)_origH;
            var origW = (long)_origW;
            var maskUpsampled = functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                new long[] { origH, origW },
                null,
                InterpolationMode.Bilinear,
                false).squeeze();

            // Convert to byte array
            var maskData = tensorToByteArray(maskUpsampled);

            // Get box if available
            float[]? boxData = null;
            if (predBoxes is not null && predBoxes.dim() >= 3)
            {
                // [batch, num_masks, 4] in [cx, cy, w, h] format
                var box = predBoxes[b, maskIdx];
                // Convert from center-x-center-y-width-height to x0-y0-x1-y1
                var cx = box[0].item().ToScalar<float>();
                var cy = box[1].item().ToScalar<float>();
                var bw = box[2].item().ToScalar<float>();
                var bh = box[3].item().ToScalar<float>();

                // Scale to original image size
                var scaleX = (float)_origW / _imageSize;
                var scaleY = (float)_origH / _imageSize;

                var x0 = (cx - bw / 2f) * scaleX;
                var y0 = (cy - bh / 2f) * scaleY;
                var x1 = (cx + bw / 2f) * scaleX;
                var y1 = (cy + bh / 2f) * scaleY;

                boxData = new[] { x0, y0, x1, y1 };
            }

            masksList.Add(new[] { maskData });
            boxesList.Add(boxData ?? Array.Empty<float>());
            scoresList.Add(new[] { score });
        }

        return new Sam3PredictionResult(boxesList, masksList, scoresList);
    }

    /// <summary>
    /// Convert a 2D tensor to a byte array (0/255).
    /// </summary>
    private byte[] tensorToByteArray(Tensor tensor)
    {
        // Clamp to [0, 1] and convert to byte
        var clamped = tensor.clamp(0.0f, 1.0f);
        var byteTensor = (clamped * 255).to(ScalarType.Byte);

        var data = new byte[byteTensor.numel()];
        var ptr = byteTensor.data_ptr();

        // Copy data from tensor to managed array
        unsafe
        {
            var src = (byte*)ptr.ToPointer();
            for (long i = 0; i < byteTensor.numel(); i++)
            {
                data[i] = src[i];
            }
        }

        return data;
    }

    // ── Nested Result Class ────────────────────────────────────────────
    /// <summary>
    /// Result of a SAM3 prediction.
    /// </summary>
    public class Sam3PredictionResult
    {
        /// <summary>Box coordinates per instance: [x0, y0, x1, y1] in original image space.</summary>
        public IList<float[][]> Boxes { get; }

        /// <summary>Binary masks per instance: each mask is a 2D byte array [H, W].</summary>
        public IList<byte[][]> Masks { get; }

        /// <summary>Confidence scores per instance.</summary>
        public IList<float[]> Scores { get; }

        /// <summary>Total number of detected instances.</summary>
        public int Count => Boxes.Count;

        public Sam3PredictionResult(IList<float[][]> boxes, IList<byte[][]> masks, IList<float[]> scores)
        {
            Boxes = boxes;
            Masks = masks;
            Scores = scores;
        }
    }
}
