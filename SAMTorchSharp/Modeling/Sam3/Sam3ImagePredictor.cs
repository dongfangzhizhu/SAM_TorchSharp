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
/// Supports point, box, and text prompts.
/// </summary>
public class Sam3ImagePredictor : IDisposable
{
    private readonly Sam3Base _model;
    private bool _isImageSet = false;
    private Tensor? _imageEmbedding;
    private IList<Tensor>? _highResFeatures;
    private long[]? _origHw;
    private Device _device;
    private readonly float _maskThreshold;
    private readonly long _imageSize;

    private readonly (long h, long w)[] _bbFeatSizes = new[]
    {
        (288L, 288L),
        (144L, 144L),
        (72L, 72L)
    };

    public float MaskThreshold => _maskThreshold;
    public Device Device => _device;

    public Sam3ImagePredictor(
        Sam3Base model,
        float maskThreshold = 0.0f,
        long imageSize = 1008)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _device = model.device;
        _maskThreshold = maskThreshold;
        _imageSize = imageSize;
    }

    public void SetImage(Tensor image)
    {
        ResetPredictor();

        long h, w;
        if (image.dim() != 3)
            throw new ArgumentException("Image must be a 3D tensor [H,W,3] or [3,H,W].");

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
        _origHw = new long[] { h, w };

        Tensor inputImage = PreprocessImage(image);
        inputImage = inputImage.unsqueeze(0).to(_device);

        var backboneOut = _model.forward_backbone(inputImage);
        var visionFeats = ExtractVisionFeatures(backboneOut);

        // Add no_mem_embed if present
        var noMemField = _model.GetType().GetField("no_mem_embed");
        if (noMemField != null)
        {
            var raw = noMemField.GetValue(_model);
            if (raw is Tensor nm && nm.numel() > 0)
            {
                visionFeats[visionFeats.Count - 1] = visionFeats[visionFeats.Count - 1] + nm;
            }
        }

        var feats = new List<Tensor>();
        for (int i = visionFeats.Count - 1; i >= 0; i--)
        {
            var feat = visionFeats[i];
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

        _imageEmbedding = feats[feats.Count - 1];
        _highResFeatures = feats.Take(feats.Count - 1).ToList();
        _isImageSet = true;
    }

    public (Tensor Masks, Tensor IouPredictions, Tensor LowResMasks) Predict(
        Tensor? pointCoords = null,
        Tensor? pointLabels = null,
        Tensor? box = null,
        IList<string>? captions = null,
        bool multimaskOutput = true,
        bool returnLogits = false)
    {
        if (!_isImageSet)
            throw new InvalidOperationException("An image must be set with SetImage before prediction.");

        // Handle text prompts
        if (captions is not null && captions.Count > 0 && _model.tokenizer is not null && _model.text_encoder is not null)
        {
            var textTokens = _model.tokenizer.EncodeTexts(captions);
            var (textEmbeddings, _, _) = _model.text_encoder.ForwardEncoded(textTokens, _device);
            Console.WriteLine($"[Sam3ImagePredictor] Text embeddings shape: {string.Join("x", textEmbeddings.size())}");
        }

        // Process point and box prompts
        Tensor? unnormCoords = null;
        Tensor? labels = null;

        if (pointCoords is not null)
        {
            if (pointLabels is null)
                throw new ArgumentException("point_labels required if point_coords supplied.");

            unnormCoords = pointCoords.to(_device);
            labels = pointLabels.to(_device).to(ScalarType.Int32);

            if (unnormCoords.dim() == 2) unnormCoords = unnormCoords.unsqueeze(0);
            if (labels.dim() == 1) labels = labels.unsqueeze(0);
        }

        if (box is not null)
        {
            var unnormBox = box.to(_device).reshape(1, 2, 2);
            var boxLabels = tensor(new long[] { 2, 3 }, dtype: ScalarType.Int32, device: _device).reshape(1, 2);
            if (unnormCoords is not null)
            {
                unnormCoords = cat(new Tensor[] { unnormBox, unnormCoords }, dim: 1);
                labels = cat(new Tensor[] { boxLabels, labels }, dim: 1);
            }
            else
            {
                unnormCoords = unnormBox;
                labels = boxLabels;
            }
        }

        Console.WriteLine("[Sam3ImagePredictor] Prompts processed. Full pipeline integration pending.");

        var batchSize = 1;
        var numMasks = multimaskOutput ? 3 : 1;
        var origH = _origHw?[0] ?? _imageSize;
        var origW = _origHw?[1] ?? _imageSize;

        return (
            torch.zeros(new long[] { batchSize, numMasks, origH, origW }, device: _device),
            torch.zeros(new long[] { batchSize, numMasks }, device: _device),
            torch.zeros(new long[] { batchSize, numMasks, 256, 256 }, device: _device)
        );
    }

    private Tensor PreprocessImage(Tensor image)
    {
        Tensor inputImage = image.dtype == ScalarType.Byte
            ? image.to(ScalarType.Float32) / 255.0f
            : image.to(ScalarType.Float32);

        if (inputImage.size(0) == 3)
            inputImage = inputImage.permute(new long[] { 1, 2, 0 });

        inputImage = (inputImage - 0.5f) / 0.5f;

        var h = inputImage.size(0);
        var w = inputImage.size(1);

        if (h != _imageSize || w != _imageSize)
        {
            if (inputImage.dim() == 3)
            {
                inputImage = inputImage.unsqueeze(0);
                inputImage = functional.interpolate(inputImage, new long[] { _imageSize, _imageSize }, null, InterpolationMode.Bilinear, false);
                inputImage = inputImage.squeeze(0);
            }
            else
            {
                inputImage = functional.interpolate(inputImage, new long[] { _imageSize, _imageSize }, null, InterpolationMode.Bilinear, false);
            }
        }

        return inputImage;
    }

    private IList<Tensor> ExtractVisionFeatures(Dictionary<string, Tensor> backboneOut)
    {
        var features = new List<Tensor>();

        foreach (var (key, value) in backboneOut)
        {
            if ((key.Contains("features") || key.Contains("feats") || key == "backbone") && value.dim() >= 4)
            {
                features.Add(value);
            }
        }

        if (features.Count == 0)
        {
            foreach (var value in backboneOut.Values)
            {
                if (value.dim() >= 4)
                    features.Add(value);
            }
        }

        return features;
    }

    public void ResetPredictor()
    {
        _isImageSet = false;
        _imageEmbedding = null;
        _highResFeatures = null;
        _origHw = null;
    }

    public void Dispose()
    {
        if (_imageEmbedding is not null)
            _imageEmbedding.Dispose();
        if (_highResFeatures is not null)
            foreach (var f in _highResFeatures) f.Dispose();
    }
}
