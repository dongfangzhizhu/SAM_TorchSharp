// Copyright (c) Sapiens AI. All rights reserved.

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SAMTorchSharp.Modeling.Sam3;

/// <summary>
/// SAM3 VL Backbone - combines vision backbone (neck) with text encoder.
/// Ported from sam3/model/vl_combiner.py.
/// </summary>
public class Sam3VLBackbone : Module
{
    private readonly Sam3DualViTDetNeck visionBackbone;
    private readonly Sam3VETextEncoder languageBackbone;
    private readonly int scalp;

    public Sam3VLBackbone(
        Sam3DualViTDetNeck visual,
        Sam3VETextEncoder text,
        int scalp = 0)
        : base(nameof(Sam3VLBackbone))
    {
        this.visionBackbone = visual;
        this.languageBackbone = text;
        this.scalp = scalp;
    }

    /// <summary>
    /// Full forward pass processing both images and text captions.
    /// </summary>
    public Dictionary<string, object> Forward(
        Tensor samples,
        IList<string> captions,
        IList<Tensor>? inputBoxes = null,
        IList<string>? additionalText = null)
    {
        var output = new Dictionary<string, object>();
        var imageOutput = ForwardImage(samples);
        foreach (var kvp in imageOutput)
        {
            output[kvp.Key] = kvp.Value;
        }
        var device = imageOutput["vision_features"].device;
        
        var textOutput = ForwardText(captions, inputBoxes, additionalText, device);
        
        foreach (var kvp in textOutput)
        {
            output[kvp.Key] = kvp.Value;
        }
        
        return output;
    }

    /// <summary>
    /// Forward pass through the vision backbone (neck).
    /// Neck returns Tuple<List<Tensor>, List<Tensor>> = (features, positions).
    /// </summary>
    public Dictionary<string, Tensor> ForwardImage(Tensor samples)
    {
        var backboneResult = visionBackbone.forward(samples);
        var sam3Features = backboneResult.Item1;
        var sam3Pos = backboneResult.Item2;

        // Apply scalp: discard lowest resolution features
        if (scalp > 0 && sam3Features.Count > scalp)
        {
            sam3Features = sam3Features.GetRange(scalp, sam3Features.Count - scalp);
            sam3Pos = sam3Pos.GetRange(scalp, sam3Pos.Count - scalp);
        }

        var output = new Dictionary<string, Tensor>();

        // Last feature level
        var sam3Src = sam3Features[sam3Features.Count - 1];
        output["vision_features"] = sam3Src;
        output["vision_pos_enc"] = sam3Pos[sam3Pos.Count - 1];
        output["backbone_fpn"] = cat(sam3Features.ToArray(), dim: 0);

        return output;
    }

    /// <summary>
    /// Forward pass through the text encoder.
    /// </summary>
    public Dictionary<string, Tensor> ForwardText(
        IList<string> captions,
        IList<Tensor>? inputBoxes = null,
        IList<string>? additionalText = null,
        Device? device = null)
    {
        device ??= CPU;
        var output = new Dictionary<string, Tensor>();

        var textToEncode = new List<string>(captions);
        if (additionalText is not null)
        {
            textToEncode.AddRange(additionalText);
        }

        var (textAttentionMask, textMemory, textEmbeds) = languageBackbone.ForwardText(textToEncode, device);
        // textAttentionMask: [batch, seq_len]
        // textMemory: [seq_len, batch, d_model]
        // textEmbeds: [batch, seq_len, d_model]

        if (additionalText is not null && additionalText.Count > 0)
        {
            var extraStart = textMemory.size(0) - (long)additionalText.Count;
            output["additional_text_features"] = textMemory.narrow(0, (long)extraStart, textMemory.size(0) - extraStart);
            output["additional_text_mask"] = textAttentionMask.narrow(0, (long)additionalText.Count, textAttentionMask.size(0) - additionalText.Count);
        }

        var captionCount = captions.Count;
        output["language_features"] = textMemory.narrow(0, 0, (long)captionCount);
        output["language_mask"] = textAttentionMask.narrow(0, 0, (long)captionCount);
        output["language_embeds"] = textEmbeds.narrow(0, 0, (long)captionCount);

        return output;
    }

    /// <summary>
    /// Forward pass through the vision backbone only.
    /// </summary>
    public Dictionary<string, Tensor> ForwardVisionOnly(Tensor samples)
    {
        return ForwardImage(samples);
    }

    /// <summary>
    /// Generate dummy language features for vision-only mode.
    /// </summary>
    public Dictionary<string, Tensor> ForwardLanguageDummy(
        int batchCount,
        int nFeatures,
        Device? device = null)
    {
        device ??= CPU;
        return new Dictionary<string, Tensor>
        {
            { "language_features", zeros(new long[] { 0, (long)batchCount, (long)nFeatures }, device: device) },
            { "language_mask", zeros(new long[] { (long)batchCount, 0 }, device: device) }
        };
    }
}
