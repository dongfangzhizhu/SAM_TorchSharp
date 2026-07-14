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
/// SAM3 Base model integrating all components including text encoder and VL backbone.
/// Ported from sam3/model/sam3_image.py Sam3Image class.
/// </summary>
public class Sam3Base : Module
{
    /// <summary>
    /// VL Backbone combining vision (neck) and text encoder.
    /// </summary>
    public readonly Sam3VLBackbone? vl_backbone;

    /// <summary>
    /// Vision-only backbone (neck). Used when no text encoder is needed.
    /// </summary>
    public readonly Sam3DualViTDetNeck? neck;

    public readonly Sam3GeometryEncoder geometry_encoder;
    public readonly Sam3TransformerEncoder transformer_encoder;
    public readonly Sam3TransformerDecoder transformer_decoder;
    public readonly Sam3SegmentationHead? segmentation_head;

    /// <summary>
    /// Dot-product scoring head for classification.
    /// </summary>
    public readonly Module<Tensor, Tensor, Tensor?, Tensor>? dot_prod_scoring;

    public readonly int hidden_dim;
    public readonly int num_feature_levels;
    public readonly bool multimask_output;
    public readonly bool use_dot_prod_scoring;
    public readonly Sam3VETextEncoder? text_encoder;
    public readonly Sam3TokenizerVE? tokenizer;

    private Device? _device_cache = null;

    /// <summary>
    /// Construct Sam3Base with full VL backbone (vision + text).
    /// </summary>
    public Sam3Base(
        Sam3VLBackbone vlBackbone,
        Sam3GeometryEncoder geometryEncoder,
        Sam3TransformerEncoder transformerEncoder,
        Sam3TransformerDecoder transformerDecoder,
        Module? segmentationHead = null,
        Module<Tensor, Tensor, Tensor?, Tensor>? dotProdScoring = null,
        int numFeatureLevels = 4,
        bool multimaskOutput = true,
        bool useDotProdScoring = true,
        Sam3VETextEncoder? textEncoder = null,
        Sam3TokenizerVE? tokenizer = null)
        : base(nameof(Sam3Base))
    {
        this.vl_backbone = vlBackbone;
        this.neck = null;
        this.geometry_encoder = geometryEncoder;
        this.transformer_encoder = transformerEncoder;
        this.transformer_decoder = transformerDecoder;
        this.segmentation_head = (Sam3SegmentationHead?)segmentationHead;
        this.dot_prod_scoring = dotProdScoring;
        this.hidden_dim = transformerDecoder.d_model;
        this.num_feature_levels = numFeatureLevels;
        this.multimask_output = multimaskOutput;
        this.use_dot_prod_scoring = useDotProdScoring;
        this.text_encoder = textEncoder;
        this.tokenizer = tokenizer;
    }

    /// <summary>
    /// Construct Sam3Base with vision-only backbone (no text encoder).
    /// </summary>
    public Sam3Base(
        Sam3DualViTDetNeck neck,
        Sam3GeometryEncoder geometryEncoder,
        Sam3TransformerEncoder transformerEncoder,
        Sam3TransformerDecoder transformerDecoder,
        Module? segmentationHead = null,
        Module<Tensor, Tensor, Tensor?, Tensor>? dotProdScoring = null,
        int numFeatureLevels = 4,
        bool multimaskOutput = true,
        bool useDotProdScoring = true)
        : base(nameof(Sam3Base))
    {
        this.vl_backbone = null;
        this.neck = neck;
        this.geometry_encoder = geometryEncoder;
        this.transformer_encoder = transformerEncoder;
        this.transformer_decoder = transformerDecoder;
        this.segmentation_head = (Sam3SegmentationHead?)segmentationHead;
        this.dot_prod_scoring = dotProdScoring;
        this.hidden_dim = transformerDecoder.d_model;
        this.num_feature_levels = numFeatureLevels;
        this.multimask_output = multimaskOutput;
        this.use_dot_prod_scoring = useDotProdScoring;
        this.text_encoder = null;
        this.tokenizer = null;
    }

    public Device device
    {
        get
        {
            if (_device_cache is null)
            {
                var p = parameters().FirstOrDefault();
                _device_cache = p?.device;
            }
            return _device_cache ?? CPU;
        }
    }

    /// <summary>
    /// Forward pass through the vision backbone only.
    /// </summary>
    public Dictionary<string, Tensor> forward_backbone(Tensor images)
    {
        if (vl_backbone is not null)
        {
            var result = vl_backbone.ForwardVisionOnly(images);
            return result;
        }
        if (neck is not null)
        {
            var backboneOut = neck.forward(images);
            var dict = new Dictionary<string, Tensor>();
            dict.Add("backbone_out", backboneOut.Item1[backboneOut.Item1.Count - 1]);
            return dict;
        }
        throw new InvalidOperationException("No backbone configured.");
    }

    /// <summary>
    /// Full forward pass: encode image + text, run encoder/decoder pipeline.
    /// </summary>
    public Dictionary<string, Tensor> Forward(
        Tensor images,
        IList<string>? captions = null,
        Sam3Prompt? geometricPrompt = null)
    {
        // 1. Run VL backbone (vision + text) or vision-only
        Dictionary<string, Tensor> backboneOut = new();
        IList<Tensor> imgFeats;
        IList<Tensor> imgPosEmbeds;
        IList<long[]> visFeatSizes;

        if (vl_backbone is not null && captions is not null && captions.Count > 0)
        {
            var vlResult = vl_backbone.Forward(images, captions);
            foreach (var kvp in vlResult)
            {
                if (kvp.Value is Tensor t)
                    backboneOut[kvp.Key] = t;
            }

            if (backboneOut.TryGetValue("backbone_fpn", out var fpnTensor))
            {
                imgFeats = new List<Tensor> { fpnTensor };
                imgPosEmbeds = backboneOut.TryGetValue("vision_pos_enc", out var pos)
                    ? new List<Tensor> { pos }
                    : new List<Tensor>();
                visFeatSizes = imgFeats.Select(f => new long[] { f.size(2), f.size(3) }).ToList();
            }
            else
            {
                imgFeats = new List<Tensor>();
                imgPosEmbeds = new List<Tensor>();
                visFeatSizes = new List<long[]>();
            }
        }
        else
        {
            if (neck is not null)
            {
                var backboneOutTuple = neck.forward(images);
                imgFeats = backboneOutTuple.Item1;
                imgPosEmbeds = backboneOutTuple.Item2;
                visFeatSizes = imgFeats.Select(f => new long[] { f.size(2), f.size(3) }).ToList();
            }
            else
            {
                imgFeats = new List<Tensor>();
                imgPosEmbeds = new List<Tensor>();
                visFeatSizes = new List<long[]>();
            }
        }

        // 2. Encode geometric prompts
        Sam3Prompt gp = geometricPrompt ?? new Sam3Prompt();

        var (geoFeats, geoMasks) = geometry_encoder.forward(
            gp, imgFeats.ToList(), visFeatSizes.ToList());

        // 3. Combine text + geometry prompts
        Tensor prompt;
        Tensor? promptMask;

        if (backboneOut.TryGetValue("language_features", out var langFeat) &&
            backboneOut.TryGetValue("language_mask", out var langMask))
        {
            // langFeat: [seq_len, batch, d_model], geoFeats: list of Tensors per level
            // Concat geoFeats along seq dim
            Tensor geoCombined = geoFeats[0];
            for (int i = 1; i < geoFeats.Count; i++)
            {
                geoCombined = cat(new[] { geoCombined, geoFeats[i] }, dim: 0);
            }

            var promptList = new List<Tensor> { langFeat, geoCombined };
            prompt = cat(promptList.ToArray(), dim: 0);

            var maskList = new List<Tensor> { langMask };
            foreach (var gm in geoMasks)
            {
                if (gm is not null) maskList.Add((Tensor)gm);
            }
            promptMask = torch.cat(maskList.ToArray(), dim: 1);
        }
        else
        {
            Tensor geoCombined = geoFeats[0];
            for (int i = 1; i < geoFeats.Count; i++)
            {
                geoCombined = cat(new[] { geoCombined, geoFeats[i] }, dim: 0);
            }
            prompt = geoCombined;
            promptMask = geoMasks.Count > 0 && geoMasks[0] is not null ? geoMasks[0] : null;
        }

        // 4. Run transformer encoder
        var memory = transformer_encoder.forward(
            imgFeats,
            promptMask is not null ? new List<Tensor> { promptMask } : null,
            imgPosEmbeds);

        // 5. Run transformer decoder
        var posEmbed = (Tensor)memory["pos_embed"];
        var memTensor = (Tensor)memory["memory"];
        var padMask = memory["padding_mask"] as Tensor;

        var decoderResult = run_decoder(
            posEmbed,
            memTensor,
            padMask,
            prompt,
            promptMask);

        var resultDict = decoderResult.Item1;
        var hs = decoderResult.Item2;

        // 6. Run segmentation head if available
        if (segmentation_head is not null)
        {
            var segResult = segmentation_head.forward(
                new List<Tensor> { imgFeats.Last() },
                hs,
                null,  // encoder_hidden_states
                prompt,
                promptMask);
            resultDict["pred_masks"] = segResult["pred_masks"];
        }

        return resultDict;
    }

    /// <summary>
    /// Run the encoder-decoder-segmentation pipeline using pre-extracted image features.
    /// This is used by the predictor to avoid re-running the backbone.
    /// </summary>
    public Dictionary<string, Tensor> RunInferenceFromFeatures(
        IList<Tensor> imgFeats,
        IList<Tensor> imgPosEmbeds,
        IList<long[]> visFeatSizes,
        IDictionary<string, Tensor>? extraBackboneFeatures = null,
        Sam3Prompt? geometricPrompt = null)
    {
        var gp = geometricPrompt ?? new Sam3Prompt();

        // 1. Encode geometric prompts
        var (geoFeats, geoMasks) = geometry_encoder.forward(gp, imgFeats.ToList(), visFeatSizes.ToList());

        // 2. Get text features from extra backbone features
        Tensor prompt;
        Tensor? promptMask;

        if (extraBackboneFeatures is not null &&
            extraBackboneFeatures.TryGetValue("language_features", out var langFeat) &&
            extraBackboneFeatures.TryGetValue("language_mask", out var langMask))
        {
            // langFeat: [seq_len, batch, d_model], geoFeats: list of Tensors per level
            Tensor geoCombined = geoFeats[0];
            for (int i = 1; i < geoFeats.Count; i++)
            {
                geoCombined = cat(new[] { geoCombined, geoFeats[i] }, dim: 0);
            }

            var promptList = new List<Tensor> { langFeat, geoCombined };
            prompt = cat(promptList.ToArray(), dim: 0);

            var maskList = new List<Tensor> { langMask };
            foreach (var gm in geoMasks)
            {
                if (gm is not null) maskList.Add((Tensor)gm);
            }
            promptMask = torch.cat(maskList.ToArray(), dim: 1);
        }
        else
        {
            Tensor geoCombined = geoFeats[0];
            for (int i = 1; i < geoFeats.Count; i++)
            {
                geoCombined = cat(new[] { geoCombined, geoFeats[i] }, dim: 0);
            }
            prompt = geoCombined;
            promptMask = geoMasks.Count > 0 && geoMasks[0] is not null ? geoMasks[0] : null;
        }

        // 3. Run transformer encoder
        var memory = transformer_encoder.forward(
            imgFeats,
            promptMask is not null ? new List<Tensor> { promptMask } : null,
            imgPosEmbeds);

        // 4. Run transformer decoder
        var posEmbed = (Tensor)memory["pos_embed"];
        var memTensor = (Tensor)memory["memory"];
        var padMask = memory["padding_mask"] as Tensor;

        var decoderResult = run_decoder(
            posEmbed,
            memTensor,
            padMask,
            prompt,
            promptMask);

        var resultDict = decoderResult.Item1;
        var hs = decoderResult.Item2;

        // 5. Run segmentation head if available
        if (segmentation_head is not null)
        {
            var segResult = segmentation_head.forward(
                new List<Tensor> { imgFeats.Last() },
                hs,
                null,  // encoder_hidden_states
                prompt,
                promptMask);
            resultDict["pred_masks"] = segResult["pred_masks"];
            if (segResult.ContainsKey("pred_logits"))
                resultDict["pred_logits"] = segResult["pred_logits"];
        }

        return resultDict;
    }

    public Tuple<Tensor, Tensor?> encode_prompts(
        IList<Tensor> img_feats,
        IList<Tensor> img_pos_embeds,
        IList<long[]> vis_feat_sizes,
        Sam3Prompt geometric_prompt)
    {
        var (geo_feats, geo_masks) = geometry_encoder.forward(
            geometric_prompt, img_feats.ToList(), vis_feat_sizes.ToList());

        var prompt = cat(geo_feats.ToArray(), dim: 0);
        var validMasks = geo_masks.Where(m => m is not null).Cast<Tensor>().ToList();
        Tensor? prompt_mask = validMasks.Count > 0 ? cat(validMasks.ToArray(), dim: 1) : null;

        return Tuple.Create(prompt, prompt_mask);
    }

    public Dictionary<string, Tensor> run_encoder(
        IList<Tensor> img_feats,
        IList<Tensor> img_pos_embeds,
        Tensor prompt,
        Tensor? prompt_mask)
    {
        var vis_feat_sizes = img_feats.Select(f => new long[] { f.size(2), f.size(3) }).ToList();

        var memory = transformer_encoder.forward(
            img_feats.ToList(),
            prompt_mask is not null ? new List<Tensor> { prompt_mask } : null,
            img_pos_embeds.ToList());

        var result = new Dictionary<string, Tensor>();
        foreach (var kvp in memory)
        {
            if (kvp.Value is Tensor t)
                result[kvp.Key] = t;
        }
        return result;
    }

    public Tuple<Dictionary<string, Tensor>, Tensor> run_decoder(
        Tensor pos_embed,
        Tensor memory,
        Tensor? src_mask,
        Tensor prompt,
        Tensor? prompt_mask)
    {
        var query_embed = transformer_decoder.get_query_embed();
        var bs = memory.size(1);
        var num_queries = query_embed.size(0);
        var d_model = (int)query_embed.size(1);

        var tgt = query_embed.unsqueeze(1).repeat(1, bs, 1);

        var decoderResult = transformer_decoder.forward(
            tgt, memory, pos_embed, src_mask,
            prompt, prompt_mask);
        var hs = decoderResult.Item1;
        var reference_boxes = decoderResult.Item2;

        var result_dict = new Dictionary<string, Tensor>();
        var lastLayerIdx = (int)hs.size(0) - 1;

        result_dict.Add("pred_logits", predict_scores(hs[lastLayerIdx], prompt));

        var box_head = new Sam3MLP(d_model, 256, 4, 3);
        var reference_boxes_inv_sig = log(reference_boxes / (1 - reference_boxes + 0.000001f) + 0.000001f);
        var anchor_box_offsets = box_head.forward(hs[lastLayerIdx]);
        var outputs_coord = (reference_boxes_inv_sig + anchor_box_offsets).sigmoid();
        result_dict.Add("pred_boxes", outputs_coord);

        return Tuple.Create(result_dict, hs[lastLayerIdx]);
    }

    private Tensor predict_scores(Tensor hs, Tensor prompt)
    {
        var scores = matmul(hs, prompt.permute(new long[] { 1, 0, 2 }));
        return scores;
    }
}
