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
/// SAM3 Base model rebuilt to match the checkpoint architecture.
/// Implements text-conditioned image detection and instance segmentation.
///
/// Architecture:
///   - Vision Backbone: Standard ViT (32 layers, embed_dim=1024, patch_size=14, image_size=1008)
///   - FPN Neck: 3-level FPN (d_model=256)
///   - Text Encoder: BERT-style (24 layers, embed_dim=1024, hidden=4096, num_heads=16)
///   - DETR Encoder: 6 layers, d_model=256, MLP hidden=2048
///   - DETR Decoder: 6 layers, d_model=256, MLP hidden=2048, with presence token + box RPB
///   - Geometry Encoder: 3 transformer layers, d_model=256
    ///   - Mask Decoder: universal segmentation head producing per-query mask logits
///   - Scoring: Dot-product scoring with text_mlp (text-object similarity)
/// </summary>
public class Sam3BaseNew : Module
{
    public readonly Sam3ViTBackbone vision_backbone;
    public readonly Sam3ViTFpnNeck fpn_neck;
    public readonly Sam3TextEncoder text_encoder;
    public readonly Sam3TokenizerVE tokenizer;
    public readonly Sam3TransformerEncoder transformer_encoder;
    public readonly Sam3TransformerDecoderNew transformer_decoder;
    public readonly Sam3GeometryEncoderNew geometry_encoder;
    public readonly Sam3MaskDecoder mask_decoder;
    public readonly Sam3DotProductScoring dot_product_scoring;

    public readonly int hidden_dim;
    public readonly int num_feature_levels;
    public readonly bool use_dot_prod_scoring;

    private Device? _device_cache = null;

    public Sam3BaseNew(
        Sam3ViTBackbone visionBackbone,
        Sam3ViTFpnNeck fpnNeck,
        Sam3TextEncoder textEncoder,
        Sam3TokenizerVE tokenizer,
        Sam3TransformerEncoder transformerEncoder,
        Sam3TransformerDecoderNew transformerDecoder,
        Sam3GeometryEncoderNew geometryEncoder,
        Sam3MaskDecoder maskDecoder,
        Sam3DotProductScoring dotProductScoring,
        int numFeatureLevels = 3,
        bool useDotProdScoring = true)
        : base(nameof(Sam3BaseNew))
    {
        this.vision_backbone = visionBackbone;
        this.fpn_neck = fpnNeck;
        this.text_encoder = textEncoder;
        this.tokenizer = tokenizer;
        this.transformer_encoder = transformerEncoder;
        this.transformer_decoder = transformerDecoder;
        this.geometry_encoder = geometryEncoder;
        this.mask_decoder = maskDecoder;
        this.dot_product_scoring = dotProductScoring;
        this.hidden_dim = transformerDecoder.d_model;
        this.num_feature_levels = numFeatureLevels;
        this.use_dot_prod_scoring = useDotProdScoring;

        RegisterComponents();
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
    /// Forward pass through the vision backbone + FPN neck.
    /// Returns multi-scale features and positional embeddings.
    /// </summary>
    public Tuple<List<Tensor>, List<Tensor>> ForwardBackbone(Tensor images)
    {
        var (allFpnFeatures, allPosEmbeddings) = ForwardAllBackboneFeatures(images);
        return Tuple.Create(
            allFpnFeatures.Take(num_feature_levels).ToList(),
            allPosEmbeddings.Take(num_feature_levels).ToList());
    }

    private Tuple<List<Tensor>, List<Tensor>> ForwardAllBackboneFeatures(Tensor images)
    {
        // 1. ViT backbone: [B, 3, H, W] -> [B, num_tokens, embed_dim]
        var vit_features = vision_backbone.forward(images);
        var B = vit_features.size(0);
        var num_tokens = vit_features.size(1);
        var embed_dim = vit_features.size(2);
        var spatial_size = (int)Math.Sqrt(num_tokens);

        // Reshape to [B, embed_dim, H, W]
        var vit_2d = vit_features.permute(new long[] { 0, 2, 1 }).reshape(new long[] { B, embed_dim, spatial_size, spatial_size });
        // 2. FPN neck: produce 4-scale features
        var allFpnFeatures = fpn_neck.forward(vit_2d);

        // 3. Compute positional embeddings for each scale
        var allPosEmbeddings = new List<Tensor>();
        foreach (var feat in allFpnFeatures)
        {
            var pos = position_encoding_2d(feat);
            allPosEmbeddings.Add(pos);
        }

        return Tuple.Create(allFpnFeatures, allPosEmbeddings);
    }

    /// <summary>
    /// Full forward pass: encode image + text, run encoder/decoder pipeline.
    /// Returns pred_boxes [bs, nq, 4], pred_logits [bs, nq, 1], and mask_tokens [bs, nq, d_model].
    /// </summary>
    public Dictionary<string, Tensor> Forward(
        Tensor images,
        IList<string>? captions = null,
        Sam3Prompt? geometricPrompt = null)
    {
        // 1. Run vision backbone
        var (segmentationFeatures, allPosEmbeds) = ForwardAllBackboneFeatures(images);
        var imgFeats = segmentationFeatures.Take(num_feature_levels).ToList();
        var imgPosEmbeds = allPosEmbeds.Take(num_feature_levels).ToList();

        // 2. Get text features
        Tensor? langFeat = null;
        Tensor? langMask = null;

        if (captions is not null && captions.Count > 0 && text_encoder != null)
        {
            var (mask, memory, inputs) = text_encoder.ForwardText(captions);
            langFeat = memory; // [seq_len, batch, d_model]
            langMask = mask;    // [batch, seq_len]
        }

        // 3. Encode geometric prompts
        Sam3Prompt gp = geometricPrompt ?? new Sam3Prompt();
        var (geoFeatsTensor, geoMaskTensor) = geometry_encoder.forward(gp, imgFeats,
            imgFeats.Select(f => new long[] { f.size(2), f.size(3) }).ToList());

        // 4. Combine text + geometry prompts
        Tensor prompt;
        Tensor promptMask;

        if (langFeat is not null)
        {
            var promptList = new List<Tensor> { langFeat, geoFeatsTensor };
            prompt = cat(promptList.ToArray(), dim: 0);

            if (langFeat.size(1) != geoFeatsTensor.size(1))
                throw new ArgumentException("Caption and image batch sizes must match.", nameof(captions));

            // Both prompt sources use sequence-first features and batch-first masks.
            promptMask = torch.cat(new[] { langMask, geoMaskTensor }, dim: 1);
        }
        else
        {
            // Geometry encoder already returns [sequence, batch, channels].
            prompt = geoFeatsTensor;
            promptMask = geoMaskTensor;
        }

        // 5. Run transformer encoder
        var encoderMemory = transformer_encoder.forward(
            imgFeats,
            null, // padding masks
            imgPosEmbeds,
            prompt,
            promptMask);

        var memTensor = (Tensor)encoderMemory["memory"];
        var posEmbed = (Tensor)encoderMemory["pos_embed"];

        // 6. Run transformer decoder (pass imgFeats for spatial shapes)
        var decoderResult = run_decoder(
            imgFeats,
            posEmbed,
            memTensor,
            null,
            prompt,
            promptMask);

        var resultDict = decoderResult.Item1;
        var hs = decoderResult.Item2;

        if (imgFeats.Count > 0)
        {
            var (maskLogits, semanticLogits) = mask_decoder.forward(hs, segmentationFeatures, memTensor, prompt, promptMask);
            resultDict["pred_masks"] = maskLogits;
            resultDict["semantic_seg"] = semanticLogits;
        }

        return resultDict;
    }

    /// <summary>
    /// Run inference from pre-extracted features.
    /// </summary>
    public Dictionary<string, Tensor> RunInferenceFromFeatures(
        List<Tensor> imgFeats,
        List<Tensor> imgPosEmbeds,
        IList<string>? captions = null,
        Sam3Prompt? geometricPrompt = null)
    {
        var gp = geometricPrompt ?? new Sam3Prompt();

        var detectorFeatures = imgFeats.Take(num_feature_levels).ToList();
        var detectorPositions = imgPosEmbeds.Take(num_feature_levels).ToList();

        // 1. Encode geometric prompts
        var (geoFeatsTensor, geoMaskTensor) = geometry_encoder.forward(gp, detectorFeatures,
            detectorFeatures.Select(f => new long[] { f.size(2), f.size(3) }).ToList());

        // 2. Get text features
        Tensor? langFeat = null;
        Tensor? langMask = null;

        if (captions is not null && captions.Count > 0 && text_encoder != null)
        {
            var (mask, memory, inputs) = text_encoder.ForwardText(captions);
            langFeat = memory;
            langMask = mask;
        }

        // 3. Combine prompts
        Tensor prompt;
        Tensor promptMask;

        if (langFeat is not null)
        {
            var promptList = new List<Tensor> { langFeat, geoFeatsTensor };
            prompt = cat(promptList.ToArray(), dim: 0);
            if (langFeat.size(1) != geoFeatsTensor.size(1))
                throw new ArgumentException("Caption and image batch sizes must match.", nameof(captions));

            promptMask = torch.cat(new[] { langMask, geoMaskTensor }, dim: 1);
        }
        else
        {
            prompt = geoFeatsTensor;
            promptMask = geoMaskTensor;
        }

        // 4. Run transformer encoder
        var encoderMemory = transformer_encoder.forward(
            detectorFeatures, null, detectorPositions, prompt, promptMask);
        var memTensor = (Tensor)encoderMemory["memory"];
        var posEmbed = (Tensor)encoderMemory["pos_embed"];

        // 5. Run decoder (pass imgFeats for spatial shapes)
        var decoderResult = run_decoder(detectorFeatures, posEmbed, memTensor, null, prompt, promptMask);
        var resultDict = decoderResult.Item1;
        var hs = decoderResult.Item2;

        if (imgFeats.Count == 4)
        {
            var (maskLogits, semanticLogits) = mask_decoder.forward(hs, imgFeats, memTensor, prompt, promptMask);
            resultDict["pred_masks"] = maskLogits;
            resultDict["semantic_seg"] = semanticLogits;
        }

        return resultDict;
    }

    /// <summary>
    /// Run the encoder-decoder pipeline.
    /// Passes spatial_shapes from FPN features to enable box RPB in the decoder.
    /// </summary>
    public Tuple<Dictionary<string, Tensor>, Tensor> run_decoder(
        List<Tensor> src_feats,  // FPN features for spatial shapes
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

        var tgt = query_embed.unsqueeze(1).repeat(new long[] { 1, (int)bs, 1 });

        // Compute spatial shapes from FPN features for RPB [H, W]
        long[] spatialShapes = null;
        if (src_feats != null && src_feats.Count > 0)
        {
            var lastFeat = src_feats[src_feats.Count - 1];  // Highest res feature
            spatialShapes = new long[] { lastFeat.size(2), lastFeat.size(3) };  // [H, W]
        }

        var decoderResult = transformer_decoder.forward(
            tgt, memory, prompt, prompt_mask, pos_embed, spatialShapes);
        var hs = decoderResult.Item1;  // [num_layers, nq, bs, d_model]
        var reference_boxes = decoderResult.Item2;  // [num_layers, nq, bs, 4]
        var presenceLogits = decoderResult.Item3;

        var result_dict = new Dictionary<string, Tensor>();

        // predict_scores via dot product scoring
        // hs: [num_layers, nq, bs, d_model] -> need batch-first for DotProductScoring
        // Python DotProductScoring expects: hs [num_layers, bs, nq, d_model]
        var hs_batch_first = hs.permute(new long[] { 0, 2, 1, 3 });  // [num_layers, bs, nq, d_model]

        var scores = dot_product_scoring.forward(hs_batch_first, prompt, prompt_mask);  // [num_layers, bs, nq, 1]

        // Take last layer scores -> [bs, nq, 1]
        var scoresLast = scores.narrow(0, (long)scores.size(0) - 1, 1).squeeze(0);  // [bs, nq, 1]
        result_dict.Add("pred_logits", scoresLast);
        if (presenceLogits is not null)
        {
            result_dict["presence_logit_dec"] = presenceLogits.narrow(0, presenceLogits.size(0) - 1, 1).squeeze(0);
        }

        // reference_boxes: [num_layers, nq, bs, 4] -> take last layer -> [bs, nq, 4]
        var lastRef = reference_boxes.narrow(0, (long)reference_boxes.size(0) - 1, 1).squeeze(0);  // [nq, bs, 4]
        var outputs_coord = lastRef.permute(new long[] { 1, 0, 2 });  // [bs, nq, 4]
        result_dict.Add("pred_boxes", outputs_coord);

        return Tuple.Create(result_dict, hs_batch_first.narrow(0, (long)hs_batch_first.size(0) - 1, 1).squeeze(0));  // [bs, nq, d_model]
    }

    /// <summary>
    /// 2D positional encoding (sinusoidal).
    /// Sam3PositionEmbeddingSine divides num_pos_feats by 2 internally,
    /// so passing C produces output of C dimensions.
    /// </summary>
    private Tensor position_encoding_2d(Tensor x)
    {
        var C = x.size(1);
        var pe = new Sam3PositionEmbeddingSine((int)C, normalize: true);
        return pe.forward(x);
    }
}
