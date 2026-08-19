// Copyright (c) Sapiens AI. All rights reserved.

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System;
using System.Collections.Generic;

namespace SAMTorchSharp.Modeling.Sam3;

/// <summary>
/// Builder for constructing Sam3BaseNew models matching the checkpoint architecture.
/// Checkpoint architecture (from config.json):
///   - Vision Backbone: ViT (32 layers, embed_dim=1024, num_heads=16, patch_size=14)
///                     image_size=1008, so output is 72x72 tokens
///   - FPN Neck: 3-level FPN (backbone_feature_sizes: [288, 144, 72])
///   - Text Encoder: BERT-style (24 layers, embed_dim=1024, hidden=4096, num_heads=16)
///   - DETR Encoder: 6 layers, d_model=256, MLP hidden=2048
///   - DETR Decoder: 6 layers, d_model=256, MLP hidden=2048
///   - Geometry Encoder: 3 transformer layers, d_model=256
///   - Mask Decoder: Pixel decoder + mask embedder + prompt cross attn + output_upscaling + hypernetwork
///   - Scoring: Dot-product scoring
///   - low_res_mask_size: 288
///   - num_feature_levels: 3
/// </summary>
public class BuildSam3New
{
    private int d_model = 256;
    private int visionEmbedDim = 1024;
    private int visionDepth = 32;
    private int visionNumHeads = 16;
    private int visionPatchSize = 14;
    private int textWidth = 1024;
    private int textLayers = 24;
    private int textNumHeads = 16;
    private int contextLength = 32;
    private int vocabSize = 49408;
    private int numFeatureLevels = 3;
    private int numQueries = 200;
    private int decoderLayers = 6;
    private int encoderLayers = 6;
    private int geometryLayers = 3;
    private int decoderDimFeedforward = 2048;
    private int encoderDimFeedforward = 2048;
    private int geometryDimFeedforward = 2048;
    private bool withTextEncoder = true;
    private string bpeFilePath;

    public BuildSam3New SetDModel(int value) { d_model = value; return this; }
    public BuildSam3New SetVisionDepth(int value) { visionDepth = value; return this; }
    public BuildSam3New SetVisionEmbedDim(int value) { visionEmbedDim = value; return this; }
    public BuildSam3New SetVisionNumHeads(int value) { visionNumHeads = value; return this; }
    public BuildSam3New SetVisionPatchSize(int value) { visionPatchSize = value; return this; }
    public BuildSam3New SetTextWidth(int value) { textWidth = value; return this; }
    public BuildSam3New SetTextLayers(int value) { textLayers = value; return this; }
    public BuildSam3New SetTextNumHeads(int value) { textNumHeads = value; return this; }
    public BuildSam3New SetContextLength(int value) { contextLength = value; return this; }
    public BuildSam3New SetVocabSize(int value) { vocabSize = value; return this; }
    public BuildSam3New SetNumFeatureLevels(int value) { numFeatureLevels = value; return this; }
    public BuildSam3New SetNumQueries(int value) { numQueries = value; return this; }
    public BuildSam3New SetDecoderLayers(int value) { decoderLayers = value; return this; }
    public BuildSam3New SetEncoderLayers(int value) { encoderLayers = value; return this; }
    public BuildSam3New SetGeometryLayers(int value) { geometryLayers = value; return this; }
    public BuildSam3New SetDecoderDimFeedforward(int value) { decoderDimFeedforward = value; return this; }
    public BuildSam3New SetEncoderDimFeedforward(int value) { encoderDimFeedforward = value; return this; }
    public BuildSam3New SetGeometryDimFeedforward(int value) { geometryDimFeedforward = value; return this; }
    public BuildSam3New WithTextEncoder(bool value = true) { withTextEncoder = value; return this; }
    public BuildSam3New SetBpeFilePath(string path) { bpeFilePath = path; return this; }

    public Sam3BaseNew Build()
    {
        // 1. Vision Backbone: Standard ViT
        // image_size=1008, patch_size=14 => 72x72 tokens
        var visionBackbone = new Sam3ViTBackbone(
            img_size: 1008,
            patch_size: visionPatchSize,
            in_chans: 3,
            embed_dim: visionEmbedDim,
            depth: visionDepth,
            num_heads: visionNumHeads);

        // 2. FPN Neck: 4 levels (config says num_feature_levels=3 for DETR, but FPN has 4 levels)
        // Level 3 (highest res) is used by mask decoder, levels 0-2 by DETR encoder
        var fpnNeck = new Sam3ViTFpnNeck(
            inChannels: visionEmbedDim,
            d_model: d_model,
            num_levels: 4);

        // 3. Text Encoder (BERT-style)
        Sam3TextEncoder textEncoder = null;
        Sam3TokenizerVE tokenizer = null;

        if (withTextEncoder)
        {
            tokenizer = !string.IsNullOrEmpty(bpeFilePath)
                ? new Sam3TokenizerVE(bpeFilePath, contextLength)
                : new Sam3TokenizerVE(contextLength: contextLength);

            textEncoder = new Sam3TextEncoder(
                d_model: d_model,
                text_width: textWidth,
                num_heads: textNumHeads,
                num_layers: textLayers,
                context_length: contextLength,
                vocab_size: vocabSize,
                tokenizer: tokenizer);
        }

        // 4. Transformer Encoder (DETR-style)
        var transformerEncoder = new Sam3TransformerEncoder(
            d_model: d_model,
            nhead: 8,
            num_layers: encoderLayers,
            dim_feedforward: encoderDimFeedforward,
            num_feature_levels: numFeatureLevels);

        // 5. Transformer Decoder (DETR-style)
        var transformerDecoder = new Sam3TransformerDecoderNew(
            d_model: d_model,
            nhead: 8,
            num_layers: decoderLayers,
            num_queries: numQueries,
            dim_feedforward: decoderDimFeedforward);

        // 6. Geometry Encoder (Transformer-based)
        var geometryEncoder = new Sam3GeometryEncoderNew(
            d_model: d_model,
            num_geo_layers: geometryLayers);

        // 7. Mask Decoder
        var maskDecoder = new Sam3MaskDecoder(d_model: d_model);

        // 8. Dot-product Scoring
        var dotProductScoring = new Sam3DotProductScoring(d_model: d_model);

        // 9. Assemble
        return new Sam3BaseNew(
            visionBackbone: visionBackbone,
            fpnNeck: fpnNeck,
            textEncoder: textEncoder,
            tokenizer: tokenizer,
            transformerEncoder: transformerEncoder,
            transformerDecoder: transformerDecoder,
            geometryEncoder: geometryEncoder,
            maskDecoder: maskDecoder,
            dotProductScoring: dotProductScoring,
            numFeatureLevels: numFeatureLevels,
            useDotProdScoring: true);
    }
}
