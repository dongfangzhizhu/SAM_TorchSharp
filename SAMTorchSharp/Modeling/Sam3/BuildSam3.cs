// Copyright (c) Sapiens AI. All rights reserved.

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System;
using System.Collections.Generic;

namespace SAMTorchSharp.Modeling.Sam3;

/// <summary>
/// Builder for constructing Sam3Base models.
/// Ported from sam3/model/model_builder.py patterns
/// </summary>
public class BuildSam3
{
    private int embed_dim = 256;
    private int depth = 12;
    private int num_heads = 8;
    private int mlp_ratio = 4;
    private int num_feature_levels = 4;
    private int num_queries = 900;
    private int decoder_layers = 6;
    private int multiplex_count = 1;
    private int num_multimask_outputs = 3;
    private bool add_sam2_neck = false;
    private bool use_rope = true;

    public BuildSam3 SetEmbedDim(int value) { embed_dim = value; return this; }
    public BuildSam3 SetDepth(int value) { depth = value; return this; }
    public BuildSam3 SetNumHeads(int value) { num_heads = value; return this; }
    public BuildSam3 SetMlpRatio(int value) { mlp_ratio = value; return this; }
    public BuildSam3 SetNumFeatureLevels(int value) { num_feature_levels = value; return this; }
    public BuildSam3 SetNumQueries(int value) { num_queries = value; return this; }
    public BuildSam3 SetDecoderLayers(int value) { decoder_layers = value; return this; }
    public BuildSam3 SetMultiplexCount(int value) { multiplex_count = value; return this; }
    public BuildSam3 SetNumMultimaskOutputs(int value) { num_multimask_outputs = value; return this; }
    public BuildSam3 WithSam2Neck(bool value = true) { add_sam2_neck = value; return this; }

    /// <summary>
    /// Build a Sam3Base model with default ViT-H configuration.
    /// </summary>
    public Sam3Base Build()
    {
        // 1. Build ViT-Det backbone
        var vit_backbone = new Sam3ViTDetBackbone(
            patch_size: 14,
            embed_dim: 1024,
            depth: 48,
            num_heads: 16,
            mlp_ratio: mlp_ratio * 2,
            use_rope: use_rope);

        // 2. Build position encoding
        var position_encoding = new Sam3PositionEmbeddingSine(embed_dim / 2, normalize: true);

        // 3. Build FPN neck
        var neck = new Sam3DualViTDetNeck(
            vit_backbone,
            position_encoding,
            embed_dim,
            scale_factors: new float[] { 4.0f, 2.0f, 1.0f, 0.5f },
            add_sam2_neck: add_sam2_neck);

        // 4. Build geometry encoder
        var geometry_encoder = new Sam3GeometryEncoder(embed_dim, num_input_point_coords: 2, num_feature_levels);

        // 5. Build transformer encoder
        var transformer_encoder = new Sam3TransformerEncoder(
            d_model: embed_dim,
            nhead: num_heads,
            num_layers: 1,
            dim_feedforward: embed_dim * 4,
            num_feature_levels: num_feature_levels);

        // 6. Build transformer decoder
        var transformer_decoder = new Sam3TransformerDecoder(
            d_model: embed_dim,
            nhead: num_heads,
            num_layers: decoder_layers,
            num_queries: num_queries,
            dim_feedforward: embed_dim * 4);

        // 7. Build segmentation head
        var segmentation_head = new Sam3SegmentationHead(
            transformer_dim: embed_dim,
            transformer: transformer_decoder,
            num_queries: num_queries,
            multiplex_count: multiplex_count,
            num_multimask_outputs: num_multimask_outputs);

        // 8. Assemble
        var sam3 = new Sam3Base(
            backbone: neck,
            geometry_encoder: geometry_encoder,
            transformer_encoder: transformer_encoder,
            transformer_decoder: transformer_decoder,
            segmentation_head: segmentation_head,
            num_feature_levels: num_feature_levels,
            multimask_output: true);

        return sam3;
    }
}
