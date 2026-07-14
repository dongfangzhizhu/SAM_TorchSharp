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
/// Ported from sam3/model/model_builder.py patterns.
/// </summary>
public class BuildSam3
{
    private int embedDim = 256;
    private int depth = 12;
    private int numHeads = 8;
    private int mlpRatio = 4;
    private int numFeatureLevels = 4;
    private int numQueries = 900;
    private int decoderLayers = 6;
    private int multiplexCount = 1;
    private int numMultimaskOutputs = 3;
    private bool addSam2Neck = false;
    private bool useRope = true;
    private bool withTextEncoder = true;

    // Text encoder parameters (CLIP VE)
    private int textWidth = 1024;
    private int textHeads = 16;
    private int textLayers = 24;
    private int contextLength = 32;
    private int vocabSize = 49408;

    // Path to tokenizer BPE merges file
    private string? bpeFilePath;

    public BuildSam3 SetEmbedDim(int value) { embedDim = value; return this; }
    public BuildSam3 SetDepth(int value) { depth = value; return this; }
    public BuildSam3 SetNumHeads(int value) { numHeads = value; return this; }
    public BuildSam3 SetMlpRatio(int value) { mlpRatio = value; return this; }
    public BuildSam3 SetNumFeatureLevels(int value) { numFeatureLevels = value; return this; }
    public BuildSam3 SetNumQueries(int value) { numQueries = value; return this; }
    public BuildSam3 SetDecoderLayers(int value) { decoderLayers = value; return this; }
    public BuildSam3 SetMultiplexCount(int value) { multiplexCount = value; return this; }
    public BuildSam3 SetNumMultimaskOutputs(int value) { numMultimaskOutputs = value; return this; }
    public BuildSam3 WithSam2Neck(bool value = true) { addSam2Neck = value; return this; }
    public BuildSam3 WithTextEncoder(bool value = true) { withTextEncoder = value; return this; }
    public BuildSam3 SetBpeFilePath(string path) { bpeFilePath = path; return this; }

    /// <summary>
    /// Build a Sam3Base model with default ViT-H configuration.
    /// </summary>
    public Sam3Base Build()
    {
        // 1. Build ViT-Det backbone
        var vitBackbone = new Sam3ViTDetBackbone(
            patch_size: 14,
            embed_dim: 1024,
            depth: 48,
            num_heads: 16,
            mlp_ratio: (float)(mlpRatio * 2),
            use_rope: useRope);

        // 2. Build position encoding
        var positionEncoding = new Sam3PositionEmbeddingSine(embedDim / 2, normalize: true);

        // 3. Build FPN neck
        var neck = new Sam3DualViTDetNeck(
            vitBackbone,
            positionEncoding,
            embedDim,
            scale_factors: new float[] { 4.0f, 2.0f, 1.0f, 0.5f },
            add_sam2_neck: addSam2Neck);

        // 4. Build text encoder (if enabled)
        Sam3VETextEncoder? textEncoder = null;
        Sam3TokenizerVE? tokenizer = null;
        Sam3VLBackbone? vlBackbone = null;

        if (withTextEncoder)
        {
            tokenizer = bpeFilePath is not null
                ? new Sam3TokenizerVE(bpeFilePath, contextLength)
                : new Sam3TokenizerVE(contextLength: contextLength);

            textEncoder = new Sam3VETextEncoder(
                dModel: embedDim,
                textWidth: textWidth,
                textHeads: textHeads,
                textLayers: textLayers,
                contextLength: contextLength,
                vocabSize: vocabSize,
                useLnPost: true,
                tokenizer: tokenizer);

            vlBackbone = new Sam3VLBackbone(neck, textEncoder, scalp: 0);
        }

        // 5. Build geometry encoder
        var geometryEncoder = new Sam3GeometryEncoder(
            embed_dim: embedDim,
            num_input_point_coords: 2,
            num_levels: numFeatureLevels);

        // 6. Build transformer encoder
        var transformerEncoder = new Sam3TransformerEncoder(
            d_model: embedDim,
            nhead: numHeads,
            num_layers: 1,
            dim_feedforward: embedDim * 4,
            num_feature_levels: numFeatureLevels);

        // 7. Build transformer decoder
        var transformerDecoder = new Sam3TransformerDecoder(
            d_model: embedDim,
            nhead: numHeads,
            num_layers: decoderLayers,
            num_queries: numQueries,
            dim_feedforward: embedDim * 4);

        // 8. Build segmentation head
        var segmentationHead = new Sam3SegmentationHead(
            transformer_dim: embedDim,
            transformer: transformerDecoder,
            num_queries: numQueries,
            multiplex_count: multiplexCount,
            num_multimask_outputs: numMultimaskOutputs);

        // 9. Assemble
        if (vlBackbone is not null)
        {
            return new Sam3Base(
                vlBackbone: vlBackbone,
                geometryEncoder: geometryEncoder,
                transformerEncoder: transformerEncoder,
                transformerDecoder: transformerDecoder,
                segmentationHead: segmentationHead,
                numFeatureLevels: numFeatureLevels,
                multimaskOutput: true,
                textEncoder: textEncoder,
                tokenizer: tokenizer);
        }
        else
        {
            return new Sam3Base(
                neck: neck,
                geometryEncoder: geometryEncoder,
                transformerEncoder: transformerEncoder,
                transformerDecoder: transformerDecoder,
                segmentationHead: segmentationHead,
                numFeatureLevels: numFeatureLevels,
                multimaskOutput: true);
        }
    }
}
