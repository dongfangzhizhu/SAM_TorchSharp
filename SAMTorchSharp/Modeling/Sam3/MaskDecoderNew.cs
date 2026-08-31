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
/// Pixel decoder for SAM3 mask decoder.
/// Matches: detector_model.mask_decoder.pixel_decoder
/// Architecture: 3 conv layers (256->256) with 3x3 kernels + layer norm
/// </summary>
public class Sam3PixelDecoder : Module
{
    private readonly List<Conv2d> conv_layers;
    private readonly List<GroupNorm> norms;

    public Sam3PixelDecoder(int d_model = 256)
        : base(nameof(Sam3PixelDecoder))
    {
        conv_layers = new List<Conv2d>();
        norms = new List<GroupNorm>();

        for (int i = 0; i < 3; i++)
        {
            conv_layers.Add(Conv2d(d_model, d_model, kernelSize: 3, padding: 1));
            norms.Add(GroupNorm(8, d_model));
        }

        for (int i = 0; i < conv_layers.Count; i++)
        {
            register_module("conv_" + i.ToString(), conv_layers[i]);
            register_module("norm_" + i.ToString(), norms[i]);
        }

        RegisterComponents();
    }

    public Tensor forward(IReadOnlyList<Tensor> features)
    {
        if (features.Count != conv_layers.Count + 1)
            throw new ArgumentException($"SAM 3 pixel decoder requires {conv_layers.Count + 1} FPN features.", nameof(features));

        var x = features[^1];
        for (int i = 0; i < conv_layers.Count; i++)
        {
            var lateral = features[features.Count - i - 2];
            x = functional.interpolate(x, size: new long[] { lateral.size(2), lateral.size(3) }, mode: InterpolationMode.Nearest);
            x = x + lateral;
            x = conv_layers[i].forward(x);
            x = functional.relu(norms[i].forward(x));
        }
        return x;
    }
}

/// <summary>
/// Mask embedder for SAM3.
/// Matches: detector_model.mask_decoder.mask_embedder
/// Architecture: 3 FC layers (256->256)
/// </summary>
public class Sam3MaskEmbedder : Module
{
    private readonly List<Linear> layers;

    public Sam3MaskEmbedder(int d_model = 256)
        : base(nameof(Sam3MaskEmbedder))
    {
        layers = new List<Linear>();
        for (int i = 0; i < 3; i++)
        {
            layers.Add(Linear(d_model, d_model));
        }

        for (int i = 0; i < layers.Count; i++)
        {
            register_module("layer_" + i.ToString(), layers[i]);
        }

        RegisterComponents();
    }

    public Tensor forward(Tensor x)
    {
        for (int i = 0; i < layers.Count; i++)
        {
            x = layers[i].forward(x);
            if (i < layers.Count - 1)
            {
                x = functional.gelu(x);
            }
        }
        return x;
    }
}

/// <summary>
/// Prompt cross attention for mask decoder.
/// Matches: detector_model.mask_decoder.prompt_cross_attn
/// </summary>
public class Sam3PromptCrossAttn : Module
{
    private readonly Linear q_proj;
    private readonly Linear k_proj;
    private readonly Linear v_proj;
    private readonly Linear o_proj;
    private readonly int d_model;
    private readonly int nhead;

    public Sam3PromptCrossAttn(int d_model = 256, int nhead = 8)
        : base(nameof(Sam3PromptCrossAttn))
    {
        this.d_model = d_model;
        this.nhead = nhead;

        q_proj = Linear(d_model, d_model);
        k_proj = Linear(d_model, d_model);
        v_proj = Linear(d_model, d_model);
        o_proj = Linear(d_model, d_model);

        register_module("q_proj", q_proj);
        register_module("k_proj", k_proj);
        register_module("v_proj", v_proj);
        register_module("o_proj", o_proj);

        RegisterComponents();
    }

    public Tensor forward(Tensor query, Tensor keyValue, Tensor? keyPaddingMask = null)
    {
        // query: [target length, batch, d_model]
        // keyValue: [source length, batch, d_model]
        var B = query.size(1);
        var N = query.size(0);
        var S = keyValue.size(0);

        // Project query
        var q = q_proj.forward(query);  // [nq, bs, d_model]
        var k = k_proj.forward(keyValue);
        var v = v_proj.forward(keyValue);

        // Split heads: [N, B, d_model] -> [B, nhead, N, hd]
        var q_h = q.reshape(new long[] { N, B, nhead, d_model / nhead }).transpose(0, 1).transpose(1, 2);  // [bs, nhead, nq, hd]
        var k_h = k.reshape(new long[] { S, B, nhead, d_model / nhead }).transpose(0, 1).transpose(1, 2);
        var v_h = v.reshape(new long[] { S, B, nhead, d_model / nhead }).transpose(0, 1).transpose(1, 2);

        var scale = 1.0f / (float)Math.Sqrt(d_model / nhead);
        // k_h.T: [bs, nhead, hd, HW]
        var k_h_t = k_h.transpose(2, 3);
        var attn_weights = (q_h * scale).matmul(k_h_t);  // [bs, nhead, nq, HW]
        if (keyPaddingMask is not null)
        {
            var mask = keyPaddingMask.ndim == 3 ? keyPaddingMask.squeeze(0) : keyPaddingMask;
            attn_weights = attn_weights.masked_fill(mask.logical_not().unsqueeze(1).unsqueeze(1), float.NegativeInfinity);
        }
        var attn_probs = functional.softmax(attn_weights, dim: 3);
        var attn_out = attn_probs.matmul(v_h);  // [bs, nhead, nq, hd]

        // Merge heads and transpose back: [bs, nhead, nq, hd] -> [nq, bs, d_model]
        attn_out = attn_out.transpose(1, 2).transpose(0, 1).reshape(new long[] { N, B, d_model });  // [nq, bs, d_model]

        // Project output
        attn_out = o_proj.forward(attn_out);  // [nq, bs, d_model]

        return attn_out;
    }
}

/// <summary>
/// SAM 3 universal segmentation head used by the image detector.
///
/// Components:
/// - pixel_decoder: top-down fusion of four FPN levels
/// - prompt_cross_attn: cross-attention from visual encoder tokens to the prompt
/// - mask_embedder: MLP projecting object queries
/// - semantic_projection/instance_projection: 1x1 convolutions over fused pixels
/// </summary>
public class Sam3MaskDecoder : Module
{
    private readonly Sam3PixelDecoder pixel_decoder;
    private readonly Sam3MaskEmbedder mask_embedder;
    private readonly Sam3PromptCrossAttn prompt_cross_attn;
    private readonly LayerNorm prompt_cross_attn_norm;
    private readonly Conv2d semantic_projection;
    private readonly Conv2d instance_projection;
    private readonly int d_model;

    public Sam3MaskDecoder(int d_model = 256)
        : base(nameof(Sam3MaskDecoder))
    {
        this.d_model = d_model;

        pixel_decoder = new Sam3PixelDecoder(d_model);
        prompt_cross_attn = new Sam3PromptCrossAttn(d_model);
        prompt_cross_attn_norm = LayerNorm(d_model);
        mask_embedder = new Sam3MaskEmbedder(d_model);
        semantic_projection = Conv2d(d_model, 1, kernelSize: 1);
        instance_projection = Conv2d(d_model, d_model, kernelSize: 1);

        RegisterComponents();
    }

    /// <summary>
    /// Forward pass producing low-resolution mask logits.
    /// </summary>
    public Tuple<Tensor, Tensor> forward(
        Tensor objectQueries,
        IReadOnlyList<Tensor> imageFeatures,
        Tensor encoderHiddenStates,
        Tensor prompt,
        Tensor? promptMask)
    {
        if (imageFeatures.Count != 4)
            throw new ArgumentException("SAM 3 segmentation requires four FPN feature levels.", nameof(imageFeatures));

        var normalizedVisualTokens = prompt_cross_attn_norm.forward(encoderHiddenStates);
        var attendedVisualTokens = prompt_cross_attn.forward(normalizedVisualTokens, prompt, promptMask);
        var visualTokens = attendedVisualTokens + encoderHiddenStates;

        var lastFeature = imageFeatures[^1];
        var spatialSize = lastFeature.size(2) * lastFeature.size(3);
        var encodedLastFeature = visualTokens.narrow(0, 0, spatialSize)
            .permute(1, 2, 0)
            .reshape(lastFeature.shape);
        var decoderFeatures = imageFeatures.ToArray();
        decoderFeatures[^1] = encodedLastFeature;

        var pixelEmbedding = pixel_decoder.forward(decoderFeatures);
        var instanceEmbedding = instance_projection.forward(pixelEmbedding);
        var maskEmbedding = mask_embedder.forward(objectQueries);
        var maskLogits = einsum("bqc,bchw->bqhw", maskEmbedding, instanceEmbedding);
        var semanticLogits = semantic_projection.forward(pixelEmbedding);
        return Tuple.Create(maskLogits, semanticLogits);
    }
}

/// <summary>
/// Dot-product scoring head for SAM3.
/// Matches: detector_model.dot_product_scoring
/// Architecture: mean_pool(prompt via text_mlp + text_proj) + hs_proj -> single scalar score per query
/// Python reference: outputs_class = dot_prod_scoring_head(hs, prompt, prompt_mask)
/// Returns shape: [num_layers, bs, nq, 1]
///
/// Key components from checkpoint:
/// - text_mlp: 2-layer MLP (256 -> 2048 -> 256) applied to prompt before pooling
/// - text_mlp_out_norm: LayerNorm after text_mlp
/// - text_proj: Linear (256 -> 256) for projected prompt
/// - query_proj: Linear (256 -> 256) for projected hs (used as "query_proj" in keys)
/// </summary>
public class Sam3DotProductScoring : Module
{
    private readonly Linear text_proj;          // Linear(d_model, d_proj) - projects pooled prompt
    private readonly Linear query_proj;         // Linear(d_model, d_proj) - projects hs (named "query_proj" in checkpoint)
    private readonly ModuleList<Linear> text_mlp;  // Optional MLP on prompt before pooling
    private readonly LayerNorm text_mlp_out_norm;
    private readonly bool clamp_logits;
    private readonly float clamp_max_val;
    private readonly int d_proj;

    public Sam3DotProductScoring(
        int d_model = 256,
        int? d_proj = null,
        bool has_text_mlp = true,
        bool clamp_logits = true,
        float clamp_max_val = 12.0f)
        : base(nameof(Sam3DotProductScoring))
    {
        this.d_proj = d_proj ?? d_model;
        this.clamp_logits = clamp_logits;
        this.clamp_max_val = clamp_max_val;

        // text_proj: Linear(d_model, d_proj)
        text_proj = Linear(d_model, this.d_proj);
        register_module("text_proj", text_proj);

        // query_proj: Linear(d_model, d_proj) - named "query_proj" in checkpoint
        query_proj = Linear(d_model, this.d_proj);
        register_module("query_proj", query_proj);

        // Optional text_mlp: 2-layer MLP (d_model -> 2048 -> d_model)
        if (has_text_mlp)
        {
            text_mlp = new ModuleList<Linear>(new[] {
                Linear(d_model, 2048),   // layer1: 256 -> 2048
                Linear(2048, d_model)    // layer2: 2048 -> 256
            });
            register_module("text_mlp", text_mlp);

            text_mlp_out_norm = LayerNorm(d_model);
            register_module("text_mlp_out_norm", text_mlp_out_norm);
        }
        else
        {
            text_mlp = null;
            text_mlp_out_norm = null;
        }

        RegisterComponents();
    }

    /// <summary>
    /// Forward pass.
    /// hs: [num_layers, bs, nq, d_model] - decoder hidden states (batch-first)
    /// prompt: [seq_len, bs, d_model] - text features from BERT
    /// prompt_mask: [bs, seq_len] - boolean mask (1 = valid, 0 = padding)
    /// Returns: scores [num_layers, bs, nq, 1] - single scalar per query
    /// </summary>
    public Tensor forward(Tensor hs, Tensor prompt, Tensor prompt_mask)
    {
        // Mean pool text over valid tokens
        // prompt_mask may be [bs, seq_len] or [1, bs, seq_len]; normalize to [bs, seq_len]
        if (prompt_mask.ndim == 3)
        {
            prompt_mask = prompt_mask.squeeze(0);  // Remove extra batch dim if present
        }

        // is_valid: [seq_len, bs, 1]; 1 for valid, 0 for padding
        // Note: prompt_mask is 1=valid, 0=padding (opposite of typical attention mask)
        var is_valid = prompt_mask.permute(new long[] { 1, 0 }).unsqueeze(-1).to(ScalarType.Float32);  // [seq_len, bs, 1]

        // num_valid: [bs, 1]; clamp at min 1.0 to avoid division by zero
        var num_valid = clamp(is_valid.sum(dim: 0), 1.0f);  // [bs, 1]

        // Apply text_mlp to prompt before pooling (if present)
        Tensor pooled_prompt_input = prompt;
        if (text_mlp is not null)
        {
            for (int l = 0; l < text_mlp.Count; l++)
            {
                pooled_prompt_input = text_mlp[l].forward(pooled_prompt_input);
                if (l < text_mlp.Count - 1)
                    pooled_prompt_input = functional.relu(pooled_prompt_input);
            }
            pooled_prompt_input = text_mlp_out_norm.forward(pooled_prompt_input);
        }

        // pooled_prompt: [bs, d_model]
        var pooled_prompt = (pooled_prompt_input * is_valid).sum(dim: 0) / num_valid.squeeze(-1);  // [bs, d_model]

        // Project both to d_proj dimensions
        var proj_pooled_prompt = text_proj.forward(pooled_prompt);  // [bs, d_proj]
        var proj_hs = query_proj.forward(hs);  // [num_layers, bs, nq, d_proj]

        // Scale factor
        var scale = 1.0f / (float)Math.Sqrt(d_proj);

        // Dot product: [num_layers, bs, nq, d_proj] x [bs, d_proj, 1] -> [num_layers, bs, nq, 1]
        var scores = matmul(proj_hs, proj_pooled_prompt.unsqueeze(-1)) * scale;

        // Clamp to avoid numerical issues
        if (clamp_logits)
        {
            scores = clamp(scores, -clamp_max_val, clamp_max_val);
        }

        return scores;
    }
}
