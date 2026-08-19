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
    private readonly List<LayerNorm> norms;

    public Sam3PixelDecoder(int d_model = 256)
        : base(nameof(Sam3PixelDecoder))
    {
        conv_layers = new List<Conv2d>();
        norms = new List<LayerNorm>();

        for (int i = 0; i < 3; i++)
        {
            conv_layers.Add(Conv2d(d_model, d_model, kernelSize: 3, padding: 1));
            norms.Add(LayerNorm(d_model));
        }

        for (int i = 0; i < conv_layers.Count; i++)
        {
            register_module("conv_" + i.ToString(), conv_layers[i]);
            register_module("norm_" + i.ToString(), norms[i]);
        }

        RegisterComponents();
    }

    public Tensor forward(Tensor x)
    {
        for (int i = 0; i < conv_layers.Count; i++)
        {
            x = conv_layers[i].forward(x);
            var B = x.size(0);
            var C = x.size(1);
            var H = x.size(2);
            var W = x.size(3);
            var x_perm = x.permute(new long[] { 0, 2, 3, 1 });
            x_perm = norms[i].forward(x_perm);
            x = x_perm.permute(new long[] { 0, 3, 1, 2 });
            x = functional.gelu(x);
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

    public Tensor forward(Tensor query, Tensor key_value)
    {
        // query: [nq, bs, d_model]
        // key_value: [bs, d_model, H, W]
        var B = query.size(1);
        var N = query.size(0);
        var H = key_value.size(2);
        var W = key_value.size(3);

        // Flatten spatial dims: [bs, d_model, H, W] -> [bs, d_model, H*W] -> [bs, d_model, H*W]
        var kv_flat = key_value.flatten(2);  // [bs, d_model, H*W]
        kv_flat = kv_flat.transpose(1, 2);   // [bs, H*W, d_model]

        // Transpose to seq-first: [H*W, bs, d_model]
        var kv_seq = kv_flat.transpose(0, 1);

        // Project query
        var q = q_proj.forward(query);  // [nq, bs, d_model]
        var k = k_proj.forward(kv_seq);  // [H*W, bs, d_model]
        var v = v_proj.forward(kv_seq);  // [H*W, bs, d_model]

        // Split heads: [N, B, d_model] -> [B, nhead, N, hd]
        var q_h = q.reshape(new long[] { N, B, nhead, d_model / nhead }).transpose(0, 1).transpose(1, 2);  // [bs, nhead, nq, hd]
        var k_h = k.reshape(new long[] { H * W, B, nhead, d_model / nhead }).transpose(0, 1).transpose(1, 2);  // [bs, nhead, HW, hd]
        var v_h = v.reshape(new long[] { H * W, B, nhead, d_model / nhead }).transpose(0, 1).transpose(1, 2);  // [bs, nhead, HW, hd]

        var scale = 1.0f / (float)Math.Sqrt(d_model / nhead);
        // k_h.T: [bs, nhead, hd, HW]
        var k_h_t = k_h.transpose(2, 3);
        var attn_weights = (q_h * scale).matmul(k_h_t);  // [bs, nhead, nq, HW]
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
/// Full Mask Decoder for SAM3 (detector-only version).
/// This checkpoint's mask_decoder produces mask tokens, not masks.
/// The actual segmentation head (MultiplexMaskDecoder) is not in this checkpoint.
///
/// Components:
/// - pixel_decoder: 3 conv layers processing highest-res FPN feature
/// - prompt_cross_attn: cross-attention between object queries and spatial features
/// - mask_embedder: MLP to produce mask tokens
/// - semantic_projection: projects mask tokens to semantic space
/// - instance_projection: projects mask tokens to instance space
/// </summary>
public class Sam3MaskDecoder : Module
{
    private readonly Sam3PixelDecoder pixel_decoder;
    private readonly Sam3MaskEmbedder mask_embedder;
    private readonly Sam3PromptCrossAttn prompt_cross_attn;
    private readonly LayerNorm prompt_cross_attn_norm;
    private readonly Linear semantic_projection;
    private readonly Linear instance_projection;
    private readonly int d_model;

    public Sam3MaskDecoder(int d_model = 256)
        : base(nameof(Sam3MaskDecoder))
    {
        this.d_model = d_model;

        pixel_decoder = new Sam3PixelDecoder(d_model);
        prompt_cross_attn = new Sam3PromptCrossAttn(d_model);
        prompt_cross_attn_norm = LayerNorm(d_model);
        mask_embedder = new Sam3MaskEmbedder(d_model);
        semantic_projection = Linear(d_model, d_model);
        instance_projection = Linear(d_model, d_model);

        RegisterComponents();
    }

    /// <summary>
    /// Forward pass producing mask tokens (not masks).
    /// obj_queries: [nq, bs, d_model] - object queries from decoder (last layer)
    /// img_feats: [bs, d_model, H, W] - highest resolution FPN feature
    /// Returns: mask_tokens [bs, nq, d_model], semantic_feats [bs, nq, d_model], instance_feats [bs, nq, d_model]
    /// </summary>
    public Tuple<Tensor, Tensor, Tensor> forward(
        Tensor obj_queries,
        Tensor img_feat)
    {
        // obj_queries: [nq, bs, d_model]
        var N = obj_queries.size(0);
        var B = obj_queries.size(1);
        var d_model = (int)obj_queries.size(2);

        // Pixel decoder processes the highest res feature
        var pixel_feat = pixel_decoder.forward(img_feat);  // [bs, d_model, H, W]

        // Prompt cross attention
        var cross_attn_out = prompt_cross_attn.forward(obj_queries, img_feat);
        cross_attn_out = prompt_cross_attn_norm.forward(cross_attn_out);  // [nq, bs, d_model]

        // Mask embedder processes the cross-attention output
        var cross_attn_perm = cross_attn_out.permute(new long[] { 1, 0, 2 });  // [bs, nq, d_model]
        var mask_tokens = mask_embedder.forward(cross_attn_perm);  // [bs, nq, d_model]

        // Semantic and instance projections
        var semantic_feats = semantic_projection.forward(mask_tokens);  // [bs, nq, d_model]
        var instance_feats = instance_projection.forward(mask_tokens);  // [bs, nq, d_model]

        return Tuple.Create(mask_tokens, semantic_feats, instance_feats);
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
