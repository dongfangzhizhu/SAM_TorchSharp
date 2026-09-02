// Copyright (c) Sapiens AI. All rights reserved.

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System;
using System.Collections.Generic;

namespace SAMTorchSharp.Modeling.Sam3;

/// <summary>
/// MLP helper for transformer layers.
/// </summary>
public class Sam3MLP : Module<Tensor, Tensor>
{
    private readonly Linear linear1;
    private readonly Linear linear2;
    private readonly Module<Tensor, Tensor> activation;
    private readonly Dropout dropout;

    public Sam3MLP(int inFeatures, int hiddenFeatures, int outFeatures, int numLayers = 2, bool sigmoid_output = true, string activationType = "gelu", float dropoutRate = 0.1f)
        : base(nameof(Sam3MLP))
    {
        linear1 = Linear(inFeatures, hiddenFeatures);
        linear2 = Linear(hiddenFeatures, outFeatures);
        activation = activationType.ToLower() == "relu" ? ReLU() : GELU();
        dropout = Dropout(dropoutRate);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        x = linear2.forward(dropout.forward(activation.forward(linear1.forward(x))));
        return x;
    }
}

/// <summary>
/// Transformer encoder layer for SAM3 with SEPARATE Q/K/V projections.
/// Matches: detector_model.detr_encoder.layers.{N}
/// Architecture: d_model=256, nhead=8, MLP hidden=2048
///
/// Checkpoint keys per layer:
///   self_attn.q_proj/k_proj/v_proj/o_proj (8 params)
///   cross_attn.q_proj/k_proj/v_proj/o_proj (8 params)
///   layer_norm1/2/3 (6 params)
///   mlp.fc1/fc2 (4 params)
/// </summary>
public class Sam3TransformerEncoderLayer : Module
{
    private readonly Linear self_attn_q_proj;
    private readonly Linear self_attn_k_proj;
    private readonly Linear self_attn_v_proj;
    private readonly Linear self_attn_o_proj;

    private readonly Linear cross_attn_q_proj;
    private readonly Linear cross_attn_k_proj;
    private readonly Linear cross_attn_v_proj;
    private readonly Linear cross_attn_o_proj;

    private readonly Linear linear1;
    private readonly Linear linear2;
    private readonly Module<Tensor, Tensor> activation;
    private readonly float dropout;

    private readonly LayerNorm norm1;
    private readonly LayerNorm norm2;
    private readonly LayerNorm norm3;

    private readonly int d_model;
    private readonly int nhead;
    private readonly int dim_feedforward;
    private readonly int head_dim;
    private readonly float attn_scale;

    public Sam3TransformerEncoderLayer(
        int d_model = 256,
        int nhead = 8,
        int dim_feedforward = 2048,
        float dropout = 0.1f,
        string activation_type = "gelu")
        : base(nameof(Sam3TransformerEncoderLayer))
    {
        this.d_model = d_model;
        this.nhead = nhead;
        this.dim_feedforward = dim_feedforward;
        this.head_dim = d_model / nhead;
        this.attn_scale = 1.0f / (float)Math.Sqrt(head_dim);
        this.dropout = dropout;

        self_attn_q_proj = Linear(d_model, d_model);
        self_attn_k_proj = Linear(d_model, d_model);
        self_attn_v_proj = Linear(d_model, d_model);
        self_attn_o_proj = Linear(d_model, d_model);

        cross_attn_q_proj = Linear(d_model, d_model);
        cross_attn_k_proj = Linear(d_model, d_model);
        cross_attn_v_proj = Linear(d_model, d_model);
        cross_attn_o_proj = Linear(d_model, d_model);

        linear1 = Linear(d_model, dim_feedforward);
        linear2 = Linear(dim_feedforward, d_model);
        activation = activation_type.ToLower() == "relu" ? ReLU() : GELU();

        norm1 = LayerNorm(d_model);
        norm2 = LayerNorm(d_model);
        norm3 = LayerNorm(d_model);

        RegisterComponents();
    }

    private Tensor dot_product_attention(Tensor q, Tensor k, Tensor v, Tensor? key_padding_mask = null)
    {
        // q, k, v: [seq, batch, d_model]
        var querySeq = q.size(0);
        var memorySeq = k.size(0);
        var B = q.size(1);

        if (k.size(1) != B || v.size(1) != B || v.size(0) != memorySeq)
            throw new ArgumentException("Attention tensors must have matching batch and memory dimensions.");

        var q_h = q.reshape(new long[] { querySeq, B, nhead, head_dim }).permute(1, 2, 0, 3);
        var k_h = k.reshape(new long[] { memorySeq, B, nhead, head_dim }).permute(1, 2, 0, 3);
        var v_h = v.reshape(new long[] { memorySeq, B, nhead, head_dim }).permute(1, 2, 0, 3);

        var k_h_t = k_h.transpose(2, 3);
        var attn = (q_h * attn_scale).matmul(k_h_t);
        if (key_padding_mask is not null)
        {
            if (key_padding_mask.dim() != 2 || key_padding_mask.size(0) != B ||
                key_padding_mask.size(1) != memorySeq)
                throw new ArgumentException("Key padding mask must have shape [batch, memory sequence].");
            var maskBias = key_padding_mask.to_type(attn.dtype).unsqueeze(1).unsqueeze(1) * -1.0e9f;
            attn = attn + maskBias;
        }
        attn = functional.softmax(attn, dim: -1);
        var attn_out = attn.matmul(v_h);

        attn_out = attn_out.permute(2, 0, 1, 3).reshape(new long[] { querySeq, B, d_model });
        return attn_out;
    }

    public Dictionary<string, object> forward(
        Tensor tgt,
        Tensor memory,
        Tensor? tgt_key_padding_mask = null,
        Tensor? memory_key_padding_mask = null,
        Tensor? pos = null,
        Tensor? query_pos = null,
        Tensor? tgt_mask = null,
        Tensor? memory_mask = null)
    {
        return forward_post(tgt, memory, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, tgt_mask, memory_mask);
    }

    private Dictionary<string, object> forward_post(
        Tensor tgt,
        Tensor memory,
        Tensor? tgt_key_padding_mask,
        Tensor? memory_key_padding_mask,
        Tensor? pos,
        Tensor? query_pos,
        Tensor? tgt_mask,
        Tensor? memory_mask)
    {
        // Self-attention
        var q_tgt = WithPosEmbed(tgt, query_pos);
        var k_tgt = WithPosEmbed(tgt, query_pos);
        var v_tgt = tgt;

        var q_s = self_attn_q_proj.forward(q_tgt);
        var k_s = self_attn_k_proj.forward(k_tgt);
        var v_s = self_attn_v_proj.forward(v_tgt);

        var self_out = self_attn_o_proj.forward(dot_product_attention(q_s, k_s, v_s, tgt_key_padding_mask));

        var tgt2 = tgt + self_out;
        tgt2 = norm1.forward(tgt2);

        // Cross-attention (memory)
        var q_cross = tgt2;
        var k_cross = WithPosEmbed(memory, pos);
        var v_cross = memory;

        var q_c = cross_attn_q_proj.forward(q_cross);
        var k_c = cross_attn_k_proj.forward(k_cross);
        var v_c = cross_attn_v_proj.forward(v_cross);

        var cross_out = cross_attn_o_proj.forward(dot_product_attention(q_c, k_c, v_c, memory_key_padding_mask));

        var tgt3 = tgt2 + cross_out;
        tgt3 = norm2.forward(tgt3);

        // FFN
        var ffn = linear2.forward(activation.forward(linear1.forward(tgt3)));
        var tgt4 = norm3.forward(tgt3 + ffn);

        return new Dictionary<string, object>
        {
            { "output", tgt4 }
        };
    }

    public static Tensor WithPosEmbed(Tensor tensor, Tensor? pos)
    {
        return pos is null ? tensor : tensor + pos;
    }
}

/// <summary>
/// Transformer encoder for SAM3 DETR-style encoder.
/// Matches: detector_model.detr_encoder
/// Stacks multiple encoder layers and concatenates multi-scale features.
/// </summary>
public class Sam3TransformerEncoder : Module
{
    private readonly List<Sam3TransformerEncoderLayer> layers;
    private readonly int d_model;
    private readonly int num_feature_levels;
    private readonly Parameter? level_embed;

    public Sam3TransformerEncoder(
        int d_model = 256,
        int nhead = 8,
        int num_layers = 6,
        int dim_feedforward = 2048,
        int num_feature_levels = 3)
        : base(nameof(Sam3TransformerEncoder))
    {
        this.d_model = d_model;
        this.num_feature_levels = num_feature_levels;

        layers = new List<Sam3TransformerEncoderLayer>();
        for (int i = 0; i < num_layers; i++)
        {
            var layer = new Sam3TransformerEncoderLayer(d_model, nhead, dim_feedforward);
            layers.Add(layer);
            register_module("layer_" + i.ToString(), layer);
        }

        if (num_feature_levels > 1)
        {
            level_embed = Parameter(torch.randn(new long[] { num_feature_levels, d_model }), requires_grad: true);
        }

        RegisterComponents();
    }

    public Dictionary<string, object> forward(
        IList<Tensor> src,
        IList<Tensor>? src_key_padding_masks,
        IList<Tensor>? pos,
        Tensor? prompt = null,
        Tensor? prompt_key_padding_mask = null)
    {
        var srcList = src as List<Tensor> ?? src.ToList();
        var srcFlatten = new List<Tensor>();
        var maskFlatten = new List<Tensor?>();
        var lvlPosEmbedFlatten = new List<Tensor>();
        var spatialShapes = new List<long>();
        var hasMask = src_key_padding_masks is not null && src_key_padding_masks.Any(m => m is not null);

        for (int lvl = 0; lvl < srcList.Count; lvl++)
        {
            var s = srcList[lvl];
            var bs = s.size(0);
            var c = s.size(1);
            var h = s.size(2);
            var w = s.size(3);
            spatialShapes.Add(h * w);

            var srcFlat = flatten(s, start_dim: 2).transpose(1, 2);
            srcFlatten.Add(srcFlat);

            if (hasMask && src_key_padding_masks![lvl] is not null)
            {
                maskFlatten.Add(flatten(src_key_padding_masks![lvl], start_dim: 1));
            }

            Tensor? posFlat = null;
            if (pos is not null && lvl < pos.Count && pos[lvl] is not null)
            {
                posFlat = flatten(pos[lvl], start_dim: 2).transpose(1, 2);
            }

            if (level_embed is not null && posFlat is not null)
            {
                posFlat = posFlat + level_embed![lvl].unsqueeze(0).unsqueeze(0);
            }
            lvlPosEmbedFlatten.Add(posFlat ?? srcFlat.clone());
        }

        var srcConcat = cat(srcFlatten, dim: 1);  // [bs, total_spatial, d_model]
        var lvlPosConcat = cat(lvlPosEmbedFlatten, dim: 1);  // [bs, total_spatial, d_model]

        // Transpose to seq-first format [total_spatial, bs, d_model] for decoder compatibility
        Tensor output = srcConcat.transpose(0, 1);
        lvlPosConcat = lvlPosConcat.transpose(0, 1);
        var imageMask = hasMask ? cat(maskFlatten.Where(m => m is not null).Select(m => m!).ToArray(), dim: 1) : null;
        var crossMemory = prompt ?? output;
        foreach (var layer in layers)
        {
            var result = layer.forward(output, crossMemory,
                imageMask, prompt_key_padding_mask, pos: null, query_pos: lvlPosConcat);
            output = (Tensor)result["output"];
        }

        var levelStartIndices = new long[] { 0 };
        for (int i = 0; i < spatialShapes.Count - 1; i++)
        {
            levelStartIndices = levelStartIndices.Concat(new long[] { levelStartIndices.Last() + spatialShapes[i] }).ToArray();
        }

        return new Dictionary<string, object>
        {
            { "memory", output },
            { "pos_embed", lvlPosConcat },
            { "level_start_index", levelStartIndices },
            { "spatial_shapes", spatialShapes }
        };
    }
}
