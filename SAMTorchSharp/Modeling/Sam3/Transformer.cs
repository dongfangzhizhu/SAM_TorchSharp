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
/// Generic MLP helper (2-layer).
/// </summary>
public class Sam3MLP : Module
{
    private readonly Linear linear1;
    private readonly Linear linear2;
    private readonly Module<Tensor, Tensor> activation;
    private readonly Dropout dropout;

    public Sam3MLP(int in_dim, int hidden_dim, int out_dim, int num_layers = 3, bool sigmoid_output = false)
        : base(nameof(Sam3MLP))
    {
        activation = sigmoid_output ? Sigmoid() : GELU();
        linear1 = Linear(in_dim, hidden_dim);
        linear2 = Linear(hidden_dim, out_dim);
        dropout = Dropout(0.1f);
    }

    public Tensor forward(Tensor x)
    {
        x = linear2.forward(dropout.forward(activation.forward(linear1.forward(x))));
        return x;
    }
}

/// <summary>
/// Transformer encoder layer for SAM3.
/// </summary>
public class Sam3TransformerEncoderLayer : Module
{
    private readonly MultiheadAttention self_attn;
    private readonly MultiheadAttention cross_attn_image;
    private readonly Linear linear1;
    private readonly Linear linear2;
    private readonly Module<Tensor, Tensor> activation;
    private readonly float dropout;
    private readonly LayerNorm norm1;
    private readonly LayerNorm norm2;
    private readonly LayerNorm norm3;
    private readonly bool pre_norm;
    private readonly bool pos_enc_at_attn;
    private readonly bool pos_enc_at_cross_attn_queries;
    private readonly bool pos_enc_at_cross_attn_keys;
    private readonly int d_model;
    private readonly int dim_feedforward;

    public Sam3TransformerEncoderLayer(
        int d_model = 256,
        int nhead = 8,
        int dim_feedforward = 2048,
        float dropout = 0.1f,
        string activation_type = "gelu",
        bool pre_norm = true,
        bool pos_enc_at_attn = false,
        bool pos_enc_at_cross_attn_queries = false,
        bool pos_enc_at_cross_attn_keys = false)
        : base(nameof(Sam3TransformerEncoderLayer))
    {
        this.d_model = d_model;
        this.dim_feedforward = dim_feedforward;
        this.pre_norm = pre_norm;
        this.pos_enc_at_attn = pos_enc_at_attn;
        this.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries;
        this.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys;

        self_attn = MultiheadAttention(d_model, nhead, dropout: dropout);
        cross_attn_image = MultiheadAttention(d_model, nhead, dropout: dropout);

        linear1 = Linear(d_model, dim_feedforward);
        linear2 = Linear(dim_feedforward, d_model);
        activation = activation_type.ToLower() == "relu" ? ReLU() : GELU();

        this.dropout = dropout;

        norm1 = LayerNorm(d_model);
        norm2 = LayerNorm(d_model);
        norm3 = LayerNorm(d_model);
    }

    public static Tensor WithPosEmbed(Tensor tensor, Tensor? pos)
    {
        return pos is null ? tensor : tensor + pos;
    }

    private Tensor dropout_tensor(Tensor x, float p)
    {
        if (p == 0.0f || !training)
            return x;
        return x * torch.rand(x.shape, device: x.device, dtype: x.dtype) / (1.0f - p);
    }

    public Tensor forward(
        Tensor tgt,
        Tensor memory,
        Tensor? tgt_mask = null,
        Tensor? memory_mask = null,
        Tensor? tgt_key_padding_mask = null,
        Tensor? memory_key_padding_mask = null,
        Tensor? pos = null,
        Tensor? query_pos = null)
    {
        if (pre_norm)
        {
            return forward_pre(tgt, memory, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, tgt_mask, memory_mask);
        }
        else
        {
            return forward_post(tgt, memory, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, tgt_mask, memory_mask);
        }
    }

    private Tensor forward_post(
        Tensor tgt,
        Tensor memory,
        Tensor? tgt_key_padding_mask,
        Tensor? memory_key_padding_mask,
        Tensor? pos,
        Tensor? query_pos,
        Tensor? tgt_mask = null,
        Tensor? memory_mask = null)
    {
        var q = WithPosEmbed(tgt, query_pos);
        var k = q;

        var attn_out = self_attn.forward(q, k, tgt, tgt_key_padding_mask, false, tgt_mask).Item1;
        var tgt2 = tgt + dropout_tensor(attn_out, dropout);
        tgt2 = norm1.forward(tgt2);

        var cross_q = WithPosEmbed(tgt2, query_pos);
        var cross_k = WithPosEmbed(memory, pos);
        var cross_out = cross_attn_image.forward(cross_q, cross_k, memory, memory_key_padding_mask, false, memory_mask).Item1;
        var tgt3 = tgt2 + dropout_tensor(cross_out, dropout);
        tgt3 = norm2.forward(tgt3);

        var ffn = linear2.forward(dropout_tensor(activation.forward(linear1.forward(tgt3)), dropout));
        var tgt4 = tgt3 + ffn;
        tgt4 = norm3.forward(tgt4);

        return tgt4;
    }

    private Tensor forward_pre(
        Tensor tgt,
        Tensor memory,
        Tensor? tgt_key_padding_mask,
        Tensor? memory_key_padding_mask,
        Tensor? pos,
        Tensor? query_pos,
        Tensor? tgt_mask = null,
        Tensor? memory_mask = null)
    {
        var normed_tgt = norm1.forward(tgt);
        var q = WithPosEmbed(normed_tgt, query_pos);
        var k = q;

        var attn_out = self_attn.forward(q, k, normed_tgt, tgt_key_padding_mask, false, null).Item1;
        var tgt2 = tgt + dropout_tensor(attn_out, dropout);

        var normed_tgt2 = norm2.forward(tgt2);
        var cross_q = WithPosEmbed(normed_tgt2, query_pos);
        var cross_k = WithPosEmbed(memory, pos);
        var cross_out = cross_attn_image.forward(cross_q, cross_k, normed_tgt2, memory_key_padding_mask, false, null).Item1;
        var tgt3 = tgt2 + dropout_tensor(cross_out, dropout);

        var normed_tgt3 = norm3.forward(tgt3);
        var ffn = linear2.forward(dropout_tensor(activation.forward(linear1.forward(normed_tgt3)), dropout));
        var tgt4 = tgt3 + ffn;

        return tgt4;
    }
}

/// <summary>
/// SAM3 Transformer encoder.
/// </summary>
public class Sam3TransformerEncoder : Module
{
    private readonly List<Sam3TransformerEncoderLayer> layers;
    private readonly int num_layers;
    private readonly int d_model;
    private readonly int num_feature_levels;
    private readonly Parameter? level_embed;

    public Sam3TransformerEncoder(
        int d_model = 256,
        int nhead = 8,
        int num_layers = 1,
        int dim_feedforward = 2048,
        int num_feature_levels = 4,
        bool use_act_checkpoint = false)
        : base(nameof(Sam3TransformerEncoder))
    {
        this.d_model = d_model;
        this.num_feature_levels = num_feature_levels;

        layers = new List<Sam3TransformerEncoderLayer>();
        for (int i = 0; i < num_layers; i++)
        {
            layers.Add(new Sam3TransformerEncoderLayer(
                d_model, nhead, dim_feedforward,
                pre_norm: true,
                pos_enc_at_attn: false,
                pos_enc_at_cross_attn_queries: false,
                pos_enc_at_cross_attn_keys: false));
        }

        this.num_layers = num_layers;
        if (num_feature_levels > 1)
        {
            level_embed = Parameter(torch.randn(new long[] { num_feature_levels, d_model }), requires_grad: true);
        }
    }

    public Dictionary<string, object> forward(
        IList<Tensor> src,
        IList<Tensor>? src_key_padding_masks,
        IList<Tensor>? pos)
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

        var srcConcat = cat(srcFlatten, dim: 1);
        var lvlPosConcat = cat(lvlPosEmbedFlatten, dim: 1);

        Tensor output = srcConcat;
        foreach (var layer in layers)
        {
            output = layer.forward(output, output);
        }

        var levelStartIndices = new long[] { 0 };
        for (int i = 0; i < spatialShapes.Count - 1; i++)
        {
            levelStartIndices = levelStartIndices.Concat(new long[] { levelStartIndices.Last() + spatialShapes[i] }).ToArray();
        }

        return new Dictionary<string, object>
        {
            { "memory", output.transpose(0, 1) },
            { "padding_mask", hasMask ? cat(maskFlatten.Where(m => m is not null).Cast<Tensor>().ToArray(), dim: 1).transpose(0, 1) : null },
            { "pos_embed", lvlPosConcat.transpose(0, 1) },
            { "level_start_index", torch.tensor(levelStartIndices) },
            { "spatial_shapes", torch.tensor(spatialShapes.ToArray()) },
            { "valid_ratios", torch.ones(new long[] { 1, spatialShapes.Count, 2 }, device: srcConcat.device) }
        };
    }
}
