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
/// Transformer decoder layer for SAM3.
/// Ported from sam3/model/decoder.py
/// </summary>
public class Sam3TransformerDecoderLayer : Module
{
    private readonly MultiheadAttention self_attn;
    private readonly MultiheadAttention cross_attn_pixel;
    private readonly MultiheadAttention? cross_attn_text;
    private readonly LayerNorm norm1;
    private readonly LayerNorm norm2;
    private readonly LayerNorm? catext_norm;
    private readonly Sam3MLP linear1;
    private readonly Module activation;
    private readonly Sam3MLP linear2;
    private readonly LayerNorm norm3;
    private readonly float dropout;
    private readonly bool use_text_cross_attention;

    public Sam3TransformerDecoderLayer(
        int d_model = 256,
        int nhead = 8,
        int dim_feedforward = 2048,
        float dropout = 0.1f,
        string activation_type = "gelu",
        bool use_text_cross_attention = false)
        : base(nameof(Sam3TransformerDecoderLayer))
    {
        this.dropout = dropout;
        this.use_text_cross_attention = use_text_cross_attention;

        self_attn = MultiheadAttention(d_model, nhead, dropout: dropout);
        cross_attn_pixel = MultiheadAttention(d_model, nhead, dropout: dropout);
        cross_attn_text = use_text_cross_attention ? MultiheadAttention(d_model, nhead, dropout: dropout) : null;

        norm1 = LayerNorm(d_model);
        norm2 = LayerNorm(d_model);
        catext_norm = use_text_cross_attention ? LayerNorm(d_model) : null;

        linear1 = new Sam3MLP(d_model, dim_feedforward, d_model, 1);
        activation = activation_type.ToLower() == "relu" ? ReLU() : GELU();
        linear2 = new Sam3MLP(d_model, dim_feedforward, d_model, 1);
        norm3 = LayerNorm(d_model);
    }

    public static Tensor WithPosEmbed(Tensor tensor, Tensor? pos)
    {
        return pos is null ? tensor : tensor + pos;
    }

    public Tuple<Tensor, Tensor?> forward(
        Tensor tgt,
        Tensor? tgt_query_pos,
        Tensor memory,
        Tensor? memory_pos,
        Tensor? memory_key_padding_mask,
        Tensor? memory_text,
        Tensor? text_attention_mask)
    {
        // Self attention
        var q = WithPosEmbed(tgt, tgt_query_pos);
        var k = q;
        var tgt2 = self_attn.forward(q, k, tgt, attn_mask: null, need_weights: false).Item1;
        tgt = tgt + tgt2;
        tgt = norm1.forward(tgt);

        // Cross attention with pixel features
        q = WithPosEmbed(tgt, tgt_query_pos);
        var cross_out = cross_attn_pixel.forward(q, memory, tgt, key_padding_mask: memory_key_padding_mask, need_weights: false);
        tgt2 = cross_out.Item1;
        tgt = tgt + tgt2;
        tgt = norm2.forward(tgt);

        // Cross attention with text (optional)
        if (use_text_cross_attention && memory_text != null && cross_attn_text != null)
        {
            tgt2 = cross_attn_text.forward(WithPosEmbed(tgt, tgt_query_pos), memory_text, tgt,
                key_padding_mask: text_attention_mask, need_weights: false).Item1;
            tgt = tgt + tgt2;
            tgt = catext_norm!.forward(tgt);
        }

        // FFN
        var tgt3 = linear2.forward(activation.forward(linear1.forward(tgt)));
        tgt = tgt + dropout_tensor(tgt3, dropout, training);
        tgt = norm3.forward(tgt);

        return Tuple.Create<Tensor, Tensor?>(tgt, null);
    }

    private Tensor dropout_tensor(Tensor x, float p, bool training)
    {
        if (p == 0.0f || !training)
            return x;
        return x * torch.rand(x.shape, device: x.device) / (1.0f - p);
    }
}

/// <summary>
/// Transformer decoder for SAM3 (DETR-style).
/// Ported from sam3/model/decoder.py
/// </summary>
public class Sam3TransformerDecoder : Module
{
    private readonly List<Sam3TransformerDecoderLayer> layers;
    private readonly Embedding query_embed;
    private readonly Sam3MLP bbox_embed;
    private readonly int num_layers;
    private readonly int num_queries;
    private readonly int _d_model;

    public int d_model => _d_model;

    public Sam3TransformerDecoder(
        int d_model = 256,
        int nhead = 8,
        int num_layers = 6,
        int num_queries = 900,
        int dim_feedforward = 2048,
        bool use_text_cross_attention = false)
        : base(nameof(Sam3TransformerDecoder))
    {
        _d_model = d_model;
        this.num_queries = num_queries;
        this.num_layers = num_layers;

        query_embed = Embedding(num_queries, d_model);

        layers = new List<Sam3TransformerDecoderLayer>();
        for (int i = 0; i < num_layers; i++)
            layers.Add(new Sam3TransformerDecoderLayer(
                d_model, nhead, dim_feedforward,
                use_text_cross_attention: use_text_cross_attention));

        bbox_embed = new Sam3MLP(d_model, dim_feedforward, 4, 3);
    }

    public Tensor get_query_embed() => query_embed.weight;

    public Tuple<Tensor, Tensor?, Tensor?, Tensor?> forward(
        Tensor tgt,
        Tensor memory,
        Tensor? pos_embed,
        Tensor? memory_key_padding_mask,
        Tensor? memory_text,
        Tensor? text_attention_mask,
        List<Tuple<int, int>>? feat_sizes = null)
    {
        var bs = memory.size(1);
        var d_model = _d_model;

        var hs = new List<Tensor>();
        var reference_points = torch.zeros(new long[] { num_queries, bs, 4 }, device: memory.device);

        for (int i = 0; i < num_layers; i++)
        {
            var layer = layers[i];
            var layerResult = layer.forward(
                tgt, pos_embed, memory, pos_embed, memory_key_padding_mask, memory_text, text_attention_mask); var tgt_out = layerResult.Item1;

            hs.Add(tgt_out);
            tgt = tgt_out;
        }

        var hs_cat = stack(hs, dim: 0); // [num_layers, num_queries, bs, d_model]
        return Tuple.Create(hs_cat, reference_points, null, null);
    }
}
