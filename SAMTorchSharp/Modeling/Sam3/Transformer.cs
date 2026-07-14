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
/// Generic MLP helper.
/// </summary>
public class Sam3MLP : Module
{
    private readonly List<Linear> layers;
    private readonly Module activation;

    public Sam3MLP(int in_dim, int hidden_dim, int out_dim, int num_layers = 3, bool sigmoid_output = false)
        : base(nameof(Sam3MLP))
    {
        layers = new List<Linear>();
        activation = sigmoid_output ? Sigmoid() : GELU();

        layers.Add(Linear(in_dim, hidden_dim));
        for (int i = 1; i < num_layers - 1; i++)
            layers.Add(Linear(hidden_dim, hidden_dim));
        layers.Add(Linear(hidden_dim, out_dim));
    }

    public Tensor forward(Tensor x)
    {
        for (int i = 0; i < layers.Count - 1; i++)
        {
            x = activation.forward(layers[i].forward(x));
        }
        return layers[layers.Count - 1].forward(x);
    }
}

/// <summary>
/// Transformer encoder layer for SAM3.
/// Ported from sam3/model/decoder.py (adapted for encoder usage)
/// </summary>
public class Sam3TransformerEncoderLayer : Module
{
    private readonly MultiheadAttention self_attn;
    private readonly LayerNorm norm1;
    private readonly Sam3MLP linear1;
    private readonly Module activation;
    private readonly Sam3MLP linear2;
    private readonly LayerNorm norm3;
    private readonly float dropout;
    private readonly int d_model;

    public Sam3TransformerEncoderLayer(
        int d_model = 256,
        int nhead = 8,
        int dim_feedforward = 2048,
        float dropout = 0.1f,
        string activation_type = "gelu")
        : base(nameof(Sam3TransformerEncoderLayer))
    {
        this.d_model = d_model;
        this.dropout = dropout;

        self_attn = MultiheadAttention(d_model, nhead, dropout: dropout);
        norm1 = LayerNorm(d_model);
        linear1 = new Sam3MLP(d_model, dim_feedforward, d_model, 1);
        activation = activation_type.ToLower() == "relu" ? ReLU() : GELU();
        linear2 = new Sam3MLP(d_model, dim_feedforward, d_model, 1);
        norm3 = LayerNorm(d_model);
    }

    public static Tensor WithPosEmbed(Tensor tensor, Tensor? pos)
    {
        return pos is null ? tensor : tensor + pos;
    }

    public Tensor forward(Tensor src, Tensor? pos = null, Tensor? src_key_padding_mask = null)
    {
        var q = k = WithPosEmbed(src, pos);

        var attn_output = self_attn.forward(q, k, src, key_padding_mask: src_key_padding_mask).Item1;
        var src2 = src + attn_output;
        src2 = norm1.forward(src2);

        var src3 = linear2.forward(activation.forward(linear1.forward(src2)));
        src3 = src2 + dropout_tensor(src3, dropout, training);
        src = norm3.forward(src);

        return src;
    }

    private Tensor dropout_tensor(Tensor x, float p, bool training)
    {
        if (p == 0.0f || !training)
            return x;
        return x * torch.rand(x.shape, device: x.device) / (1.0f - p);
    }
}

/// <summary>
/// SAM3 Transformer encoder.
/// Processes image features with prompt conditioning.
/// Ported from sam3/model/decoder.py (encoder section)
/// </summary>
public class Sam3TransformerEncoder : Module
{
    private readonly List<Sam3TransformerEncoderLayer> layers;

    public Sam3TransformerEncoder(int d_model = 256, int nhead = 8, int num_layers = 1, int dim_feedforward = 2048)
        : base(nameof(Sam3TransformerEncoder))
    {
        layers = new List<Sam3TransformerEncoderLayer>();
        for (int i = 0; i < num_layers; i++)
            layers.Add(new Sam3TransformerEncoderLayer(d_model, nhead, dim_feedforward));
    }

    public Dictionary<string, Tensor> forward(
        List<Tensor> src,
        List<Tensor>? src_pos,
        Tensor? prompt,
        Tensor? prompt_pos,
        Tensor? prompt_key_padding_mask,
        List<Tuple<int, int>>? feat_sizes = null,
        Dictionary<string, object>? encoder_extra_kwargs = null)
    {
        // Concatenate multi-level features
        var memory = cat(src, dim: 0);
        var pos = src_pos != null ? cat(src_pos, dim: 0) : null;

        // Process through encoder layers
        foreach (var l in layers)
        {
            memory = l.forward(memory, pos);
        }

        // Split back into levels
        var level_start_index = new List<long>();
        var spatial_shapes = new List<long>();
        var output_feats = new List<Tensor>();
        var pos_offset = 0;

        if (feat_sizes != null)
        {
            foreach (var (h, w) in feat_sizes)
            {
                var n = h * w;
                spatial_shapes.Add(n);
                level_start_index.Add(pos_offset);
                output_feats.Add(memory.narrow(0, pos_offset, n));
                pos_offset += n;
            }
        }
        else
        {
            spatial_shapes.Add(memory.size(0));
            level_start_index.Add(0);
            output_feats.Add(memory);
        }

        var padding_mask = torch.ones(new long[] { 1, memory.size(0) }, device: memory.device);

        return new Dictionary<string, Tensor>
        {
            { "memory", cat(output_feats, dim: 0) },
            { "pos_embed", pos != null ? cat(output_feats.Select((_, i) => pos.narrow(0, level_start_index[i], spatial_shapes[i])).ToList(), dim: 0) : memory.clone() },
            { "padding_mask", padding_mask },
            { "level_start_index", torch.tensor(level_start_index.ToArray()) },
            { "spatial_shapes", torch.tensor(spatial_shapes.ToArray()) },
            { "valid_ratios", torch.ones(1, spatial_shapes.Count, 2, device: memory.device) }
        };
    }
}
