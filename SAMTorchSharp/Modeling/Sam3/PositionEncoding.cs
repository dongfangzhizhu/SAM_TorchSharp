// Copyright (c) Sapiens AI. All rights reserved.

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System;

namespace SAMTorchSharp.Modeling.Sam3;

/// <summary>
/// Sinusoidal position encoding.
/// Ported from sam3/model/position_encoding.py
/// </summary>
public class Sam3PositionEmbeddingSine : Module
{
    private readonly int pos_dim;
    private readonly bool normalize;
    private readonly float scale;

    public Sam3PositionEmbeddingSine(int num_pos_feats = 64, float scale = 10000.0f, bool normalize = false, float? scale_override = null)
        : base(nameof(Sam3PositionEmbeddingSine))
    {
        pos_dim = num_pos_feats;
        this.normalize = normalize;
        this.scale = scale_override ?? scale;
    }

    public Tensor forward(Tensor x)
    {
        var B = x.size(0);
        var C = x.size(1);
        var H = x.size(2);
        var W = x.size(3);

        var y_embed = arange(1, H + 1).to(x.dtype).unsqueeze(0).expand(new long[] { B, -1 }).to(ScalarType.Float32);
        var x_embed = arange(1, W + 1).to(x.dtype).unsqueeze(0).expand(new long[] { B, -1 }).to(ScalarType.Float32);

        if (normalize)
        {
            var eps = 1e-6f;
            y_embed = y_embed / (y_embed[-1, -1].item<float>() + eps);
            x_embed = x_embed / (x_embed[-1, -1].item<float>() + eps);
        }

        var dim_t = arange(0, pos_dim, 2).to(ScalarType.Float32) / pos_dim;
        var prod_dim_t = scale * dim_t;
        var factor_x = sin(x_embed.unsqueeze(-1) * prod_dim_t);
        var factor_y = sin(y_embed.unsqueeze(-1) * prod_dim_t);

        var vec_x = cat(new[] { factor_x.cos(), factor_x.sin() }, dim: -1);
        var vec_y = cat(new[] { factor_y.cos(), factor_y.sin() }, dim: -1);

        var pos = cat(new[] { vec_y, vec_x }, dim: -1).view(new long[] { B, -1, H, W });
        return pos;
    }
}
