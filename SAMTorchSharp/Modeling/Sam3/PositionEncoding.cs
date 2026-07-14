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
    private readonly int num_pos_feats;
    private readonly float temperature;
    private readonly bool normalize;
    private readonly float scale;

    public Sam3PositionEmbeddingSine(int num_pos_feats = 64, float temperature = 10000.0f, bool normalize = true, float? scale_override = null)
        : base(nameof(Sam3PositionEmbeddingSine))
    {
        // num_pos_feats is total; we use half for sin and half for cos
        this.num_pos_feats = num_pos_feats / 2;
        this.temperature = temperature;
        this.normalize = normalize;
        this.scale = scale_override ?? (2.0f * (float)Math.PI);
    }

    public Tensor forward(Tensor x)
    {
        // x: [B, C, H, W]
        var B = x.size(0);
        var H = x.size(2);
        var W = x.size(3);

        var y_embed = arange(1, H + 1, dtype: ScalarType.Float32, device: x.device).unsqueeze(0).unsqueeze(-1).expand(new long[] { B, H, W });
        var x_embed = arange(1, W + 1, dtype: ScalarType.Float32, device: x.device).unsqueeze(0).unsqueeze(0).expand(new long[] { B, H, W });

        if (normalize)
        {
            var eps = 1e-6f;
            var y_max = y_embed.max();
            var x_max = x_embed.max();
            y_embed = y_embed / (y_max + eps) * scale;
            x_embed = x_embed / (x_max + eps) * scale;
        }

        var dim_t = arange(0, num_pos_feats, dtype: ScalarType.Float32, device: x.device);
        dim_t = pow(temperature, 2.0f * (dim_t / 2) / num_pos_feats);

        var pos_x = x_embed.unsqueeze(-1) / dim_t.unsqueeze(0).unsqueeze(0); // [B, H, W, num_pos_feats]
        var pos_y = y_embed.unsqueeze(-1) / dim_t.unsqueeze(0).unsqueeze(0); // [B, H, W, num_pos_feats]

        // Even indices: sin, Odd indices: cos
        var half_n = num_pos_feats / 2;
        var pos_x_sin = pos_x.narrow(-1, 0, half_n).sin();
        var pos_x_cos = pos_x.narrow(-1, half_n, half_n).cos();
        var pos_y_sin = pos_y.narrow(-1, 0, half_n).sin();
        var pos_y_cos = pos_y.narrow(-1, half_n, half_n).cos();

        var pos_x_out = cat(new[] { pos_x_sin, pos_x_cos }, dim: -1);
        var pos_y_out = cat(new[] { pos_y_sin, pos_y_cos }, dim: -1);

        var pos = cat(new[] { pos_y_out, pos_x_out }, dim: -1)
            .permute(new long[] { 0, 3, 1, 2 }); // [B, 2*num_pos_feats, H, W]

        return pos;
    }
}
