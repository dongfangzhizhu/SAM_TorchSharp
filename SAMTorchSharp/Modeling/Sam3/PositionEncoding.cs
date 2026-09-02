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

        var dim_t = FrequencyDivisors(x.device);

        var pos_x = x_embed.unsqueeze(-1) / dim_t.unsqueeze(0).unsqueeze(0); // [B, H, W, num_pos_feats]
        var pos_y = y_embed.unsqueeze(-1) / dim_t.unsqueeze(0).unsqueeze(0); // [B, H, W, num_pos_feats]

        var pos_x_out = InterleaveSinCos(pos_x);
        var pos_y_out = InterleaveSinCos(pos_y);

        var pos = cat(new[] { pos_y_out, pos_x_out }, dim: -1)
            .permute(new long[] { 0, 3, 1, 2 }); // [B, 2*num_pos_feats, H, W]

        return pos;
    }

    /// <summary>
    /// Encode normalized one-dimensional x/y coordinate tensors using the
    /// same frequencies as the two-dimensional image position encoding.
    /// </summary>
    public Tuple<Tensor, Tensor> EncodeXY(Tensor x, Tensor y)
    {
        if (x.dim() != 1 || y.dim() != 1 || x.size(0) != y.size(0))
            throw new ArgumentException("Normalized x and y coordinates must be equal-length vectors.");

        var divisors = FrequencyDivisors(x.device);
        var posX = InterleaveSinCos(x.unsqueeze(-1) * scale / divisors);
        var posY = InterleaveSinCos(y.unsqueeze(-1) * scale / divisors);
        return Tuple.Create(posX, posY);
    }

    /// <summary>
    /// Encode normalized center-x, center-y, width and height box coordinates.
    /// Output order matches the official model: y encoding, x encoding, h, w.
    /// </summary>
    public Tensor EncodeBoxes(Tensor x, Tensor y, Tensor width, Tensor height)
    {
        var (posX, posY) = EncodeXY(x, y);
        if (width.dim() != 1 || height.dim() != 1 ||
            width.size(0) != x.size(0) || height.size(0) != x.size(0))
            throw new ArgumentException("Box coordinates must be equal-length vectors.");
        return cat(new[] { posY, posX, height.unsqueeze(-1), width.unsqueeze(-1) }, dim: 1);
    }

    private Tensor FrequencyDivisors(Device device)
    {
        var indices = arange(0, num_pos_feats, dtype: ScalarType.Float32, device: device);
        return pow(temperature, 2.0f * floor(indices / 2.0f) / num_pos_feats);
    }

    private static Tensor InterleaveSinCos(Tensor values)
    {
        var pairCount = values.size(-1) / 2;
        var even = values.narrow(-1, 0, pairCount * 2).reshape(values.shape[..^1].Append(pairCount).Append(2).ToArray());
        return stack(new[] { even.select(-1, 0).sin(), even.select(-1, 1).cos() }, dim: -1)
            .flatten(start_dim: -2);
    }
}
