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
/// MLP as used in Vision Transformer, MLP-Mixer and related networks.
/// Ported from sam3/model/vitdet.py
/// </summary>
public class Sam3Mlp : Module
{
    public readonly Linear fc1;
    public readonly Linear fc2;
    public readonly Module act;
    public readonly Dropout drop1;
    public readonly Dropout drop2;
    public readonly Module norm;

    public Sam3Mlp(
        long in_features,
        long? hidden_features = null,
        long? out_features = null,
        string act_layer = "gelu",
        Module? norm_layer = null,
        bool bias = true,
        double drop = 0.0)
        : base(nameof(Sam3Mlp))
    {
        out_features ??= in_features;
        hidden_features ??= in_features;

        fc1 = Linear(in_features, hidden_features.Value, bias: bias);
        act = GetActivation(act_layer);
        drop1 = Dropout(drop);
        norm = norm_layer ?? Identity();
        fc2 = Linear(hidden_features.Value, out_features.Value, bias: bias);
        drop2 = Dropout(drop);
    }

    public Tensor forward(Tensor x)
    {
        x = fc1.forward(x);
        x = act.forward(x);
        x = drop1.forward(x);
        x = norm.forward(x);
        x = fc2.forward(x);
        x = drop2.forward(x);
        return x;
    }

    private static Module GetActivation(string act_layer)
    {
        return act_layer.ToLower() switch
        {
            "relu" => ReLU(),
            "sigmoid" => Sigmoid(),
            _ => GELU()
        };
    }
}

/// <summary>
/// DropPath (Stochastic Depth) as used in ViTDet.
/// Ported from sam3/model/vitdet.py
/// </summary>
public class Sam3DropPath : Module
{
    private readonly float drop_prob;

    public Sam3DropPath(float drop_prob = 0.0f)
        : base(nameof(Sam3DropPath))
    {
        this.drop_prob = drop_prob;
    }

    public Tensor forward(Tensor x)
    {
        if (drop_prob == 0.0f)
            return x;

        var keep_prob = 1.0f - drop_prob;
        var shape = new long[] { x.size(0) };
        var random_tensor = keep_prob + torch.rand(shape, dtype: x.dtype, device: x.device);
        random_tensor = floor(random_tensor);
        var output = x.div(keep_prob) * random_tensor;
        return output;
    }
}

/// <summary>
/// Compute axial rotary position embeddings.
/// Ported from sam3/model/vitdet.py
/// </summary>
public class Sam3ComputeAxialCis : Module
{
    private readonly int dim;
    private readonly int end_x;
    private readonly int end_y;
    private readonly float theta;
    private readonly float scale_pos;
    private readonly int offset;

    public Sam3ComputeAxialCis(
        int dim,
        int end_x,
        int end_y,
        float theta = 10000.0f,
        float scale_pos = 1.0f,
        int offset = 0)
        : base(nameof(Sam3ComputeAxialCis))
    {
        this.dim = dim;
        this.end_x = end_x;
        this.end_y = end_y;
        this.theta = theta;
        this.scale_pos = scale_pos;
        this.offset = offset;
    }

    public Tensor get_freqs_cis()
    {
        var dim_half = dim / 4;
        var exponents = arange(0, dim, 4).narrow(0, 0, dim_half).to(ScalarType.Float32) / dim;
        var inv_freq = 1.0 / pow(theta, exponents);

        var t_x = (arange(0, end_x * end_y) % end_x).to(ScalarType.Float32) * scale_pos + offset;
        var t_y = (arange(0, end_x * end_y) / end_x).to(ScalarType.Float32) * scale_pos + offset;

        var freqs_x = outer(t_x, inv_freq);
        var freqs_y = outer(t_y, inv_freq);

        var freqs_cis_x = ones_like(freqs_x).polar(freqs_x);
        var freqs_cis_y = ones_like(freqs_y).polar(freqs_y);

        return cat(new[] { freqs_cis_x, freqs_cis_y }, dim: -1);
    }
}

/// <summary>
/// Apply rotary position embeddings to query/key tensors.
/// Ported from sam3/model/vitdet.py
/// </summary>
public class Sam3ApplyRotaryEnc : Module
{
    private readonly Tensor freqs_cis;

    public Sam3ApplyRotaryEnc(Tensor freqs_cis)
        : base(nameof(Sam3ApplyRotaryEnc))
    {
        this.freqs_cis = freqs_cis;
    }

    public Tuple<Tensor, Tensor> forward(Tensor xq, Tensor xk)
    {
        if (freqs_cis is null || freqs_cis.numel() == 0)
            return Tuple.Create(xq, xk);

        var ndim = xq.dim();
        var shape = new long[ndim];
        for (int i = 0; i < ndim; i++)
            shape[i] = i >= ndim - 2 ? xq.size(i) : 1;

        var freqs_cis_view = freqs_cis.view(shape);

        // Reshape for complex multiplication
        var xq_ = xq.to(ScalarType.Float32).reshape(new long[] { xq.size(0), xq.size(1), -1, 2 });
        var xq_complex = view_as_complex(xq_);
        var xq_out = view_as_real(xq_complex * freqs_cis_view).flatten(ndim - 2).to(xq.dtype);

        if (xk.size(ndim - 2) == 0)
            return Tuple.Create(xq_out, xk);

        var xk_ = xk.to(ScalarType.Float32).reshape(new long[] { xk.size(0), xk.size(1), -1, 2 });
        var xk_complex = view_as_complex(xk_);

        // Repeat freqs along seq_len dim to match k seq_len
        var r = xk_complex.size(-2) / xq_complex.size(-2);
        var repeat_shape = new long[freqs_cis_view.dims.Length];
        for (int i = 0; i < repeat_shape.Length; i++)
            repeat_shape[i] = i >= repeat_shape.Length - 2 ? freqs_cis_view.size(i) : 1;
        var freqs_cis_repeat = freqs_cis_view.reshape(repeat_shape).repeat(
            new long[] { 1, 1, r, 1 });

        var xk_out = view_as_real(xk_complex * freqs_cis_repeat).flatten(ndim - 2).to(xk.dtype);
        return Tuple.Create(xq_out, xk_out);
    }
}

/// <summary>
/// Window partition for swin-like attention.
/// Ported from sam3/model/vitdet.py
/// </summary>
public class Sam3WindowPartition : Module
{
    private readonly int window_size;

    public Sam3WindowPartition(int window_size)
        : base(nameof(Sam3WindowPartition))
    {
        this.window_size = window_size;
    }

    public Tuple<Tensor, Tuple<long, long>> forward(Tensor x)
    {
        var B = x.size(0);
        var H = x.size(1);
        var W = x.size(2);
        var C = x.size(3);

        var pad_h = (window_size - H % window_size) % window_size;
        var pad_w = (window_size - W % window_size) % window_size;
        var x_pad = functional.pad(x, new long[] { 0, 0, 0, pad_w, 0, pad_h, 0, 0 });

        var Hp = H + pad_h;
        var Wp = W + pad_w;

        var x_windows = x_pad.reshape(new long[] { B, Hp / window_size, window_size, Wp / window_size, window_size, C });
        x_windows = x_windows.permute(new long[] { 0, 1, 3, 2, 4, 5 }).reshape(new long[] {
            B * (Hp / window_size) * (Wp / window_size), window_size, window_size, C
        });

        return Tuple.Create(x_windows, Tuple.Create(Hp, Wp));
    }
}

/// <summary>
/// Window unpartition.
/// Ported from sam3/model/vitdet.py
/// </summary>
public class Sam3WindowUnpartition : Module
{
    private readonly int window_size;
    private readonly int orig_H;
    private readonly int orig_W;

    public Sam3WindowUnpartition(int window_size, int H, int W)
        : base(nameof(Sam3WindowUnpartition))
    {
        this.window_size = window_size;
        orig_H = H;
        orig_W = W;
    }

    public Tensor forward(Tensor windows, Tuple<long, long> Hp_Wp)
    {
        var Hp = Hp_Wp.Item1;
        var Wp = Hp_Wp.Item2;
        var B = windows.size(0) / (Hp / window_size) / (Wp / window_size);
        var C = windows.size(3);

        var x = windows.reshape(new long[] {
            B, Hp / window_size, Wp / window_size, window_size, window_size, C
        });
        x = x.permute(new long[] { 0, 1, 3, 2, 4, 5 }).reshape(new long[] { B, Hp, Wp, C });

        if (orig_H < Hp || orig_W < Wp)
            x = x.narrow(1, 0, orig_H).narrow(2, 0, orig_W);

        return x;
    }
}

/// <summary>
/// Attention module with RoPE support.
/// Ported from sam3/model/vitdet.py
/// </summary>
public class Sam3Attention : Module
{
    private readonly long dim;
    private readonly long num_heads;
    private readonly long head_dim;
    private readonly float scale;
    private readonly Linear qkv;
    private readonly Dropout attn_drop;
    private readonly Linear proj;
    private readonly Dropout proj_drop;
    private readonly Sam3ApplyRotaryEnc? rope;

    public Sam3Attention(
        long dim,
        int num_heads = 8,
        float qkv_bias = 0.0f,
        float attn_drop = 0.0f,
        float proj_drop = 0.0f,
        Sam3ApplyRotaryEnc? rope = null)
        : base(nameof(Sam3Attention))
    {
        this.dim = dim;
        this.num_heads = num_heads;
        this.head_dim = dim / num_heads;
        this.scale = 1.0f / (float)Math.Sqrt(head_dim);
        this.rope = rope;

        qkv = Linear(dim, dim * 3);
        this.attn_drop = Dropout(attn_drop);
        proj = Linear(dim, dim);
        this.proj_drop = Dropout(proj_drop);
    }

    public Tensor forward(Tensor x)
    {
        var B = x.size(0);
        var N = x.size(1);
        var C = x.size(2);

        var qkv_weight = qkv.forward(x).reshape(new long[] { B, N, 3, num_heads, (int)head_dim }).permute(new long[] { 2, 0, 3, 1, 4 });
        var q = qkv_weight[0];
        var k = qkv_weight[1];
        var v = qkv_weight[2];

        if (rope != null)
        {
            var tuple = rope.forward(q, k);
            q = tuple.Item1;
            k = tuple.Item2;
        }

        var attn = (q * scale).matmul(k.transpose(-2, -1));
        attn = attn.softmax(dim: -1);
        attn = attn_drop.forward(attn);

        x = attn.matmul(v).reshape(new long[] { B, N, C });
        x = proj.forward(x);
        x = proj_drop.forward(x);
        return x;
    }
}

/// <summary>
/// Basic ViT Block with residual connection.
/// Ported from sam3/model/vitdet.py
/// </summary>
public class Sam3Block : Module
{
    private readonly long dim;
    private readonly Sam3Attention attn;
    private readonly Sam3Mlp mlp;
    private readonly LayerNorm norm1;
    private readonly LayerNorm norm2;
    private readonly Sam3DropPath drop_path;

    public Sam3Block(
        long dim,
        int num_heads = 8,
        float mlp_ratio = 4.0f,
        float attn_drop = 0.0f,
        float proj_drop = 0.0f,
        float drop = 0.0f,
        Sam3ApplyRotaryEnc? rope = null)
        : base(nameof(Sam3Block))
    {
        this.dim = dim;

        norm1 = LayerNorm(new long[] { dim }, elementwise_affine: true);
        attn = new Sam3Attention(dim, num_heads: num_heads, attn_drop: attn_drop, proj_drop: proj_drop, rope: rope);

        norm2 = LayerNorm(new long[] { dim }, elementwise_affine: true);
        var mlp_hidden = (long)(dim * mlp_ratio);
        mlp = new Sam3Mlp(dim, hidden_features: mlp_hidden, drop: drop);

        drop_path = new Sam3DropPath(drop);
    }

    public Tensor forward(Tensor x)
    {
        x = x + drop_path.forward(attn.forward(norm1.forward(x)));
        x = x + drop_path.forward(mlp.forward(norm2.forward(x)));
        return x;
    }
}

/// <summary>
/// Patch embedding for ViT.
/// Ported from sam3/model/vitdet.py
/// </summary>
public class Sam3PatchEmbed : Module
{
    private readonly Conv2d proj;
    private readonly LayerNorm norm;

    public Sam3PatchEmbed(
        int patch_size = 16,
        int in_chans = 3,
        int embed_dim = 768)
        : base(nameof(Sam3PatchEmbed))
    {
        proj = Conv2d(in_chans, embed_dim, kernelSize: patch_size, stride: patch_size);
        norm = LayerNorm(embed_dim);
    }

    public Tuple<Tensor, Tuple<long, long>> forward(Tensor x)
    {
        x = proj.forward(x);
        var B = x.size(0);
        var C = x.size(1);
        var H = x.size(2);
        var W = x.size(3);
        x = x.flatten(2).transpose(1, 2);
        x = norm.forward(x);
        return Tuple.Create(x, Tuple.Create(H, W));
    }
}

/// <summary>
/// ViTDet backbone for SAM3.
/// Combines patch embedding with stacked transformer blocks.
/// Ported from sam3/model/vitdet.py
/// </summary>
public class Sam3ViTDetBackbone : Module
{
    private readonly Sam3PatchEmbed patch_embed;
    private readonly List<Sam3Block> blocks;
    private readonly LayerNorm? norm;
    private readonly int[] stage_depths;
    private readonly int[] channel_list;

    public Sam3ViTDetBackbone(
        int patch_size = 16,
        int in_chans = 3,
        int embed_dim = 768,
        int depth = 12,
        int num_heads = 12,
        float mlp_ratio = 4.0f,
        float drop = 0.0f,
        float attn_drop = 0.0f,
        bool use_rope = false)
        : base(nameof(Sam3ViTDetBackbone))
    {
        patch_embed = new Sam3PatchEmbed(patch_size, in_chans, embed_dim);
        blocks = new List<Sam3Block>();

        Sam3ApplyRotaryEnc? rope_module = null;
        if (use_rope)
        {
            var cis = new Sam3ComputeAxialCis(embed_dim / num_heads, patch_size, patch_size);
            rope_module = new Sam3ApplyRotaryEnc(cis.get_freqs_cis());
        }

        for (int i = 0; i < depth; i++)
        {
            var block_drop = drop * i / Math.Max(depth - 1, 1);
            var block = new Sam3Block(
                dim: embed_dim,
                num_heads: num_heads,
                mlp_ratio: mlp_ratio,
                attn_drop: attn_drop,
                proj_drop: attn_drop,
                drop: block_drop,
                rope: rope_module);
            blocks.Add(block);
        }

        norm = LayerNorm(embed_dim);

        // Define stage depths (typical ViTDet: [3, 3, 9, 3] for 4 stages)
        stage_depths = new int[] { 3, 3, 9, 3 };
        channel_list = new int[] { embed_dim, embed_dim, embed_dim, embed_dim };
    }

    public int[] GetChannelList() => channel_list;

    public Tensor forward(Tensor x)
    {
        var patchResult = patch_embed.forward(x); var x_embed = patchResult.Item1; var spatial_shape = patchResult.Item2;
        var B = x_embed.size(0); var N = x_embed.size(1); var C = x_embed.size(2);

        // Register pos_embed buffer
        var pos_embed = torch.zeros(new long[] { 1, N, C }, device: x_embed.device);
        RegisterBuffer("pos_embed", pos_embed);

        var idx = 0;
        Tensor last_feat = null!;

        for (int i = 0; i < blocks.Count; i++)
        {
            x_embed = blocks[i].forward(x_embed + pos_embed);

            // Save intermediate features at certain stages
            if (idx < stage_depths.Length && i == stage_depths.Take(idx + 1).Sum() - 1)
            {
                last_feat = x_embed.reshape(new long[] { B, spatial_shape.Item1, spatial_shape.Item2, C })
                    .permute(new long[] { 0, 3, 1, 2 }).contiguous();
                idx++;
            }
        }

        if (norm != null && last_feat != null)
            last_feat = norm.forward(last_feat.permute(new long[] { 0, 2, 3, 1 })).permute(new long[] { 0, 3, 1, 2 });

        return last_feat;
    }
}
