// Copyright (c) Sapiens AI. All rights reserved.

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System;
using System.Collections.Generic;

namespace SAMTorchSharp.Modeling.Sam3;

/// <summary>
/// Standard ViT Block used in the checkpoint's vision_encoder.backbone.
/// Uses separate q_proj, k_proj, v_proj, o_proj (not fused qkv).
/// Architecture: [32 layers, embed_dim=1024, num_heads=16, head_dim=64, mlp_ratio=4.625 (4736)]
/// </summary>
public class Sam3ViTBlock : Module
{
    private readonly LayerNorm norm1;
    private readonly LayerNorm norm2;
    private readonly Linear q_proj;
    private readonly Linear k_proj;
    private readonly Linear v_proj;
    private readonly Linear o_proj;
    private readonly Linear fc1;
    private readonly Linear fc2;
    private readonly long dim;
    private readonly long num_heads;
    private readonly long head_dim;
    private readonly float scale;

    public Sam3ViTBlock(long dim, int num_heads = 16, double drop_path = 0.0)
        : base(nameof(Sam3ViTBlock))
    {
        this.dim = dim;
        this.num_heads = num_heads;
        this.head_dim = dim / num_heads;
        this.scale = 1.0f / (float)Math.Sqrt((double)head_dim);

        norm1 = LayerNorm(new long[] { dim }, elementwise_affine: true);
        norm2 = LayerNorm(new long[] { dim }, elementwise_affine: true);

        q_proj = Linear(dim, dim);
        k_proj = Linear(dim, dim);
        v_proj = Linear(dim, dim);
        o_proj = Linear(dim, dim);

        // MLP: 1024 -> 4736 -> 1024 (mlp_ratio = 4736/1024 = 4.625)
        fc1 = Linear(dim, 4736);
        fc2 = Linear(4736, dim);

        RegisterComponents();
    }

    public Tensor forward(Tensor x)
    {
        // Self-attention with separate projections
        var attn_out = forward_attention(norm1.forward(x));
        x = x + attn_out;

        // MLP
        x = x + fc2.forward(functional.gelu(fc1.forward(norm2.forward(x))));
        return x;
    }

    private Tensor forward_attention(Tensor x)
    {
        var B = x.size(0);
        var N = x.size(1);
        var C = x.size(2);

        var q = q_proj.forward(x).reshape(new long[] { B, N, num_heads, head_dim }).transpose(1, 2); // [B, H, N, hd]
        var k = k_proj.forward(x).reshape(new long[] { B, N, num_heads, head_dim }).transpose(1, 2);
        var v = v_proj.forward(x).reshape(new long[] { B, N, num_heads, head_dim }).transpose(1, 2);

        var q_scaled = q * scale;
        var attn = functional.scaled_dot_product_attention(q_scaled, k, v);

        var attn_out = attn.transpose(1, 2).reshape(new long[] { B, N, C });
        return o_proj.forward(attn_out);
    }
}

/// <summary>
/// Patch Embedding for standard ViT.
/// Checkpoint: Conv2d(3, 1024, 14x14, stride=14)
/// </summary>
public class Sam3ViTPatchEmbed : Module
{
    private readonly Conv2d proj;
    private readonly LayerNorm norm;

    public Sam3ViTPatchEmbed(int patch_size = 14, int in_chans = 3, int embed_dim = 1024)
        : base(nameof(Sam3ViTPatchEmbed))
    {
        proj = Conv2d(in_chans, embed_dim, kernelSize: patch_size, stride: patch_size);
        norm = LayerNorm(embed_dim);
        RegisterComponents();
    }

    public Tuple<Tensor, long, long> forward(Tensor x)
    {
        x = proj.forward(x);
        var B = x.size(0);
        var C = x.size(1);
        var H = x.size(2);
        var W = x.size(3);
        x = flatten(x, start_dim: 2).transpose(1, 2);
        x = norm.forward(x);
        return Tuple.Create(x, H, W);
    }
}

/// <summary>
/// Standard ViT Backbone for SAM3 checkpoint.
/// Matches: detector_model.vision_encoder.backbone
/// Architecture: [32 layers, embed_dim=1024, num_heads=16, patch_size=14]
///
/// Position embeddings: Initialized to pretrained size (576 = 24x24 for 336x336).
/// Interpolated to target size during forward pass if needed.
/// </summary>
public class Sam3ViTBackbone : Module
{
    private readonly Sam3ViTPatchEmbed patch_embed;
    private readonly Parameter pos_embed_field;
    private readonly List<Sam3ViTBlock> blocks;
    private readonly LayerNorm norm;
    private readonly int embed_dim;
    private readonly int depth;
    private readonly int patch_size;
    private readonly int target_num_patches;

    public Sam3ViTBackbone(
        int img_size = 1008,
        int patch_size = 14,
        int in_chans = 3,
        int embed_dim = 1024,
        int depth = 32,
        int num_heads = 16)
        : base(nameof(Sam3ViTBackbone))
    {
        this.embed_dim = embed_dim;
        this.depth = depth;
        this.patch_size = patch_size;
        this.target_num_patches = (img_size / patch_size) * (img_size / patch_size);

        patch_embed = new Sam3ViTPatchEmbed(patch_size, in_chans, embed_dim);

        blocks = new List<Sam3ViTBlock>();
        for (int i = 0; i < depth; i++)
        {
            blocks.Add(new Sam3ViTBlock(embed_dim, num_heads));
            register_module("block_" + i.ToString(), blocks[i]);
        }

        norm = LayerNorm(embed_dim);

        // Initialize with checkpoint-pretrained size (576 = 24x24 for 336x336 pretraining).
        // The forward() method interpolates to the target size (5184 = 72x72 for 1008x1008) when needed.
        const int pretrained_patches = 576;
        pos_embed_field = Parameter(zeros(new long[] { 1, pretrained_patches, embed_dim }), requires_grad: false);
        register_buffer("pos_embed_buffer", pos_embed_field);

        RegisterComponents();
    }

    public int EmbedDim => embed_dim;
    public int Depth => depth;
    public int PatchSize => patch_size;

    public Tensor forward(Tensor x)
    {
        var patchResult = patch_embed.forward(x);
        var x_embed = patchResult.Item1;
        var H_out = patchResult.Item2;
        var W_out = patchResult.Item3;

        var B = x_embed.size(0);
        var num_tokens = x_embed.size(1);

        var pos = get_buffer("pos_embed_buffer");
        if (pos is not null && pos.numel() > 0)
        {
            var pos_tokens = pos.size(1);
            if (pos_tokens != num_tokens)
            {
                var src_h = (long)Math.Sqrt(pos_tokens);
                var src_w = src_h;
                var tgt_h = H_out;
                var tgt_w = W_out;

                var pos_2d = pos.squeeze(0).transpose(0, 1).reshape(new long[] { embed_dim, src_h, src_w });
                pos_2d = functional.interpolate(pos_2d.unsqueeze(0),
                    size: new long[] { tgt_h, tgt_w },
                    mode: InterpolationMode.Bilinear, align_corners: false).squeeze(0);
                pos_2d = pos_2d.transpose(0, 1).reshape(new long[] { 1, tgt_h * tgt_w, embed_dim });
                x_embed = x_embed + pos_2d.to(x_embed.device);
            }
            else
            {
                x_embed = x_embed + pos.to(x_embed.device);
            }
        }

        foreach (var block in blocks)
        {
            x_embed = block.forward(x_embed);
        }

        return norm.forward(x_embed);
    }

    public Tensor get_pos_embed()
    {
        return get_buffer("pos_embed_buffer")!;
    }
}
