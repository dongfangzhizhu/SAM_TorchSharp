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
/// Geometry encoder transformer layer for SAM3.
/// </summary>
public class Sam3GeometryEncoderLayer : Module
{
    private readonly LayerNorm layer_norm1;
    private readonly LayerNorm layer_norm2;
    private readonly LayerNorm layer_norm3;
    private readonly Linear self_attn_q_proj;
    private readonly Linear self_attn_k_proj;
    private readonly Linear self_attn_v_proj;
    private readonly Linear self_attn_o_proj;
    private readonly Linear cross_attn_q_proj;
    private readonly Linear cross_attn_k_proj;
    private readonly Linear cross_attn_v_proj;
    private readonly Linear cross_attn_o_proj;
    private readonly Linear mlp_fc1;
    private readonly Linear mlp_fc2;
    private readonly long d_model;
    private readonly int num_heads;
    private readonly long head_dim;

    public Sam3GeometryEncoderLayer(long d_model = 256, int num_heads = 8)
        : base(nameof(Sam3GeometryEncoderLayer))
    {
        this.d_model = d_model;
        this.num_heads = num_heads;
        this.head_dim = d_model / num_heads;

        layer_norm1 = LayerNorm(d_model);
        layer_norm2 = LayerNorm(d_model);
        layer_norm3 = LayerNorm(d_model);

        self_attn_q_proj = Linear(d_model, d_model);
        self_attn_k_proj = Linear(d_model, d_model);
        self_attn_v_proj = Linear(d_model, d_model);
        self_attn_o_proj = Linear(d_model, d_model);

        cross_attn_q_proj = Linear(d_model, d_model);
        cross_attn_k_proj = Linear(d_model, d_model);
        cross_attn_v_proj = Linear(d_model, d_model);
        cross_attn_o_proj = Linear(d_model, d_model);

        mlp_fc1 = Linear(d_model, 2048);
        mlp_fc2 = Linear(2048, d_model);

        RegisterComponents();
    }

    private Tensor forward_self_attn(Tensor x)
    {
        // SAM3 uses PyTorch's sequence-first convention: [sequence, batch, channels].
        var N = x.size(0);
        var B = x.size(1);

        var q = self_attn_q_proj.forward(x).reshape(new long[] { N, B, num_heads, head_dim }).permute(1, 2, 0, 3);
        var k = self_attn_k_proj.forward(x).reshape(new long[] { N, B, num_heads, head_dim }).permute(1, 2, 0, 3);
        var v = self_attn_v_proj.forward(x).reshape(new long[] { N, B, num_heads, head_dim }).permute(1, 2, 0, 3);

        var attn = functional.scaled_dot_product_attention(q, k, v);

        var attn_out = attn.permute(2, 0, 1, 3).reshape(new long[] { N, B, d_model });
        return self_attn_o_proj.forward(attn_out);
    }

    private Tensor forward_cross_attn(Tensor query, Tensor memory, Tensor? memory_pos = null)
    {
        var N = query.size(0);
        var B = query.size(1);
        var M = memory.size(0);
        if (memory.size(1) != B)
            throw new ArgumentException("Query and memory batch sizes must match.", nameof(memory));

        var keyValue = memory_pos is null ? memory : memory + memory_pos;

        var q = cross_attn_q_proj.forward(query).reshape(new long[] { N, B, num_heads, head_dim }).permute(1, 2, 0, 3);
        var k = cross_attn_k_proj.forward(keyValue).reshape(new long[] { M, B, num_heads, head_dim }).permute(1, 2, 0, 3);
        var v = cross_attn_v_proj.forward(memory).reshape(new long[] { M, B, num_heads, head_dim }).permute(1, 2, 0, 3);

        var attn = functional.scaled_dot_product_attention(q, k, v);

        var attn_out = attn.permute(2, 0, 1, 3).reshape(new long[] { N, B, d_model });
        return cross_attn_o_proj.forward(attn_out);
    }

    public Tuple<Tensor, Tensor> forward(Tensor query, Tensor memory, Tensor? memory_pos = null)
    {
        if (query.dim() != 3 || memory.dim() != 3)
            throw new ArgumentException("Geometry attention expects [sequence, batch, channels] tensors.");
        if (query.size(1) != memory.size(1))
            throw new ArgumentException("Query and memory batch sizes must match.", nameof(memory));

        var q_normed = layer_norm1.forward(query);
        var self_attn_out = forward_self_attn(q_normed);
        query = query + self_attn_out;

        var ca_normed = layer_norm2.forward(query);
        var cross_attn_out = forward_cross_attn(ca_normed, memory);
        query = query + cross_attn_out;

        var mlp_out = mlp_fc2.forward(functional.relu(mlp_fc1.forward(layer_norm3.forward(query))));
        query = query + mlp_out;

        return Tuple.Create<Tensor, Tensor>(query, null);
    }
}

/// <summary>
/// Geometry Encoder for SAM3 - simplified working implementation.
/// </summary>
public class Sam3GeometryEncoderNew : Module
{
    private readonly List<Sam3GeometryEncoderLayer> layers;
    private readonly Embedding label_embed;
    private readonly Embedding cls_embed;
    private readonly Linear final_proj;
    private readonly LayerNorm final_norm;
    private readonly LayerNorm encode_norm;
    private readonly Sam3PositionEmbeddingSine position_encoding;
    private readonly int d_model;
    private readonly int num_geo_layers;

    public Sam3GeometryEncoderNew(int d_model = 256, int num_geo_layers = 3)
        : base(nameof(Sam3GeometryEncoderNew))
    {
        this.d_model = d_model;
        this.num_geo_layers = num_geo_layers;

        label_embed = Embedding(2, d_model);
        cls_embed = Embedding(1, d_model);
        final_proj = Linear(d_model, d_model);
        final_norm = LayerNorm(d_model);
        encode_norm = LayerNorm(d_model);
        position_encoding = new Sam3PositionEmbeddingSine(num_pos_feats: d_model, normalize: true);

        layers = new List<Sam3GeometryEncoderLayer>();
        for (int i = 0; i < num_geo_layers; i++)
        {
            layers.Add(new Sam3GeometryEncoderLayer(d_model));
            register_module("geo_layer_" + i.ToString(), layers[i]);
        }

        RegisterComponents();
    }

    public Tuple<Tensor, Tensor> forward(
        Sam3Prompt geo_prompt,
        List<Tensor> img_feats,
        List<long[]> img_sizes)
    {
        var bs = img_feats[0].size(0);
        var device = img_feats[0].device;

        bool hasPoints = geo_prompt.points is not null && geo_prompt.points.numel() > 0;
        bool hasBoxes = geo_prompt.boxes is not null && geo_prompt.boxes.numel() > 0;
        bool hasMasks = geo_prompt.masks is not null && geo_prompt.masks.numel() > 0;

        var allFeats = new List<Tensor>();
        var allMasks = new List<Tensor>();

        if (hasPoints)
        {
            var points = geo_prompt.points; // [batch, num_points, 2]
            var num_points = (int)points.size(1);
            var h = (int)img_sizes[0][0];
            var w = (int)img_sizes[0][1];

            // Normalize points to [-1, 1] for grid_sample
            var normW = (float)(w - 1);
            var normH = (float)(h - 1);
            var normFactor = tensor(new float[] { 1.0f / normW, 1.0f / normH }, device: device);
            var coords = points * normFactor.unsqueeze(0) * 2.0f - 1.0f;
            coords = coords.clamp(-1.0f, 1.0f);

            // grid_sample expects [batch, out_h, out_w, 2]
            var grid = coords.unsqueeze(2); // [batch, num_points, 1, 2]

            var sampled = functional.grid_sample(
                img_feats[0],
                grid,
                mode: GridSampleMode.Bilinear,
                padding_mode: GridSamplePaddingMode.Zeros,
                align_corners: false
            );
            // sampled: [batch, d_model, num_points, 1]
            var result = sampled.squeeze(-1).permute(new long[] { 2, 0, 1 }); // [num_points, batch, d_model]
            allFeats.Add(result);
            allMasks.Add(zeros(new long[] { bs, num_points }, dtype: ScalarType.Bool, device: device));
        }

        if (hasBoxes)
        {
            var boxes = geo_prompt.boxes; // [batch, num_boxes, 4]
            var num_boxes = (int)boxes.size(1);
            var levelFeats = new List<Tensor>();

            for (int lvl = 0; lvl < Math.Min(4, img_feats.Count); lvl++)
            {
                var h = (int)img_sizes[lvl][0];
                var w = (int)img_sizes[lvl][1];

                // Scale boxes to feature map pixel coords
                var scaleTensor = tensor(new float[] { (float)w, (float)h, (float)w, (float)h }, device: boxes.device);
                var scaledBoxes = boxes * scaleTensor; // [batch, num_boxes, 4]

                // Use adaptive_avg_pool2d per box region
                var B = (int)scaledBoxes.size(0);
                var NB = (int)scaledBoxes.size(1);
                var C = (int)img_feats[lvl].size(1);
                var FH = (int)img_feats[lvl].size(2);
                var FW = (int)img_feats[lvl].size(3);

                var boxList = new List<Tensor>();
                for (int b = 0; b < B; b++)
                {
                    for (int nb = 0; nb < NB; nb++)
                    {
                        float x1f = scaledBoxes[b, nb, 0].item<float>();
                        float y1f = scaledBoxes[b, nb, 1].item<float>();
                        float x2f = scaledBoxes[b, nb, 2].item<float>();
                        float y2f = scaledBoxes[b, nb, 3].item<float>();

                        int x1 = Math.Max(0, Math.Min((int)Math.Round(x1f), FW - 1));
                        int y1 = Math.Max(0, Math.Min((int)Math.Round(y1f), FH - 1));
                        int x2 = Math.Max(x1 + 1, Math.Min((int)Math.Round(x2f), FW));
                        int y2 = Math.Max(y1 + 1, Math.Min((int)Math.Round(y2f), FH));

                        var roi = extractROI(img_feats[lvl], b, x1, y1, x2, y2);
                        if (roi.size(2) > 0 && roi.size(3) > 0)
                        {
                            var pooledFeat = functional.adaptive_avg_pool2d(roi, new long[] { 7, 7 });
                            boxList.Add(pooledFeat);
                        }
                        else
                        {
                            boxList.Add(zeros(new long[] { 1, C, 7, 7 }, device: device));
                        }
                    }
                }

                var stacked = stack(boxList.ToArray(), dim: 0); // [B*NB, C, 7, 7]
                var reshaped = stacked.reshape(new long[] { B, NB, C, 7, 7 }); // [B, NB, C, 7, 7]
                var boxMean = reshaped.mean(new long[] { 3, 4 }); // [B, NB, C]
                var seqFormat = boxMean.permute(new long[] { 1, 0, 2 }); // [NB, B, C]
                levelFeats.Add(seqFormat);
            }

            var combinedBoxes = levelFeats.Count == 1 ? levelFeats[0] : cat(levelFeats.ToArray(), dim: 0);
            var totalSeqLen = (int)combinedBoxes.size(0);
            allFeats.Add(combinedBoxes);
            allMasks.Add(zeros(new long[] { bs, totalSeqLen }, dtype: ScalarType.Bool, device: device));
        }

        Tensor geoCombined;
        if (allFeats.Count == 0)
        {
            geoCombined = cls_embed.weight.view(1, 1, d_model).repeat(1, bs, 1).to(device);
        }
        else if (allFeats.Count == 1)
        {
            geoCombined = allFeats[0];
        }
        else
        {
            geoCombined = cat(allFeats.ToArray(), dim: 0);
        }

        var finalSeqLen = geoCombined.size(0);
        var geoMask = zeros(new long[] { bs, finalSeqLen }, dtype: ScalarType.Bool, device: device);

        // Match SequenceGeometryEncoder: post projection/norm followed by the
        // pre-norm self/cross-attention encoder stack and final norm.
        geoCombined = final_norm.forward(final_proj.forward(geoCombined));
        if (layers.Count > 0)
        {
            var image = img_feats[^1];
            var imagePos = position_encoding.forward(image)
                .flatten(2).transpose(1, 2).transpose(0, 1);
            var imageMemory = image.flatten(2).transpose(1, 2).transpose(0, 1);
            foreach (var layer in layers)
            {
                geoCombined = layer.forward(geoCombined, imageMemory, imagePos).Item1;
            }
            geoCombined = encode_norm.forward(geoCombined);
        }

        return Tuple.Create(geoCombined, geoMask);
    }

    public Tuple<Tensor, Tensor> EncodeEmptyPrompt(long batchSize, Device device)
    {
        if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));
        var clsToken = cls_embed.weight.view(1, 1, d_model).repeat(1, batchSize, 1).to(device);
        var clsMask = zeros(new long[] { batchSize, 1 }, dtype: ScalarType.Bool, device: device);
        return Tuple.Create(clsToken, clsMask);
    }

    /// <summary>
    /// Extract ROI from feature map using narrow (C# equivalent of slicing).
    /// </summary>
    private Tensor extractROI(Tensor feat, int batchIdx, int x1, int y1, int x2, int y2)
    {
        var roi = feat.narrow(0, batchIdx, 1);
        roi = roi.narrow(2, y1, y2 - y1);
        roi = roi.narrow(3, x1, x2 - x1);
        return roi;
    }

    public long GetGeoSequenceLength(Sam3Prompt geo_prompt, List<Tensor> img_feats, List<long[]> img_sizes)
    {
        long totalSeq = 0;
        if (geo_prompt.points is not null && geo_prompt.points.numel() > 0)
            totalSeq += geo_prompt.num_points;
        if (geo_prompt.boxes is not null && geo_prompt.boxes.numel() > 0)
        {
            for (int lvl = 0; lvl < Math.Min(4, img_feats.Count); lvl++)
            {
                var feat_size = img_sizes[lvl];
                totalSeq += feat_size[0] * feat_size[1];
            }
        }
        if (geo_prompt.masks is not null && geo_prompt.masks.numel() > 0)
            totalSeq += geo_prompt.masks.size(2) * geo_prompt.masks.size(3);
        return totalSeq;
    }
}
