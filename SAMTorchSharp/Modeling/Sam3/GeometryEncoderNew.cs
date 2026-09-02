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
    private readonly Linear points_direct_project;
    private readonly Linear boxes_direct_project;
    private readonly Linear points_pos_enc_project;
    private readonly Linear boxes_pos_enc_project;
    private readonly Linear points_pool_project;
    private readonly Conv2d boxes_pool_project;
    private readonly Linear final_proj;
    private readonly LayerNorm final_norm;
    private readonly LayerNorm encode_norm;
    private readonly LayerNorm vision_layer_norm;
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
        points_direct_project = Linear(2, d_model);
        points_pool_project = Linear(d_model, d_model);
        boxes_direct_project = Linear(4, d_model);
        boxes_pool_project = Conv2d(d_model, d_model, kernelSize: 7);
        points_pos_enc_project = Linear(d_model, d_model);
        boxes_pos_enc_project = Linear(d_model + 2, d_model);
        final_proj = Linear(d_model, d_model);
        final_norm = LayerNorm(d_model);
        encode_norm = LayerNorm(d_model);
        vision_layer_norm = LayerNorm(d_model);
        position_encoding = new Sam3PositionEmbeddingSine(num_pos_feats: d_model, normalize: true);

        layers = new List<Sam3GeometryEncoderLayer>();
        for (int i = 0; i < num_geo_layers; i++)
        {
            layers.Add(new Sam3GeometryEncoderLayer(d_model));
            register_module("geo_layer_" + i.ToString(), layers[i]);
        }

        RegisterComponents();
    }

    private static Tensor SamplePoints(Tensor image, Tensor points)
    {
        // image: [B,C,H,W], points: [N,B,2] in [0,1]. grid_sample with
        // align_corners=false matches the bilinear sampler used by SAM3.
        var grid = points.permute(1, 0, 2).unsqueeze(2) * 2 - 1;
        return functional.grid_sample(image, grid, mode: GridSampleMode.Bilinear,
            padding_mode: GridSamplePaddingMode.Zeros, align_corners: false)
            .squeeze(3).permute(2, 0, 1);
    }

    private static Tensor RoiAlign(Tensor image, Tensor boxes, int outputSize)
    {
        // image: [B,C,H,W], boxes: [N,B,4] cx,cy,w,h normalized to [0,1].
        // The grid uses one bin-center sample per output bin, equivalent to
        // torchvision roi_align with sampling_ratio=1.
        var b = image.size(0);
        var c = image.size(1);
        var h = image.size(2);
        var w = image.size(3);
        var n = boxes.size(0);
        var x1 = (boxes.select(-1, 0) - boxes.select(-1, 2) / 2) * w;
        var y1 = (boxes.select(-1, 1) - boxes.select(-1, 3) / 2) * h;
        var x2 = (boxes.select(-1, 0) + boxes.select(-1, 2) / 2) * w;
        var y2 = (boxes.select(-1, 1) + boxes.select(-1, 3) / 2) * h;
        var rows = torch.arange(outputSize, dtype: ScalarType.Float32, device: image.device)
            .view(1, 1, outputSize, 1);
        var cols = torch.arange(outputSize, dtype: ScalarType.Float32, device: image.device)
            .view(1, 1, 1, outputSize);
        var gridX = (x1.unsqueeze(-1).unsqueeze(-1) + (cols + 0.5) *
            (x2 - x1).unsqueeze(-1).unsqueeze(-1) / outputSize - 0.5) / w;
        var gridY = (y1.unsqueeze(-1).unsqueeze(-1) + (rows + 0.5) *
            (y2 - y1).unsqueeze(-1).unsqueeze(-1) / outputSize - 0.5) / h;
        var grid = torch.stack(new[] { gridX.expand(n, b, outputSize, outputSize),
            gridY.expand(n, b, outputSize, outputSize) }, dim: -1)
            .permute(1, 0, 2, 3, 4).reshape(b, n * outputSize, outputSize, 2) * 2 - 1;
        var sampled = functional.grid_sample(image, grid, mode: GridSampleMode.Bilinear,
            padding_mode: GridSamplePaddingMode.Zeros, align_corners: false);
        return sampled.view(b, c, n, outputSize, outputSize)
            .permute(0, 2, 1, 3, 4).reshape(b * n, c, outputSize, outputSize);
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
        var pooledImage = vision_layer_norm.forward(img_feats[^1].permute(0, 2, 3, 1))
            .permute(0, 3, 1, 2);

        if (hasPoints)
        {
            var points = geo_prompt.points!; // [num_points, batch, 2]
            RequirePromptShape(points, bs, 2, nameof(geo_prompt.points));
            var labels = geo_prompt.point_labels ?? ones(new long[] { points.size(0), bs }, dtype: ScalarType.Int64, device: device);
            RequireLabelShape(labels, points.size(0), bs, nameof(geo_prompt.point_labels));
            var pointCount = points.size(0);
            var (posX, posY) = position_encoding.EncodeXY(
                points.select(-1, 0).flatten(), points.select(-1, 1).flatten());
            var pos = cat(new[] { posY, posX }, dim: 1)
                .reshape(new long[] { pointCount, bs, d_model });
            var encoded = points_direct_project.forward(points)
                + points_pos_enc_project.forward(pos)
                + label_embed.forward(labels.to_type(ScalarType.Int64));
            encoded = encoded + points_pool_project.forward(SamplePoints(pooledImage, points));
            allFeats.Add(encoded);
            allMasks.Add(geo_prompt.point_mask ?? zeros(new long[] { bs, points.size(0) }, dtype: ScalarType.Bool, device: device));
        }

        if (hasBoxes)
        {
            var boxes = geo_prompt.boxes!; // [num_boxes, batch, 4]
            RequirePromptShape(boxes, bs, 4, nameof(geo_prompt.boxes));
            var num_boxes = (int)boxes.size(0);
            var labels = geo_prompt.box_labels ?? ones(new long[] { num_boxes, bs }, dtype: ScalarType.Int64, device: device);
            RequireLabelShape(labels, num_boxes, bs, nameof(geo_prompt.box_labels));
            var (boxX, boxY) = position_encoding.EncodeXY(
                boxes.select(-1, 0).flatten(), boxes.select(-1, 1).flatten());
            var boxPos = cat(new[] {
                boxY,
                boxX,
                boxes.select(-1, 3).flatten().unsqueeze(-1),
                boxes.select(-1, 2).flatten().unsqueeze(-1)
            }, dim: 1).reshape(new long[] { num_boxes, bs, d_model + 2 });
            var encoded = boxes_direct_project.forward(boxes)
                + boxes_pos_enc_project.forward(boxPos)
                + label_embed.forward(labels.to_type(ScalarType.Int64));
            var pooledBoxes = RoiAlign(pooledImage, boxes, 7);
            var boxPool = boxes_pool_project.forward(pooledBoxes)
                .view(bs, num_boxes, d_model).transpose(0, 1);
            encoded = encoded + boxPool;
            allFeats.Add(encoded);
            allMasks.Add(geo_prompt.box_mask ?? zeros(new long[] { bs, num_boxes }, dtype: ScalarType.Bool, device: device));
        }

        Tensor geoCombined;
        if (allFeats.Count == 0)
        {
            geoCombined = cls_embed.weight.view(1, 1, d_model).repeat(1, bs, 1).to(device);
            allMasks.Add(zeros(new long[] { bs, 1 }, dtype: ScalarType.Bool, device: device));
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
        var geoMask = allMasks.Count == 1
            ? allMasks[0]
            : cat(allMasks.ToArray(), dim: 1);
        if (geoMask.size(1) != finalSeqLen)
            throw new InvalidOperationException("Geometry prompt masks do not match the encoded sequence length.");

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

    private static void RequirePromptShape(Tensor prompt, long batchSize, long channels, string name)
    {
        if (prompt.dim() != 3 || prompt.size(1) != batchSize || prompt.size(2) != channels)
            throw new ArgumentException($"{name} must have shape [sequence, batch, {channels}].", name);
    }

    private static void RequireLabelShape(Tensor labels, long sequenceLength, long batchSize, string name)
    {
        if (labels.dim() != 2 || labels.size(0) != sequenceLength || labels.size(1) != batchSize)
            throw new ArgumentException($"{name} must have shape [sequence, batch].", name);
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
