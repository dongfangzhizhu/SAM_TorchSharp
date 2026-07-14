// Copyright (c) Sapiens AI. All rights reserved.

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System;
using System.Collections.Generic;

namespace SAMTorchSharp.Modeling.Sam3;

/// <summary>
/// Represents a prompt for SAM3 geometry encoder.
/// </summary>
public class Sam3Prompt
{
    public Tensor? points { get; set; }
    public Tensor? boxes { get; set; }
    public Tensor? masks { get; set; }
    public Tensor? text { get; set; }
    public Tensor? visual_embed { get; set; }
    public Tensor? visual_mask { get; set; }
    public int batch_size { get; set; }
    public int num_points { get; set; } = 0;
    public int num_boxes { get; set; } = 0;
    public int num_masks { get; set; } = 0;
}

/// <summary>
/// Geometry encoder for points, boxes, and masks.
/// </summary>
public class Sam3GeometryEncoder : Module
{
    private readonly int embed_dim;
    private readonly int num_levels;
    private readonly Embedding point_embeddings;
    private readonly Embedding grid_point_embeddings;
    private readonly List<Linear> box_embeddings;
    private readonly Module mask_downsampler;

    public Sam3GeometryEncoder(
        int embed_dim = 256,
        int num_input_point_coords = 2,
        int num_levels = 4)
        : base(nameof(Sam3GeometryEncoder))
    {
        this.embed_dim = embed_dim;
        this.num_levels = num_levels;

        point_embeddings = Embedding(2, embed_dim);
        grid_point_embeddings = Embedding(2, embed_dim);

        box_embeddings = new List<Linear>();
        for (int i = 0; i < num_levels; i++)
            box_embeddings.Add(Linear(4, embed_dim));

        var ct2_0 = ConvTranspose2d(1, embed_dim / 4, kernelSize: 2, stride: 2);
        var gelu = GELU();
        var ct2_1 = ConvTranspose2d(embed_dim / 4, embed_dim, kernelSize: 2, stride: 2);
        mask_downsampler = Sequential(ct2_0, gelu, ct2_1);
    }

    public Tuple<List<Tensor>, List<Tensor?>> forward(
        Sam3Prompt geo_prompt,
        List<Tensor> img_feats,
        List<long[]> img_sizes,
        List<Tensor>? img_pos_embeds = null)
    {
        var geo_feats = new List<Tensor>();
        var geo_masks = new List<Tensor?>();

        if (geo_prompt.points is not null && geo_prompt.points.numel() > 0)
        {
            var points = geo_prompt.points;
            var B = img_feats[0].size(0);
            var feat_size = img_sizes[0];
            var h = (int)feat_size[0];
            var w = (int)feat_size[1];

            var point_coords = points.unsqueeze(0).expand(B, -1, -1);
            var point_features = functional.interpolate(
                img_feats[0],
                size: new long[] { (long)h, (long)w },
                mode: InterpolationMode.Bilinear,
                align_corners: false
            );

            var sampled = sample_points(point_features, point_coords);
            geo_feats.Add(sampled);
            geo_masks.Add(null);
        }

        if (geo_prompt.boxes is not null && geo_prompt.boxes.numel() > 0)
        {
            var boxes = geo_prompt.boxes;
            var B = img_feats[0].size(0);

            for (int lvl = 0; lvl < Math.Min(num_levels, img_feats.Count); lvl++)
            {
                var feat_size_l = img_sizes[lvl];
                var h = (int)feat_size_l[0];
                var w = (int)feat_size_l[1];

                var scale_tensor = torch.tensor(new float[] { (float)w, (float)h, (float)w, (float)h }, device: boxes.device);
                var scaled_boxes = boxes * scale_tensor;

                var roi_feat = extract_roi_features(img_feats[lvl], scaled_boxes, h, w);
                geo_feats.Add(roi_feat);
                geo_masks.Add(torch.ones(new long[] { B, 1, (long)h, (long)w }, device: boxes.device));
            }
        }

        if (geo_prompt.masks is not null && geo_prompt.masks.numel() > 0)
        {
            var mask_feat = ((Module<Tensor, Tensor>)mask_downsampler).forward(geo_prompt.masks);
            geo_feats.Add(mask_feat);
            var maskBs = geo_prompt.masks.size(0);
            var maskC = geo_prompt.masks.size(2);
            var maskH = geo_prompt.masks.size(3);
            geo_masks.Add(torch.ones(new long[] { maskBs, maskC, maskH }, device: geo_prompt.masks.device));
        }

        return Tuple.Create(geo_feats, geo_masks);
    }

    private Tensor sample_points(Tensor features, Tensor point_coords)
    {
        var B = features.size(0);
        var N = point_coords.size(1);

        var coords = point_coords / torch.tensor(new float[] {
            (float)(features.size(3) - 1),
            (float)(features.size(2) - 1)
        }, device: point_coords.device) * 2 - 1;

        var sampled = functional.grid_sample(features, coords.unsqueeze(2).unsqueeze(3),
            mode: GridSampleMode.Bilinear,
            padding_mode: GridSamplePaddingMode.Zeros,
            align_corners: false);

        return sampled.squeeze(3).squeeze(2);
    }

    private Tensor extract_roi_features(Tensor features, Tensor boxes, int h, int w)
    {
        var B = features.size(0);
        return functional.interpolate(features, size: new long[] { (long)h, (long)w }, mode: InterpolationMode.Bilinear, align_corners: false);
    }
}
