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
/// SAM3 Base model integrating all components.
/// Ported from sam3/model/sam3_multiplex_base.py (core model parts)
/// </summary>
public class Sam3Base : Module
{
    public readonly Sam3DualViTDetNeck backbone;
    public readonly Sam3GeometryEncoder geometry_encoder;
    public readonly Sam3TransformerEncoder transformer_encoder;
    public readonly Sam3TransformerDecoder transformer_decoder;
    public readonly Sam3SegmentationHead? segmentation_head;
    public readonly int hidden_dim;
    public readonly int num_feature_levels;
    public readonly bool multimask_output;

    private Tensor? _device_cache = null;

    public Sam3Base(
        Sam3DualViTDetNeck backbone,
        Sam3GeometryEncoder geometry_encoder,
        Sam3TransformerEncoder transformer_encoder,
        Sam3TransformerDecoder transformer_decoder,
        Module? segmentation_head = null,
        int num_feature_levels = 4,
        bool multimask_output = true)
        : base(nameof(Sam3Base))
    {
        this.backbone = backbone;
        this.geometry_encoder = geometry_encoder;
        this.transformer_encoder = transformer_encoder;
        this.transformer_decoder = transformer_decoder;
        this.segmentation_head = segmentation_head;
        this.hidden_dim = transformer_decoder.d_model;
        this.num_feature_levels = num_feature_levels;
        this.multimask_output = multimask_output;
    }

    public Tensor device
    {
        get
        {
            if (_device_cache is null)
            {
                var p = parameters().FirstOrDefault();
                _device_cache = p?.device;
            }
            return _device_cache!;
        }
    }

    /// <summary>
    /// Forward pass for image backbone.
    /// </summary>
    public Dictionary<string, Tensor> forward_backbone(Tensor images)
    {
        // images: [B, C, H, W]
        var backbone_out = backbone.forward(images);
        return new Dictionary<string, Tensor> { { "backbone_out", backbone_out } };
    }

    /// <summary>
    /// Encode prompts (points, boxes, masks, text).
    /// </summary>
    public Tuple<Tensor, Tensor> encode_prompts(
        List<Tensor> img_feats,
        List<Tensor> img_pos_embeds,
        List<long[]> vis_feat_sizes,
        Sam3Prompt geometric_prompt)
    {
        var (geo_feats, geo_masks) = geometry_encoder.forward(
            geometric_prompt, img_feats, vis_feat_sizes, img_pos_embeds);

        var prompt = cat(geo_feats, dim: 0);
        var valid_masks = geo_masks.Where(m => m != null).ToList();
        var prompt_mask = valid_masks.Count > 0 ? cat(valid_masks.Cast<Tensor>(), dim: 1) : null;

        return Tuple.Create(prompt, prompt_mask);
    }

    /// <summary>
    /// Run transformer encoder.
    /// </summary>
    public Dictionary<string, Tensor> run_encoder(
        List<Tensor> img_feats,
        List<Tensor> img_pos_embeds,
        Tensor prompt,
        Tensor? prompt_mask)
    {
        var vis_feat_sizes = img_feats.Select(f => Tuple.Create((int)f.size(2), (int)f.size(3))).ToList();

        var prompt_pos_embed = zeros_like(prompt);

        var memory = transformer_encoder.forward(
            img_feats, img_pos_embeds, prompt, prompt_pos_embed,
            prompt_mask, vis_feat_sizes);

        return memory;
    }

    /// <summary>
    /// Run transformer decoder.
    /// </summary>
    public Tuple<Dictionary<string, Tensor>, Tensor> run_decoder(
        Tensor pos_embed,
        Tensor memory,
        Tensor? src_mask,
        Tensor prompt,
        Tensor? prompt_mask)
    {
        var query_embed = transformer_decoder.get_query_embed();
        var bs = memory.size(1);
        var num_queries = query_embed.size(0);
        var d_model = query_embed.size(1);

        var tgt = query_embed.unsqueeze(1).repeat(1, bs, 1);

        var decoder_result = transformer_decoder.forward(
            tgt, memory, pos_embed, src_mask,
            prompt, prompt_mask);
        var hs = decoder_result.Item1;
        var reference_boxes = decoder_result.Item2;

        var result_dict = new Dictionary<string, Tensor>();

        // Score prediction via dot-product
        result_dict["pred_logits"] = predict_scores(hs[-1], prompt);

        // Box prediction
        var box_head = new Sam3MLP(d_model, 256, 4, 3);
        var reference_boxes_inv_sig = log(reference_boxes / (1 - reference_boxes + 0.000001f) + 0.000001f);
        var anchor_box_offsets = box_head.forward(hs[-1]);
        var outputs_coord = (reference_boxes_inv_sig + anchor_box_offsets).sigmoid();
        result_dict["pred_boxes"] = outputs_coord;

        return Tuple.Create(result_dict, hs[-1]);
    }

    private Tensor predict_scores(Tensor hs, Tensor prompt)
    {
        // Dot-product scoring: [B, Nq, D] x [Np, B, D] -> [B, Nq, Np]
        var scores = matmul(hs, prompt.permute(new long[] { 1, 0, 2 }));
        return scores;
    }
}
