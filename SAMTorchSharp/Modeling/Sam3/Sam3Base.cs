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

    private Device? _device_cache = null;

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
        this.segmentation_head = (Sam3SegmentationHead?)segmentation_head;
        this.hidden_dim = transformer_decoder.d_model;
        this.num_feature_levels = num_feature_levels;
        this.multimask_output = multimask_output;
    }

    public Device device
    {
        get
        {
            if (_device_cache is null)
            {
                var p = parameters().FirstOrDefault();
                _device_cache = p?.device;
            }
            return _device_cache ?? CPU;
        }
    }

    public Dictionary<string, Tensor> forward_backbone(Tensor images)
    {
        var backboneOut = backbone.forward(images);
        var dict = new Dictionary<string, Tensor>();
        dict.Add("backbone_out", backboneOut.Item1[backboneOut.Item1.Count - 1]);
        return dict;
    }

    public Tuple<Tensor, Tensor?> encode_prompts(
        IList<Tensor> img_feats,
        IList<Tensor> img_pos_embeds,
        IList<long[]> vis_feat_sizes,
        Sam3Prompt geometric_prompt)
    {
        var (geo_feats, geo_masks) = geometry_encoder.forward(
            geometric_prompt, img_feats.ToList(), vis_feat_sizes.ToList(), img_pos_embeds.ToList());

        var prompt = cat(geo_feats.ToArray(), dim: 0);
        var validMasks = geo_masks.Where(m => m is not null).Cast<Tensor>().ToList();
        Tensor? prompt_mask = validMasks.Count > 0 ? cat(validMasks.ToArray(), dim: 1) : null;

        return Tuple.Create(prompt, prompt_mask);
    }

    public Dictionary<string, object> run_encoder(
        IList<Tensor> img_feats,
        IList<Tensor> img_pos_embeds,
        Tensor prompt,
        Tensor? prompt_mask)
    {
        var vis_feat_sizes = img_feats.Select(f => Tuple.Create((int)f.size(2), (int)f.size(3))).ToList();

        var memory = transformer_encoder.forward(
            img_feats.ToList(),
            prompt_mask is not null ? new List<Tensor> { prompt_mask } : null,
            img_pos_embeds.ToList());

        return memory;
    }

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
        var d_model = (int)query_embed.size(1);

        var tgt = query_embed.unsqueeze(1).repeat(1, bs, 1);

        var decoderResult = transformer_decoder.forward(
            tgt, memory, pos_embed, src_mask,
            prompt, prompt_mask);
        var hs = decoderResult.Item1;
        var reference_boxes = decoderResult.Item2;

        var result_dict = new Dictionary<string, Tensor>();
        var lastLayerIdx = (int)hs.size(0) - 1;

        result_dict.Add("pred_logits", predict_scores(hs[lastLayerIdx], prompt));

        var box_head = new Sam3MLP(d_model, 256, 4, 3);
        var reference_boxes_inv_sig = log(reference_boxes / (1 - reference_boxes + 0.000001f) + 0.000001f);
        var anchor_box_offsets = box_head.forward(hs[lastLayerIdx]);
        var outputs_coord = (reference_boxes_inv_sig + anchor_box_offsets).sigmoid();
        result_dict.Add("pred_boxes", outputs_coord);

        return Tuple.Create(result_dict, hs[lastLayerIdx]);
    }

    private Tensor predict_scores(Tensor hs, Tensor prompt)
    {
        var scores = matmul(hs, prompt.permute(new long[] { 1, 0, 2 }));
        return scores;
    }
}
