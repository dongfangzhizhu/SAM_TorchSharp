// Copyright (c) Sapiens AI. All rights reserved.

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System;
using System.Collections.Generic;

namespace SAMTorchSharp.Modeling.Sam3;

/// <summary>
/// LayerNorm2d for 2D normalization.
/// </summary>
public class Sam3LayerNorm2d : Module<Tensor, Tensor>
{
    private readonly LayerNorm norm;

    public Sam3LayerNorm2d(int normalized_shape)
        : base(nameof(Sam3LayerNorm2d))
    {
        norm = LayerNorm(normalized_shape);
    }

    public override Tensor forward(Tensor x)
    {
        var x_perm = x.permute(new long[] { 0, 2, 3, 1 });
        var x_norm = norm.forward(x_perm);
        return x_norm.permute(new long[] { 0, 3, 1, 2 });
    }
}

/// <summary>
/// Multiplex Mask Decoder for SAM3.
/// </summary>
public class Sam3MultiplexMaskDecoder : Module
{
    private readonly int transformer_dim;
    private readonly Module transformer;
    private readonly int multiplex_count;
    private readonly int num_multimask_outputs;
    private readonly int num_mask_output_per_object;
    private readonly int num_mask_tokens;

    private readonly Embedding mask_tokens;
    private readonly Module output_upscaling;
    private readonly ModuleList<Sam3MLP> output_hypernetworks_mlps;
    private readonly Sam3MLP iou_prediction_head;

    private readonly bool use_high_res_features;
    private readonly Conv2d? conv_s0;
    private readonly Conv2d? conv_s1;

    public Sam3MultiplexMaskDecoder(
        int transformer_dim,
        Module transformer,
        int multiplex_count = 1,
        int num_multimask_outputs = 3,
        bool use_high_res_features = false)
        : base(nameof(Sam3MultiplexMaskDecoder))
    {
        this.transformer_dim = transformer_dim;
        this.transformer = transformer;
        this.multiplex_count = multiplex_count;
        this.num_multimask_outputs = num_multimask_outputs;
        this.use_high_res_features = use_high_res_features;

        num_mask_output_per_object = num_multimask_outputs + 1;
        num_mask_tokens = multiplex_count * num_mask_output_per_object;

        mask_tokens = Embedding(num_mask_tokens, transformer_dim);

        var ct2_0 = ConvTranspose2d(transformer_dim, transformer_dim / 4, kernelSize: 2, stride: 2);
        var ln2d = new Sam3LayerNorm2d(transformer_dim / 4);
        var gelu1 = GELU();
        var ct2_1 = ConvTranspose2d(transformer_dim / 4, transformer_dim / 8, kernelSize: 2, stride: 2);
        var gelu2 = GELU();
        output_upscaling = Sequential(ct2_0, ln2d, gelu1, ct2_1, gelu2);

        if (use_high_res_features)
        {
            conv_s0 = Conv2d(transformer_dim, transformer_dim / 8, kernelSize: 1);
            conv_s1 = Conv2d(transformer_dim, transformer_dim / 4, kernelSize: 1);
        }

        output_hypernetworks_mlps = new ModuleList<Sam3MLP>();
        for (int i = 0; i < num_mask_output_per_object; i++)
            output_hypernetworks_mlps.Add(new Sam3MLP(transformer_dim, transformer_dim, transformer_dim / 8, 3));

        iou_prediction_head = new Sam3MLP(transformer_dim, 256, num_mask_output_per_object, 3, sigmoid_output: false);
    }

    public Tuple<IList<Tensor>, Tensor, Tensor> forward(
        Tensor sparse_embeddings,
        Tensor dense_embeddings,
        IList<Tensor> low_res_multimasks)
    {
        var B = sparse_embeddings.size(1);
        var N = sparse_embeddings.size(0);

        var ms = predict_masks(sparse_embeddings, dense_embeddings, (int)B, (int)N);
        var iou_preds = predict_iou(ms, (int)B);

        var res_multimasks = new List<Tensor>();
        foreach (var low_res_multimask in low_res_multimasks)
        {
            var multimask_output = cat(new[] { low_res_multimask, ms }, dim: 0);
            res_multimasks.Add(multimask_output);
        }

        return Tuple.Create<IList<Tensor>, Tensor, Tensor>(res_multimasks, iou_preds, ms);
    }

    private Tensor predict_masks(Tensor txt_embs, Tensor obj_embs, int B, int N)
    {
        var hyper_inp = txt_embs.max(dim: 0).values;
        var hyper_net_weights = output_hypernetworks_mlps[0].forward(hyper_inp);

        var obj_feat = obj_embs.permute(new long[] { 0, 3, 1, 2 });
        var obj_feat_scaled = ((Module<Tensor, Tensor>)output_upscaling).forward(obj_feat);

        var low_res_masks = obj_feat_scaled.view(new long[] { B, -1, obj_feat_scaled.size(2), obj_feat_scaled.size(3) });

        return low_res_masks;
    }

    private Tensor predict_iou(Tensor masks, int B)
    {
        var mean_vals = masks.mean(new long[] { 2, 3 }, keepdim: true);
        var inp = iou_prediction_head.forward(mean_vals.unsqueeze(1));
        return inp;
    }
}

/// <summary>
/// Segmentation head (MaskFormer-style) for SAM3.
/// </summary>
public class Sam3SegmentationHead : Module
{
    private readonly Sam3MultiplexMaskDecoder mask_decoder;
    private readonly int num_queries;
    private readonly int hidden_dim;
    private readonly bool o2m_mask_predict;

    public Sam3SegmentationHead(
        int transformer_dim,
        Module transformer,
        int num_queries = 900,
        int multiplex_count = 1,
        int num_multimask_outputs = 3,
        bool o2m_mask_predict = true)
        : base(nameof(Sam3SegmentationHead))
    {
        this.num_queries = num_queries;
        this.hidden_dim = transformer_dim;
        this.o2m_mask_predict = o2m_mask_predict;

        mask_decoder = new Sam3MultiplexMaskDecoder(
            transformer_dim, transformer,
            multiplex_count: multiplex_count,
            num_multimask_outputs: num_multimask_outputs);
    }

    public Dictionary<string, Tensor> forward(
        IList<Tensor> backbone_feats,
        Tensor obj_queries,
        Tensor? encoder_hidden_states,
        Tensor? prompt,
        Tensor? prompt_mask)
    {
        var outputs = new Dictionary<string, Tensor>();

        var queries = obj_queries;

        var (multimasks, iou_preds, masks) = mask_decoder.forward(
            prompt!,
            queries,
            new List<Tensor>()
        );

        outputs["pred_masks"] = masks;
        outputs["pred_logits"] = iou_preds;

        return outputs;
    }
}
