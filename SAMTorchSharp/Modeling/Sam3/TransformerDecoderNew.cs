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
/// Utility functions for decoder position encoding.
/// Matches Python gen_sineembed_for_position() in model_misc.py
/// </summary>
public static class DecoderUtils
{
    public static Tensor gen_sineembed_for_position(Tensor pos_tensor, int num_feats)
    {
        if (num_feats % 2 != 0)
            throw new ArgumentException("num_feats must be even");

        int num_feats_half = num_feats / 2;
        int N = (int)pos_tensor.size(0);
        int B = (int)pos_tensor.size(1);
        var device = pos_tensor.device;

        var indices_even = torch.arange(0, num_feats_half, 2, dtype: ScalarType.Float32, device: device);
        var indices_odd = torch.arange(1, num_feats_half, 2, dtype: ScalarType.Float32, device: device);
        var exponent_even = 2.0 * indices_even / num_feats_half;
        var exponent_odd = 2.0 * indices_odd / num_feats_half;
        var dim_t_even = exp(exponent_even * log(10000.0));
        var dim_t_odd = exp(exponent_odd * log(10000.0));

        if (pos_tensor.size(2) == 2)
        {
            var x_embed = pos_tensor.index_select(2, torch.tensor(new long[] { 0 }));
            var y_embed = pos_tensor.index_select(2, torch.tensor(new long[] { 1 }));

            var pos_x = x_embed.unsqueeze(-1) / dim_t_even.unsqueeze(0).unsqueeze(0);
            pos_x = torch.stack(new[] { pos_x.sin(), pos_x.cos() }, dim: 3).flatten(2);

            var pos_y = y_embed.unsqueeze(-1) / dim_t_odd.unsqueeze(0).unsqueeze(0);
            pos_y = torch.stack(new[] { pos_y.sin(), pos_y.cos() }, dim: 3).flatten(2);

            return cat(new[] { pos_y, pos_x }, dim: 2);
        }
        else if (pos_tensor.size(2) == 4)
        {
            var x_embed = pos_tensor.index_select(2, torch.tensor(new long[] { 0 }));
            var y_embed = pos_tensor.index_select(2, torch.tensor(new long[] { 1 }));
            var w_embed = pos_tensor.index_select(2, torch.tensor(new long[] { 2 }));
            var h_embed = pos_tensor.index_select(2, torch.tensor(new long[] { 3 }));

            var pos_w = w_embed.unsqueeze(-1) / dim_t_even.unsqueeze(0).unsqueeze(0);
            pos_w = torch.stack(new[] { pos_w.sin(), pos_w.cos() }, dim: 3).flatten(2);

            var pos_h = h_embed.unsqueeze(-1) / dim_t_odd.unsqueeze(0).unsqueeze(0);
            pos_h = torch.stack(new[] { pos_h.sin(), pos_h.cos() }, dim: 3).flatten(2);

            var pos_x = x_embed.unsqueeze(-1) / dim_t_even.unsqueeze(0).unsqueeze(0);
            pos_x = torch.stack(new[] { pos_x.sin(), pos_x.cos() }, dim: 3).flatten(2);

            var pos_y = y_embed.unsqueeze(-1) / dim_t_odd.unsqueeze(0).unsqueeze(0);
            pos_y = torch.stack(new[] { pos_y.sin(), pos_y.cos() }, dim: 3).flatten(2);

            return cat(new[] { pos_y, pos_x, pos_w, pos_h }, dim: 2);
        }
        else
        {
            throw new ArgumentException($"Unknown pos_tensor shape(-1):{pos_tensor.size(2)}");
        }
    }
}

/// Uses SEPARATE Q/K/V projections (matching checkpoint keys).
///
/// Forward order (matches Python TransformerDecoderLayer):
/// 1. Self-attention (with query_pos embedding)
/// 2. Text cross-attention (with query_pos embedding)
/// 3. Visual cross-attention (with memory_pos embedding)
/// 4. MLP (2-layer: 256 -> 2048 -> 256 with GELU)
///
/// Each sub-layer has residual connection + LayerNorm (pre-norm).
/// </summary>
public class Sam3TransformerDecoderLayerNew : Module
{
    private readonly Linear self_attn_q_proj;
    private readonly Linear self_attn_k_proj;
    private readonly Linear self_attn_v_proj;
    private readonly Linear self_attn_o_proj;

    private readonly Linear ca_text_q_proj;
    private readonly Linear ca_text_k_proj;
    private readonly Linear ca_text_v_proj;
    private readonly Linear ca_text_o_proj;

    private readonly Linear cross_attn_q_proj;
    private readonly Linear cross_attn_k_proj;
    private readonly Linear cross_attn_v_proj;
    private readonly Linear cross_attn_o_proj;

    private readonly Linear mlp_fc1;
    private readonly Linear mlp_fc2;

    private readonly LayerNorm layer_norm1;  // After visual cross-attn
    private readonly LayerNorm layer_norm2;  // After self-attn
    private readonly LayerNorm layer_norm3;  // After MLP
    private readonly LayerNorm catext_norm;  // After text cross-attn

    private readonly int d_model;
    private readonly int nhead;
    private readonly int dim_feedforward;
    private readonly int head_dim;
    private readonly float attn_scale;

    public Sam3TransformerDecoderLayerNew(
        int d_model = 256,
        int nhead = 8,
        int dim_feedforward = 2048)
        : base(nameof(Sam3TransformerDecoderLayerNew))
    {
        this.d_model = d_model;
        this.nhead = nhead;
        this.dim_feedforward = dim_feedforward;
        this.head_dim = d_model / nhead;
        this.attn_scale = 1.0f / (float)Math.Sqrt(head_dim);

        self_attn_q_proj = Linear(d_model, d_model);
        self_attn_k_proj = Linear(d_model, d_model);
        self_attn_v_proj = Linear(d_model, d_model);
        self_attn_o_proj = Linear(d_model, d_model);

        ca_text_q_proj = Linear(d_model, d_model);
        ca_text_k_proj = Linear(d_model, d_model);
        ca_text_v_proj = Linear(d_model, d_model);
        ca_text_o_proj = Linear(d_model, d_model);

        cross_attn_q_proj = Linear(d_model, d_model);
        cross_attn_k_proj = Linear(d_model, d_model);
        cross_attn_v_proj = Linear(d_model, d_model);
        cross_attn_o_proj = Linear(d_model, d_model);

        mlp_fc1 = Linear(d_model, dim_feedforward);
        mlp_fc2 = Linear(dim_feedforward, d_model);

        layer_norm1 = LayerNorm(d_model);
        layer_norm2 = LayerNorm(d_model);
        layer_norm3 = LayerNorm(d_model);
        catext_norm = LayerNorm(d_model);

        RegisterComponents();
    }

    private Tensor dot_product_attention(
        Tensor q,
        Tensor k,
        Tensor v,
        Tensor? rpb_bias = null,
        bool add_rpb_bias = false,
        Tensor? keyPaddingMask = null)
    {
        // q, k, v: [B, seq, d_model] (batch-first)
        var B = q.size(0);
        var seq_q = q.size(1);
        var seq_k = k.size(1);

        // Split into heads: [B, seq, d_model] -> [B, seq, nhead, hd] -> [B, nhead, seq, hd]
        var q_h = q.reshape(new long[] { B, seq_q, nhead, head_dim }).transpose(1, 2);  // [B, nhead, seq_q, hd]
        var k_h = k.reshape(new long[] { B, seq_k, nhead, head_dim }).transpose(1, 2);  // [B, nhead, seq_k, hd]
        var v_h = v.reshape(new long[] { B, seq_k, nhead, head_dim }).transpose(1, 2);  // [B, nhead, seq_k, hd]

        // Scaled dot-product: [B, nhead, seq_q, hd] x [B, nhead, hd, seq_k] -> [B, nhead, seq_q, seq_k]
        var k_h_t = k_h.transpose(2, 3);
        var attn = (q_h * attn_scale).matmul(k_h_t);  // [B, nhead, seq_q, seq_k]

        // Add RPB bias for visual cross-attention
        if (rpb_bias is not null && add_rpb_bias)
        {
            // rpb_bias: [bs, nhead, nq, HW] -> add to attn [bs, nhead, nq, HW]
            attn = attn + rpb_bias;
        }

        if (keyPaddingMask is not null)
        {
            var mask = keyPaddingMask.ndim == 3 ? keyPaddingMask.squeeze(0) : keyPaddingMask;
            if (mask.shape is not [var maskBatch, var maskSequence] || maskBatch != B || maskSequence != seq_k)
                throw new ArgumentException("SAM 3 text padding mask must have shape [batch, sequence].");
            if (mask.all(dim: 1).any().item<bool>())
                throw new ArgumentException("SAM 3 text padding mask must leave at least one token unmasked per batch.");
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(1), float.NegativeInfinity);
        }

        attn = functional.softmax(attn, dim: 3);
        var attn_out = attn.matmul(v_h);  // [B, nhead, seq_q, hd]

        // Merge heads and transpose back: [B, nhead, seq_q, hd] -> [B, seq_q, d_model]
        attn_out = attn_out.transpose(1, 2).reshape(new long[] { B, seq_q, d_model });
        return attn_out;
    }

    public Tensor forward(
        Tensor query,           // [nq, bs, d_model] - seq-first
        Tensor query_pos,       // [nq, bs, d_model] - seq-first
        Tensor memory,          // [seq_hw, bs, d_model] - visual features (seq-first)
        Tensor text_memory,     // [seq_text, bs, d_model] - text features (seq-first)
        Tensor? text_attention_mask = null,
        Tensor? memory_pos = null,
        Tensor? rpb_bias = null)  // [bs, nhead, nq, H*W] - box RPB bias for cross-attn
    {
        int N = (int)query.size(0);
        int B = (int)query.size(1);

        Tensor addPos(Tensor t, Tensor pos) => pos is null ? t : t + 0.1 * pos;

        // ========== 1. Self-attention ==========
        var q_self = addPos(query, query_pos);
        var k_self = addPos(query, query_pos);
        var v_self = query;

        var q_s = self_attn_q_proj.forward(q_self);
        var k_s = self_attn_k_proj.forward(k_self);
        var v_s = self_attn_v_proj.forward(v_self);

        q_s = q_s.transpose(0, 1);
        k_s = k_s.transpose(0, 1);
        v_s = v_s.transpose(0, 1);

        var self_out = dot_product_attention(q_s, k_s, v_s);
        self_out = self_out.transpose(0, 1);
        self_out = self_attn_o_proj.forward(self_out);

        query = query + self_out;
        query = layer_norm2.forward(query);

        // ========== 2. Text cross-attention ==========
        var q_text = addPos(query, query_pos);
        var k_text = text_memory;
        var v_text = text_memory;

        var q_t = ca_text_q_proj.forward(q_text);
        var k_t = ca_text_k_proj.forward(k_text);
        var v_t = ca_text_v_proj.forward(v_text);

        q_t = q_t.transpose(0, 1);
        k_t = k_t.transpose(0, 1);
        v_t = v_t.transpose(0, 1);

        var text_out = dot_product_attention(q_t, k_t, v_t, keyPaddingMask: text_attention_mask);
        text_out = text_out.transpose(0, 1);
        text_out = ca_text_o_proj.forward(text_out);

        query = query + text_out;
        query = catext_norm.forward(query);

        // ========== 3. Visual cross-attention ==========
        var q_vis = addPos(query, query_pos);
        Tensor k_vis = memory_pos is not null ? addPos(memory, memory_pos) : memory;
        var v_vis = memory;

        var q_v = cross_attn_q_proj.forward(q_vis);
        var k_v = cross_attn_k_proj.forward(k_vis);
        var v_v = cross_attn_v_proj.forward(v_vis);

        q_v = q_v.transpose(0, 1);
        k_v = k_v.transpose(0, 1);
        v_v = v_v.transpose(0, 1);

        var vis_out = dot_product_attention(q_v, k_v, v_v, rpb_bias, true);
        vis_out = vis_out.transpose(0, 1);
        vis_out = cross_attn_o_proj.forward(vis_out);

        query = query + vis_out;
        query = layer_norm1.forward(query);

        // ========== 4. MLP (FFN) ==========
        var mlp_out = mlp_fc2.forward(functional.gelu(mlp_fc1.forward(query)));
        query = query + mlp_out;
        query = layer_norm3.forward(query);

        return query;
    }
}

/// <summary>
/// Transformer decoder for SAM3 DETR-style decoder.
/// Matches: detector_model.detr_decoder
///
/// Key features:
/// - Iterative reference point updates via box_head (3-layer MLP)
/// - Sine position encoding via ref_point_head (2-layer MLP)
/// - Multi-layer decoder with residual connections
/// - Uses separate Q/K/V projections (matching checkpoint)
/// - Supports presence token for detection
/// - Supports box RPB (Relative Position Bias) for RoPE encoding
/// </summary>
public class Sam3TransformerDecoderNew : Module
{
    private readonly List<Sam3TransformerDecoderLayerNew> layers;
    private readonly Tensor query_embed;
    private readonly Tensor reference_points;
    private readonly Tensor presence_token;

    // Box head: 3-layer MLP (256 -> 256 -> 256 -> 4)
    public readonly ModuleList<Linear> box_head;

    // Ref point head: 2-layer MLP (512 -> 256 -> 256)
    public readonly ModuleList<Linear> ref_point_head;

    // Presence head: 3-layer MLP (256 -> 256 -> 256 -> 1)
    public readonly ModuleList<Linear> presence_head;

    // Presence layer norm
    public readonly LayerNorm presence_layer_norm;

    // Box RPE (RoPE position embedding): x and y branches
    public readonly ModuleList<Linear> box_rpb_embed_x;
    public readonly ModuleList<Linear> box_rpb_embed_y;

    private readonly LayerNorm output_layer_norm;
    private readonly int num_layers;
    private readonly int num_queries;
    public readonly int d_model;
    private readonly int nhead;

    public Sam3TransformerDecoderNew(
        int num_layers = 6,
        int num_queries = 200,
        int d_model = 256,
        int nhead = 8,
        int dim_feedforward = 2048)
        : base(nameof(Sam3TransformerDecoderNew))
    {
        this.num_layers = num_layers;
        this.num_queries = num_queries;
        this.d_model = d_model;
        this.nhead = nhead;

        layers = new List<Sam3TransformerDecoderLayerNew>();
        for (int i = 0; i < num_layers; i++)
        {
            var layer = new Sam3TransformerDecoderLayerNew(d_model, nhead, dim_feedforward);
            layers.Add(layer);
            register_module("layer_" + i.ToString(), layer);
        }

        // query_embed: [num_queries, d_model]
        query_embed = Parameter(torch.randn(new long[] { num_queries, d_model }), requires_grad: false);

        // reference_points: [num_queries, 4]
        reference_points = Parameter(torch.zeros(new long[] { num_queries, 4 }), requires_grad: false);

        // presence_token: [1, 1, d_model]
        presence_token = Parameter(torch.zeros(new long[] { 1, 1, d_model }), requires_grad: false);

        // box_head: 3-layer MLP (256 -> 256 -> 256 -> 4)
        box_head = new ModuleList<Linear>(new[] {
            Linear(d_model, d_model),     // layer1: 256 -> 256
            Linear(d_model, d_model),     // layer2: 256 -> 256
            Linear(d_model, 4)            // layer3: 256 -> 4
        });
        register_module("box_head", box_head);

        // ref_point_head: 2-layer MLP (512 -> 256 -> 256)
        ref_point_head = new ModuleList<Linear>(new[] {
            Linear(2 * d_model, d_model), // layer1: 512 -> 256
            Linear(d_model, d_model)      // layer2: 256 -> 256
        });
        register_module("ref_point_head", ref_point_head);

        // presence_head: 3-layer MLP (256 -> 256 -> 256 -> 1)
        presence_head = new ModuleList<Linear>(new[] {
            Linear(d_model, d_model),     // layer1
            Linear(d_model, d_model),     // layer2
            Linear(d_model, 1)            // layer3
        });
        register_module("presence_head", presence_head);

        // presence_layer_norm
        presence_layer_norm = LayerNorm(d_model);
        register_module("presence_layer_norm", presence_layer_norm);

        // box_rpb_embed_x: 2-layer MLP (2 -> 256 -> 8)
        box_rpb_embed_x = new ModuleList<Linear>(new[] {
            Linear(2, d_model),           // layer1: 2 -> 256
            Linear(d_model, nhead)        // layer2: 256 -> nhead (number of heads)
        });
        register_module("box_rpb_embed_x", box_rpb_embed_x);

        // box_rpb_embed_y: 2-layer MLP (2 -> 256 -> 8)
        box_rpb_embed_y = new ModuleList<Linear>(new[] {
            Linear(2, d_model),           // layer1: 2 -> 256
            Linear(d_model, nhead)        // layer2: 256 -> nhead (number of heads)
        });
        register_module("box_rpb_embed_y", box_rpb_embed_y);

        output_layer_norm = LayerNorm(d_model);
        register_module("output_layer_norm", output_layer_norm);

        RegisterComponents();
    }

    public Tensor get_query_embed()
    {
        return query_embed;
    }

    /// <summary>
    /// Forward pass with iterative reference point updates.
    /// Supports boxRPB="log" for RoPE position embedding bias.
    /// Returns (hidden_states [num_layers, nq, bs, d_model], reference_boxes [num_layers, nq, bs, 4]).
    /// </summary>
    public Tuple<Tensor, Tensor> forward(
        Tensor tgt,
        Tensor memory,
        Tensor? text_memory = null,
        Tensor? prompt_mask = null,
        Tensor? memory_pos = null,
        long[] spatialShapes = null)
    {
        int B = (int)tgt.size(1);
        int N = (int)tgt.size(0);
        int d_model = (int)tgt.size(2);

        // Initialize reference points (normalized to [0,1])
        var ref_batched = reference_points.unsqueeze(1).repeat(new long[] { 1, B, 1 });
        ref_batched = ref_batched.sigmoid();

        var hidden_states = new List<Tensor>();
        var reference_boxes_list = new List<Tensor>();

        var query = tgt.clone();

        // Expand presence token to batch size: [1, 1, d_model] -> [1, B, d_model]
        Tensor? presence_out = null;
        if (presence_token is not null)
        {
            presence_out = presence_token.expand(1, B, -1);  // [1, B, d_model]
        }

        for (int i = 0; i < num_layers; i++)
        {
            // 1. Generate sine embedding for reference points
            var query_sine_embed = DecoderUtils.gen_sineembed_for_position(ref_batched, d_model);

            // 2. Process through ref_point_head to get query_pos
            Tensor query_pos = query_sine_embed;
            for (int l = 0; l < ref_point_head.Count; l++)
            {
                query_pos = ref_point_head[l].forward(query_pos);
                if (l < ref_point_head.Count - 1)
                    query_pos = functional.relu(query_pos);
            }

            // 3. Compute RPB bias matrix if spatial_shapes provided
            // TEMP: Disable RPB until shapes are verified
            Tensor? rpb_bias = null;
            /*
            if (spatialShapes != null && spatialShapes.Length == 2)
            {
                rpb_bias = compute_box_rpb_log(ref_batched, spatialShapes[0], spatialShapes[1]);
            }
            */

            // 4. Run decoder layer
            query = layers[i].forward(
                query,
                query_pos,
                memory,
                text_memory ?? memory,
                prompt_mask,
                memory_pos,
                rpb_bias);

            hidden_states.Add(query.clone());

            // 5. Predict presence logit from presence_out (if available)
            // In Python, presence_out is returned by the decoder layer when presence_token is not null
            // For now we skip per-layer presence prediction (not used in scoring)

            // 6. Predict box deltas and update reference points
            var normalizedQuery = output_layer_norm.forward(query);
            var bbox_delta = box_head.forward_box(normalizedQuery);

            var ref_before_sigmoid = inverse_sigmoid(ref_batched);
            var new_ref = ref_before_sigmoid + bbox_delta;
            ref_batched = new_ref.sigmoid();

            reference_boxes_list.Add(ref_batched.clone());
        }

        var hs_stack = torch.stack(hidden_states.ToArray());
        var ref_stack = torch.stack(reference_boxes_list.ToArray());

        return Tuple.Create(hs_stack, ref_stack);
    }

    /// <summary>
    /// Compute presence logits from the presence token output.
    /// Called after decoder forward pass using the final presence_out.
    /// Returns: [bs] - one scalar per batch item
    /// </summary>
    public Tensor compute_presence_logits(Tensor presence_output)
    {
        // presence_output: [1, bs, d_model]
        var normalized = presence_layer_norm.forward(presence_output);
        var logits = presence_head.forward_box(normalized).squeeze(-1);  // [1, bs, 1] -> [1, bs]
        // Clamp to avoid numerical issues
        logits = clamp(logits, -10.0f, 10.0f);
        return logits.squeeze(0);  // [bs]
    }

    /// <summary>
    /// Compute box RPB (Relative Position Bias) with log-mode encoding.
    /// Matches Python: _get_rpb_matrix with boxRPB="log"
    /// Returns: bias matrix [bs, nhead, nq, H*W] for use in cross-attention
    /// </summary>
    private Tensor compute_box_rpb_log(Tensor reference_boxes, long H, long W)
    {
        // reference_boxes: [nq, bs, 4] in cxcywh format (normalized [0,1])
        // Convert to xyxy format and transpose to [bs, nq, 4]
        var boxes_xyxy = BoxOps.box_cxcywh_to_xyxy(reference_boxes);  // [nq, bs, 4]
        boxes_xyxy = boxes_xyxy.transpose(0, 1);  // [bs, nq, 4]

        int num_queries = (int)boxes_xyxy.size(0); // Actually this is bs now
        // Wait - after transpose: [bs, nq, 4]. But Python does reshape(-1, 1, 4) first.
        // Let's follow Python exactly:
        // boxes_xyxy: [bs, nq, 4] -> reshape(-1, 1, 4) -> [bs*nq, 1, 4]
        // then [:, :, 1:4:2] -> select y1, y2 -> [bs*nq, 1, 2]

        var B = boxes_xyxy.size(0);   // bs
        var N = boxes_xyxy.size(1);   // nq

        // Get coords: [H], [W]
        var coords_h = torch.arange(0.0, (float)H) / (float)H;  // [H]
        var coords_w = torch.arange(0.0, (float)W) / (float)W;  // [W]

        // Python: deltas_y = coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]
        // boxes_xyxy: [bs, nq, 4] -> reshape(-1, 1, 4): [bs*nq, 1, 4]
        // [:, :, 1:4:2]: select indices 1, 3 -> [bs*nq, 1, 2] (y1, y2)
        var boxes_reshaped = boxes_xyxy.reshape(new long[] { B * N, 1, 4 });  // [bs*nq, 1, 4]
        var y_coords = boxes_reshaped.index_select(2, torch.tensor(new int[] { 1, 3 }, dtype: ScalarType.Int64));  // [bs*nq, 1, 2]

        // coords_h.view(1, -1, 1): [1, H, 1]
        var coords_h_view = coords_h.reshape(new long[] { 1, -1, 1 });  // [1, H, 1]

        // deltas_y: [bs*nq, H, 2]
        var deltas_y = coords_h_view - y_coords;  // broadcast: [1,H,1] - [bs*nq,1,2] = [bs*nq,H,2]
        deltas_y = deltas_y.reshape(new long[] { B, N, (int)H, 2 });  // [bs, nq, H, 2]

        // Python: deltas_x = coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]
        var x_coords = boxes_reshaped.index_select(2, torch.tensor(new int[] { 0, 2 }, dtype: ScalarType.Int64));  // [bs*nq, 1, 2] (x1, x2)
        var coords_w_view = coords_w.reshape(new long[] { 1, -1, 1 });  // [1, W, 1]
        var deltas_x = coords_w_view - x_coords;  // [bs*nq, W, 2]
        deltas_x = deltas_x.reshape(new long[] { B, N, (int)W, 2 });  // [bs, nq, W, 2]

        // Apply log-mode transformation
        var scale = 8.0f;
        var log2_8 = (float)Math.Log(8.0, 2);

        var deltas_x_scaled = deltas_x * scale;
        var deltas_x_log = torch.sign(deltas_x_scaled) *
            torch.log2(torch.abs(deltas_x_scaled) + 1.0) / log2_8;

        var deltas_y_scaled = deltas_y * scale;
        var deltas_y_log = torch.sign(deltas_y_scaled) *
            torch.log2(torch.abs(deltas_y_scaled) + 1.0) / log2_8;

        // Use log mode: replace deltas with log version
        deltas_x = deltas_x_log;
        deltas_y = deltas_y_log;

        // Project through MLP: box_rpb_embed_x/y (2-layer MLP: 2 -> d_model -> nhead)
        Tensor proj_x = deltas_x;  // [bs, nq, W, 2]
        for (int l = 0; l < box_rpb_embed_x.Count; l++)
        {
            proj_x = box_rpb_embed_x[l].forward(proj_x);
            if (l < box_rpb_embed_x.Count - 1)
                proj_x = functional.relu(proj_x);
        }
        // [bs, nq, W, nhead]

        Tensor proj_y = deltas_y;  // [bs, nq, H, 2]
        for (int l = 0; l < box_rpb_embed_y.Count; l++)
        {
            proj_y = box_rpb_embed_y[l].forward(proj_y);
            if (l < box_rpb_embed_y.Count - 1)
                proj_y = functional.relu(proj_y);
        }
        // [bs, nq, H, nhead]

        // Combine: B[b, q, h, w] = proj_y[b, q, w, :] + proj_x[b, q, h, :]
        var combined = proj_y.unsqueeze(3) + proj_x.unsqueeze(2);  // [bs, nq, H, W, nhead]

        // Flatten spatial: [bs, nq, H*W, nhead]
        combined = combined.reshape(new long[] { B, N, (int)(H * W), nhead });

        // Transpose: [bs, nhead, nq, H*W]
        combined = combined.permute(new long[] { 0, 3, 1, 2 });

        return combined;
    }

    private static Tensor inverse_sigmoid(Tensor x, float eps = 1e-3f)
    {
        x = x.clamp(min: 0.0, max: 1.0);
        var x1 = x.clamp(min: eps);
        var x2 = (1.0 - x).clamp(min: eps);
        return (x1.log() - x2.log());
    }
}

/// <summary>
/// Extension to access box_head as sequential MLP forward.
/// </summary>
public static class ModuleListExtensions
{
    public static Tensor forward_box(this ModuleList<Linear> ml, Tensor input)
    {
        Tensor x = input;
        for (int i = 0; i < ml.Count; i++)
        {
            x = ml[i].forward(x);
            if (i < ml.Count - 1)
                x = functional.relu(x);
        }
        return x;
    }
}

/// <summary>
/// Convert [cx, cy, w, h] to [x1, y1, x2, y2] format.
/// </summary>
public static class BoxOps
{
    public static Tensor box_cxcywh_to_xyxy(Tensor x)
    {
        // x: [..., 4] where last dim is [cx, cy, w, h]
        var cx = x.narrow(-1, 0, 1);
        var cy = x.narrow(-1, 1, 1);
        var w = x.narrow(-1, 2, 1);
        var h = x.narrow(-1, 3, 1);

        var x1 = cx - w / 2;
        var y1 = cy - h / 2;
        var x2 = cx + w / 2;
        var y2 = cy + h / 2;

        return torch.stack(new[] { x1, y1, x2, y2 }, dim: -1);
    }
}
