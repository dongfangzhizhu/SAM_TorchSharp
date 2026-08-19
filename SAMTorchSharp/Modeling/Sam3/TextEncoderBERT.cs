// Copyright (c) Sapiens AI. All rights reserved.

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System;
using System.Collections.Generic;

namespace SAMTorchSharp.Modeling.Sam3;

/// <summary>
/// BERT-style encoder layer for SAM3 text encoder.
/// Matches: detector_model.text_encoder.text_model.encoder.layers.{N}
/// Architecture: d_model=1024, hidden=4096, num_heads=16, 24 layers
/// Uses SEPARATE q_proj/k_proj/v_proj/o_proj (matching checkpoint keys).
/// </summary>
public class Sam3TextEncoderLayer : Module
{
    private readonly Linear self_attn_q_proj;
    private readonly Linear self_attn_k_proj;
    private readonly Linear self_attn_v_proj;
    private readonly Linear self_attn_o_proj;
    private readonly LayerNorm layer_norm1;
    private readonly LayerNorm layer_norm2;
    private readonly Linear fc1;
    private readonly Linear fc2;
    private readonly long d_model;
    private readonly int nhead;
    private readonly int head_dim;
    private readonly float attn_scale;

    public Sam3TextEncoderLayer(long d_model = 1024, int num_heads = 16)
        : base(nameof(Sam3TextEncoderLayer))
    {
        this.d_model = d_model;
        this.nhead = num_heads;
        this.head_dim = (int)(d_model / num_heads);
        this.attn_scale = 1.0f / (float)Math.Sqrt(head_dim);

        self_attn_q_proj = Linear(d_model, d_model);
        self_attn_k_proj = Linear(d_model, d_model);
        self_attn_v_proj = Linear(d_model, d_model);
        self_attn_o_proj = Linear(d_model, d_model);

        layer_norm1 = LayerNorm(d_model);
        layer_norm2 = LayerNorm(d_model);

        // MLP: 1024 -> 4096 -> 1024
        fc1 = Linear(d_model, 4096);
        fc2 = Linear(4096, d_model);

        RegisterComponents();
    }

    private Tensor dot_product_attention(Tensor q, Tensor k, Tensor v)
    {
        // q, k, v: [seq, batch, d_model]
        var seq = q.size(0);
        var B = q.size(1);

        var q_h = q.reshape(new long[] { seq, B, nhead, head_dim }).transpose(0, 1);  // [B, nhead, seq, hd]
        var k_h = k.reshape(new long[] { seq, B, nhead, head_dim }).transpose(0, 1);  // [B, nhead, seq, hd]
        var v_h = v.reshape(new long[] { seq, B, nhead, head_dim }).transpose(0, 1);  // [B, nhead, seq, hd]

        var k_h_t = k_h.transpose(2, 3);  // [B, nhead, hd, seq]
        var attn = (q_h * attn_scale).matmul(k_h_t);  // [B, nhead, seq, seq]
        attn = functional.softmax(attn, dim: 3);
        var attn_out = attn.matmul(v_h);  // [B, nhead, seq, hd]

        attn_out = attn_out.transpose(0, 1).reshape(new long[] { seq, B, d_model });
        return attn_out;
    }

    public Tensor forward(Tensor x)
    {
        // Self-attention with layer norm pre-norm
        var normed = layer_norm1.forward(x);
        var q = self_attn_q_proj.forward(normed);
        var k = self_attn_k_proj.forward(normed);
        var v = self_attn_v_proj.forward(normed);

        var attnResult = self_attn_o_proj.forward(dot_product_attention(q, k, v));
        x = x + attnResult;

        // MLP
        x = x + fc2.forward(functional.gelu(fc1.forward(layer_norm2.forward(x))));
        return x;
    }
}

/// <summary>
/// BERT-style text transformer for SAM3 (replaces CLIP-based Sam3TextTransformer).
/// Matches: detector_model.text_encoder.text_model
/// Architecture: token_embedding(49408, 1024) + pos_embed_field(32, 1024) + 24 encoder layers + final_layer_norm
/// </summary>
public class Sam3BERTTextTransformer : Module
{
    private readonly Embedding token_embedding;
    private readonly Parameter pos_embed_field;
    private readonly List<Sam3TextEncoderLayer> layers;
    private readonly LayerNorm final_layer_norm;
    private readonly int context_length;

    public Sam3BERTTextTransformer(
        int vocab_size = 49408,
        int d_model = 1024,
        int num_heads = 16,
        int num_layers = 24,
        int context_length = 32)
        : base(nameof(Sam3BERTTextTransformer))
    {
        this.context_length = context_length;

        token_embedding = Embedding(vocab_size, d_model);
        pos_embed_field = Parameter(zeros(new long[] { context_length, d_model }), requires_grad: false);
        register_buffer("pos_embed_buffer", pos_embed_field);

        layers = new List<Sam3TextEncoderLayer>();
        for (int i = 0; i < num_layers; i++)
        {
            layers.Add(new Sam3TextEncoderLayer(d_model, num_heads));
            register_module("encoder_layer_" + i.ToString(), layers[i]);
        }

        final_layer_norm = LayerNorm(d_model);

        RegisterComponents();
    }

    public Tensor forward(Tensor text)
    {
        var seq_len = text.size(1);
        var x = token_embedding.forward(text);
        var pos_embed = get_buffer("pos_embed_buffer");
        if (pos_embed is not null)
        {
            x = x + pos_embed.narrow(0, 0, seq_len).unsqueeze(0);
        }

        foreach (var layer in layers)
        {
            x = layer.forward(x);
        }

        x = final_layer_norm.forward(x);
        return x;
    }

    public Tuple<Tensor, Tensor> forward_with_pool(Tensor text)
    {
        var seq_len = text.size(1);
        var encoded = forward(text);
        var pooled = encoded.select(1, seq_len - 1);
        return Tuple.Create(pooled, encoded);
    }

    public int ContextLength => context_length;
}

/// <summary>
/// Full text encoder for SAM3 including projection.
/// Matches: detector_model.text_encoder + detector_model.text_projection
/// Architecture: BERT transformer (1024) -> Linear(1024, 256) -> final projection
/// </summary>
public class Sam3TextEncoder : Module
{
    private readonly Sam3BERTTextTransformer transformer;
    private readonly Linear text_projection;
    private readonly Sam3TokenizerVE tokenizer;
    private readonly int d_model;

    public Sam3TextEncoder(
        int d_model = 256,
        int text_width = 1024,
        int num_heads = 16,
        int num_layers = 24,
        int context_length = 32,
        int vocab_size = 49408,
        Sam3TokenizerVE tokenizer = null)
        : base(nameof(Sam3TextEncoder))
    {
        this.d_model = d_model;
        this.tokenizer = tokenizer;

        transformer = new Sam3BERTTextTransformer(vocab_size, text_width, num_heads, num_layers, context_length);
        text_projection = Linear(text_width, d_model);

        RegisterComponents();
    }

    /// <summary>
    /// Forward pass for text strings.
    /// Returns: (attention_mask, text_memory_resized, inputs_embeds)
    /// text_memory_resized: [seq_len, batch, d_model]
    /// </summary>
    public Tuple<Tensor, Tensor, Tensor> ForwardText(IList<string> texts, Device? device = null)
    {
        var dev = device ?? CPU;

        Tensor tokenized;
        if (tokenizer != null)
        {
            tokenized = tokenizer.EncodeTexts(texts).to(dev);
        }
        else
        {
            tokenized = zeros(new long[] { (long)texts.Count, (long)context_length }, dtype: int64, device: dev);
        }

        var (pooled, encoded) = transformer.forward_with_pool(tokenized);
        var projected = text_projection.forward(encoded); // [batch, seq_len, d_model]
        var text_memory_resized = projected.transpose(0, 1); // [seq_len, batch, d_model]
        var text_attention_mask = (tokenized != 0).logical_not();

        return Tuple.Create(text_attention_mask, text_memory_resized, projected);
    }

    private int context_length => transformer.ContextLength;
}
