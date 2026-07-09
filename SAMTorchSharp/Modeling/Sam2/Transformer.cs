using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SAMTorchSharp.Modeling.Sam2
{
    /// <summary>
    /// 对应 sam2/modeling/sam/transformer.py: Attention。
    /// 与 SAM1 版本结构一致，唯一区别是 SAM2 支持 kv_in_dim（用于 RoPEAttention/memory attention，
    /// Phase2 才会用到），这里同样提供该参数以便后续复用。
    /// forward 中的 scaled_dot_product_attention 与手写 softmax(QK^T/sqrt(d))V 数学等价，
    /// 这里直接手写以避免依赖 TorchSharp 是否暴露该算子。
    /// </summary>
    public class Attention : Module<Tensor, Tensor, Tensor, Tensor>
    {
        public readonly long embedding_dim;
        public readonly long kv_in_dim;
        public readonly long internal_dim;
        public readonly int num_heads;
        public readonly Linear q_proj;
        public readonly Linear k_proj;
        public readonly Linear v_proj;
        public readonly Linear out_proj;
        private readonly double dropoutP;

        public Attention(
            long embeddingDim,
            int numHeads,
            long downsampleRate = 1,
            double dropout = 0.0,
            long? kvInDim = null,
            string name = "Attention") : base(name)
        {
            embedding_dim = embeddingDim;
            kv_in_dim = kvInDim ?? embeddingDim;
            internal_dim = embeddingDim / downsampleRate;
            num_heads = numHeads;
            if (internal_dim % numHeads != 0)
            {
                throw new ArgumentException("num_heads must divide embedding_dim (after downsample).");
            }

            q_proj = Linear(embeddingDim, internal_dim);
            k_proj = Linear(kv_in_dim, internal_dim);
            v_proj = Linear(kv_in_dim, internal_dim);
            out_proj = Linear(internal_dim, embeddingDim);

            dropoutP = dropout;

            RegisterComponents();
        }

        private Tensor SeparateHeads(Tensor x, int numHeads)
        {
            var (b, n, c) = (x.shape[0], x.shape[1], x.shape[2]);
            x = x.reshape(b, n, numHeads, c / numHeads);
            return x.transpose(1, 2);
        }

        private Tensor RecombineHeads(Tensor x)
        {
            var (b, nHeads, nTokens, cPerHead) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
            x = x.transpose(1, 2);
            return x.reshape(b, nTokens, nHeads * cPerHead);
        }

        public override Tensor forward(Tensor q, Tensor k, Tensor v)
        {
            q = q_proj.forward(q);
            k = k_proj.forward(k);
            v = v_proj.forward(v);

            q = SeparateHeads(q, num_heads);
            k = SeparateHeads(k, num_heads);
            v = SeparateHeads(v, num_heads);

            double scale = 1.0 / Math.Sqrt(q.size(-1));
            Tensor attn = q.matmul(k.transpose(-2, -1)) * scale;
            attn = functional.softmax(attn, dim: -1);
            if (training && dropoutP > 0.0)
            {
                attn = functional.dropout(attn, dropoutP);
            }
            Tensor outT = attn.matmul(v);

            outT = RecombineHeads(outT);
            outT = out_proj.forward(outT);
            return outT;
        }
    }

    /// <summary>
    /// 对应 sam2/modeling/sam/transformer.py: TwoWayAttentionBlock。结构与 SAM1 版本一致。
    /// </summary>
    public class TwoWayAttentionBlock : Module<Tensor, Tensor, Tensor, Tensor, (Tensor, Tensor)>
    {
        public readonly Attention self_attn;
        public readonly LayerNorm norm1;
        public readonly Attention cross_attn_token_to_image;
        public readonly LayerNorm norm2;
        public readonly MLP mlp;
        public readonly LayerNorm norm3;
        public readonly LayerNorm norm4;
        public readonly Attention cross_attn_image_to_token;
        private readonly bool skip_first_layer_pe;

        public TwoWayAttentionBlock(
            long embeddingDim,
            int numHeads,
            long mlpDim = 2048,
            Func<Module<Tensor, Tensor>>? activation = null,
            long attentionDownsampleRate = 2,
            bool skipFirstLayerPe = false,
            string name = "TwoWayAttentionBlock") : base(name)
        {
            self_attn = new Attention(embeddingDim, numHeads);
            norm1 = LayerNorm(embeddingDim);

            cross_attn_token_to_image = new Attention(embeddingDim, numHeads, downsampleRate: attentionDownsampleRate);
            norm2 = LayerNorm(embeddingDim);

            mlp = new MLP((int)embeddingDim, (int)mlpDim, (int)embeddingDim, 2, activation: activation ?? (() => ReLU()));
            norm3 = LayerNorm(embeddingDim);

            norm4 = LayerNorm(embeddingDim);
            cross_attn_image_to_token = new Attention(embeddingDim, numHeads, downsampleRate: attentionDownsampleRate);

            skip_first_layer_pe = skipFirstLayerPe;

            RegisterComponents();
        }

        public override (Tensor, Tensor) forward(Tensor queries, Tensor keys, Tensor queryPe, Tensor keyPe)
        {
            if (skip_first_layer_pe)
            {
                queries = self_attn.forward(queries, queries, queries);
            }
            else
            {
                Tensor q = queries + queryPe;
                Tensor attnOut = self_attn.forward(q, q, queries);
                queries = queries + attnOut;
            }
            queries = norm1.forward(queries);

            {
                Tensor q = queries + queryPe;
                Tensor k = keys + keyPe;
                Tensor attnOut = cross_attn_token_to_image.forward(q, k, keys);
                queries = queries + attnOut;
            }
            queries = norm2.forward(queries);

            Tensor mlpOut = mlp.forward(queries);
            queries = queries + mlpOut;
            queries = norm3.forward(queries);

            {
                Tensor q = queries + queryPe;
                Tensor k = keys + keyPe;
                Tensor attnOut = cross_attn_image_to_token.forward(k, q, queries);
                keys = keys + attnOut;
            }
            keys = norm4.forward(keys);

            return (queries, keys);
        }
    }

    /// <summary>
    /// 对应 sam2/modeling/sam/transformer.py: TwoWayTransformer。结构与 SAM1 版本一致。
    /// </summary>
    public class TwoWayTransformer : Module<Tensor, Tensor, Tensor, (Tensor, Tensor)>
    {
        public readonly int depth;
        public readonly long embedding_dim;
        public readonly int num_heads;
        public readonly long mlp_dim;
        public readonly ModuleList<TwoWayAttentionBlock> layers;
        public readonly Attention final_attn_token_to_image;
        public readonly LayerNorm norm_final_attn;

        public TwoWayTransformer(
            int depth,
            long embeddingDim,
            int numHeads,
            long mlpDim,
            Func<Module<Tensor, Tensor>>? activation = null,
            long attentionDownsampleRate = 2,
            string name = "TwoWayTransformer") : base(name)
        {
            this.depth = depth;
            embedding_dim = embeddingDim;
            num_heads = numHeads;
            mlp_dim = mlpDim;

            layers = new ModuleList<TwoWayAttentionBlock>();
            for (int i = 0; i < depth; i++)
            {
                layers.Add(new TwoWayAttentionBlock(
                    embeddingDim: embeddingDim,
                    numHeads: numHeads,
                    mlpDim: mlpDim,
                    activation: activation,
                    attentionDownsampleRate: attentionDownsampleRate,
                    skipFirstLayerPe: i == 0));
            }

            final_attn_token_to_image = new Attention(embeddingDim, numHeads, downsampleRate: attentionDownsampleRate);
            norm_final_attn = LayerNorm(embeddingDim);

            RegisterComponents();
        }

        public override (Tensor, Tensor) forward(Tensor imageEmbedding, Tensor imagePe, Tensor pointEmbedding)
        {
            var shape = imageEmbedding.shape;
            var (bs, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
            imageEmbedding = imageEmbedding.flatten(2).permute(0, 2, 1);
            imagePe = imagePe.flatten(2).permute(0, 2, 1);

            Tensor queries = pointEmbedding;
            Tensor keys = imageEmbedding;

            for (int i = 0; i < layers.Count; i++)
            {
                (queries, keys) = layers[i].forward(queries, keys, pointEmbedding, imagePe);
            }

            Tensor q = queries + pointEmbedding;
            Tensor k = keys + imagePe;
            Tensor attnOut = final_attn_token_to_image.forward(q, k, keys);
            queries = queries + attnOut;
            queries = norm_final_attn.forward(queries);

            return (queries, keys);
        }
    }
}
