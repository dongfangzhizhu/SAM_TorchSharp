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

        protected double DropoutP => dropoutP;

        protected static Tensor SeparateHeads(Tensor x, int numHeads)
        {
            var (b, n, c) = (x.shape[0], x.shape[1], x.shape[2]);
            x = x.reshape(b, n, numHeads, c / numHeads);
            return x.transpose(1, 2);
        }

        protected static Tensor RecombineHeads(Tensor x)
        {
            var (b, nHeads, nTokens, cPerHead) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
            x = x.transpose(1, 2);
            return x.reshape(b, nTokens, nHeads * cPerHead);
        }

        protected static Tensor ScaledDotProductAttention(Tensor q, Tensor k, Tensor v, double dropoutP, bool training)
        {
            double scale = 1.0 / Math.Sqrt(q.size(-1));
            Tensor attn = q.matmul(k.transpose(-2, -1)) * scale;
            attn = functional.softmax(attn, dim: -1);
            if (training && dropoutP > 0.0)
            {
                attn = functional.dropout(attn, dropoutP);
            }
            return attn.matmul(v);
        }

        public override Tensor forward(Tensor q, Tensor k, Tensor v)
        {
            q = q_proj.forward(q);
            k = k_proj.forward(k);
            v = v_proj.forward(v);

            q = SeparateHeads(q, num_heads);
            k = SeparateHeads(k, num_heads);
            v = SeparateHeads(v, num_heads);

            Tensor outT = ScaledDotProductAttention(q, k, v, dropoutP, training);

            outT = RecombineHeads(outT);
            outT = out_proj.forward(outT);
            return outT;
        }
    }

    /// <summary>
    /// 对应 sam2/modeling/sam/transformer.py: RoPEAttention。
    /// 在标准 Attention 基础上对 q 和 k（排除末尾 num_k_exclude_rope 个 object pointer token）
    /// 施加 2D 轴向旋转位置编码。用于 memory attention 的 self-attn（在视频帧特征网格上）与
    /// cross-attn（跨帧 attend 到历史 memory）。
    /// 注意：freqs 缓存 (cos/sin 表) 不是可学习参数，不参与 state_dict，每次构造/尺寸变化时重新计算。
    /// </summary>
    public class RoPEAttention : Attention
    {
        private readonly double ropeTheta;
        private readonly bool ropeKRepeat;
        private long cachedEndX;
        private long cachedEndY;
        private Tensor cachedCos;
        private Tensor cachedSin;

        public RoPEAttention(
            long embeddingDim,
            int numHeads,
            long downsampleRate = 1,
            double dropout = 0.0,
            long? kvInDim = null,
            double ropeTheta = 10000.0,
            bool ropeKRepeat = false,
            (long endX, long endY)? featSizes = null,
            string name = "RoPEAttention")
            : base(embeddingDim, numHeads, downsampleRate, dropout, kvInDim, name)
        {
            this.ropeTheta = ropeTheta;
            this.ropeKRepeat = ropeKRepeat;

            var (endX, endY) = featSizes ?? (64, 64);
            RecomputeFreqs(endX, endY);
        }

        private void RecomputeFreqs(long endX, long endY)
        {
            long headDim = internal_dim / num_heads;
            var (cos, sin) = RotaryPositionEncoding.ComputeAxialFreqs(headDim, endX, endY, ropeTheta);
            cachedCos = cos;
            cachedSin = sin;
            cachedEndX = endX;
            cachedEndY = endY;
        }

        /// <summary>
        /// 对应 Python 端 forward(q,k,v,num_k_exclude_rope)。
        /// </summary>
        public Tensor forward(Tensor q, Tensor k, Tensor v, long numKExcludeRope)
        {
            q = q_proj.forward(q);
            k = k_proj.forward(k);
            v = v_proj.forward(v);

            q = SeparateHeads(q, num_heads);
            k = SeparateHeads(k, num_heads);
            v = SeparateHeads(v, num_heads);

            long qLen = q.shape[^2];
            long side = (long)Math.Round(Math.Sqrt(qLen));
            if (side * side != cachedEndX * cachedEndY || cachedEndX != side || cachedEndY != side)
            {
                RecomputeFreqs(side, side);
            }

            long numKRope = k.shape[^2] - numKExcludeRope;
            Tensor kRope = k[TensorIndex.Ellipsis, TensorIndex.Slice(0, numKRope), TensorIndex.Colon];
            Tensor kRest = numKExcludeRope > 0
                ? k[TensorIndex.Ellipsis, TensorIndex.Slice(numKRope, null), TensorIndex.Colon]
                : null!;

            Tensor cos = cachedCos, sin = cachedSin;
            if (numKRope != qLen)
            {
                if (!ropeKRepeat)
                {
                    throw new InvalidOperationException("k 与 q 的 token 数不一致时必须设置 rope_k_repeat=true");
                }
                long r = numKRope / qLen;
                (cos, sin) = RotaryPositionEncoding.RepeatAlongTokenDim(cachedCos, cachedSin, r);
            }

            Tensor qRot = RotaryPositionEncoding.ApplyRotary(q, cachedCos, cachedSin);
            Tensor kRopeRot = RotaryPositionEncoding.ApplyRotary(kRope, cos, sin);
            Tensor kFinal = numKExcludeRope > 0 ? cat(new[] { kRopeRot, kRest }, dim: -2) : kRopeRot;

            Tensor outT = ScaledDotProductAttention(qRot, kFinal, v, DropoutP, training);
            outT = RecombineHeads(outT);
            outT = out_proj.forward(outT);
            return outT;
        }

        /// <summary>标准 3 参数 forward（num_k_exclude_rope=0）。</summary>
        public override Tensor forward(Tensor q, Tensor k, Tensor v) => forward(q, k, v, 0);
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
