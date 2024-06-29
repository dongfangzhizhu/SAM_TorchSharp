using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SAMTorchSharp.Modeling.Transformer
{
    /// <summary>
    /// 原代码为Attention，已经检验对比过
    /// </summary>
    public class Attention : Module<Tensor, Tensor, Tensor, Tensor>
    {
        private readonly int embeddingDim;
        private readonly int numHeads;
        private readonly int downsampleRate;
        private readonly int internalDim;

        private readonly Linear q_proj;
        private readonly Linear k_proj;
        private readonly Linear v_proj;
        private readonly Linear out_proj;

        public Attention(int embeddingDim, int numHeads, int downsampleRate = 1,string name="Attention") : base(name)
        {
            this.embeddingDim = embeddingDim;
            this.numHeads = numHeads;
            this.downsampleRate = downsampleRate;
            this.internalDim = embeddingDim / downsampleRate;

            if (internalDim % numHeads != 0)
            {
                throw new ArgumentException("num_heads must divide embedding_dim.");
            }

            q_proj = Linear(embeddingDim, internalDim);
            k_proj = Linear(embeddingDim, internalDim);
            v_proj = Linear(embeddingDim, internalDim);
            out_proj = Linear(internalDim, embeddingDim);

            RegisterComponents();
        }

        private Tensor _separateHeads(Tensor x, int numHeads)
        {
            var b = x.size(0);
            var n = x.size(1);
            var c = x.size(2);
            var xReshaped = x.view(b, n, numHeads, c / numHeads);
            return xReshaped.permute(0, 2, 1, 3);//检查
        }

        private Tensor _recombineHeads(Tensor x)
        {
            var b = x.size(0);
            var nHeads = x.size(1);
            var nTokens = x.size(2);
            var cPerHead = x.size(3);
            //var xTransposed = x.permute(0, 2, 1, 3);
            var xTransposed = x.transpose(1, 2);
            return xTransposed.reshape(b, nTokens, nHeads * cPerHead);
        }

        public override Tensor forward(Tensor q, Tensor k, Tensor v)
        {
            // Input projections
            q = q_proj.forward(q);
            k = k_proj.forward(k);
            v = v_proj.forward(v);
            // Separate into heads
            q = _separateHeads(q, numHeads);
            k = _separateHeads(k, numHeads);
            v = _separateHeads(v, numHeads);

            // Attention
            var cPerHead = q.size(3);
            var attn = q.matmul(k.permute(0, 1, 3, 2));
            attn = attn.div(Math.Sqrt(cPerHead));
            attn = softmax(attn, -1);

            // Get output
            var out_t = attn.matmul(v);
            out_t = _recombineHeads(out_t);
            out_t = out_proj.forward(out_t);

            return out_t;
        }
    }

    public class TwoWayAttentionBlock : Module<Tensor, Tensor, Tensor, Tensor, (Tensor, Tensor)>
    {
        private readonly Attention self_attn;
        private readonly LayerNorm norm1;
        private readonly Attention cross_attn_image_to_token;
        private readonly LayerNorm norm2;
        private readonly Module<Tensor, Tensor> mlp;
        private readonly LayerNorm norm3;
        private readonly LayerNorm norm4;
        private readonly Attention cross_attn_token_to_image;
        private readonly bool skipFirstLayerPe;

        public TwoWayAttentionBlock(int embeddingDim, int numHeads, int mlpDim = 2048, Func<Module<Tensor,Tensor>> activation = null, int attentionDownsampleRate = 2, bool skipFirstLayerPe = false)
            : base("TwoWayAttentionBlock")
        {
            if (activation is null)
            {
                activation = () => ReLU();
            }
            this.self_attn = new Attention(embeddingDim, numHeads);
            this.norm1 = LayerNorm(embeddingDim);

            this.cross_attn_token_to_image = new Attention(embeddingDim, numHeads, downsampleRate: attentionDownsampleRate);
            this.norm2 = LayerNorm(embeddingDim);

            this.mlp = new MLPBlock("mlp", embeddingDim, mlpDim, activation);
            this.norm3 = LayerNorm(embeddingDim);

            this.norm4 = LayerNorm(embeddingDim);
            this.cross_attn_image_to_token = new Attention(embeddingDim, numHeads, downsampleRate: attentionDownsampleRate);

            this.skipFirstLayerPe = skipFirstLayerPe;

            RegisterComponents();
        }

        public override (Tensor, Tensor) forward(Tensor queries, Tensor keys, Tensor queryPe, Tensor keyPe)
        {
            // Self attention block
            Tensor attnOut;
            Tensor q;
            if (skipFirstLayerPe)
            {
                queries = self_attn.forward(q: queries, k: queries, v: queries);
            }
            else
            {
                q = queries + queryPe;
                attnOut = self_attn.forward(q: q, k: q, v: queries);
                queries.add_(attnOut);
            }
            queries = norm1.forward(queries);

            // Cross attention block, tokens attending to image embedding
            q = queries.add(queryPe);
            Tensor k = keys.add(keyPe);
            attnOut = cross_attn_token_to_image.forward(q: q, k: k, v: keys);
            queries.add_(attnOut);
            queries = norm2.forward(queries);

            // MLP block
            Tensor mlpOut = mlp.forward(queries);
            queries = queries.add(mlpOut);
            queries = norm3.forward(queries);

            // Cross attention block, image embedding attending to tokens
            q = queries.add(queryPe);
            k = keys.add(keyPe);
            attnOut = cross_attn_image_to_token.forward(q: k, k: q, v: queries);
            keys = keys.add(attnOut);
            keys = norm4.forward(keys);

            return (queries, keys);
        }
    }
    public class TwoWayTransformer : Module<Tensor, Tensor, Tensor, (Tensor, Tensor)>
    {
        private readonly int depth;
        private readonly int embeddingDim;
        private readonly int numHeads;
        private readonly int mlpDim;
        private readonly ModuleList<TwoWayAttentionBlock> layers;
        private readonly Attention final_attn_token_to_image;
        private readonly LayerNorm norm_final_attn;

        public TwoWayTransformer(int depth, int embeddingDim, int numHeads, int mlpDim,Func<Module<Tensor,Tensor>> activation = null, int attentionDownsampleRate = 2,string name= "TwoWayTransformer"): base(name)
        {
            this.depth = depth;
            this.embeddingDim = embeddingDim;
            this.numHeads = numHeads;
            this.mlpDim = mlpDim;
            if (activation is null)
            {
                activation = () => ReLU();
            }

            this.layers = new ModuleList<TwoWayAttentionBlock>();
            for (int i = 0; i < depth; i++)
            {
                this.layers.append(new TwoWayAttentionBlock(
                    embeddingDim: embeddingDim,
                    numHeads: numHeads,
                    mlpDim: mlpDim,
                    activation: activation,
                    attentionDownsampleRate: attentionDownsampleRate,
                    skipFirstLayerPe: i == 0
                ));
            }

            this.final_attn_token_to_image = new Attention(embeddingDim, numHeads, downsampleRate: attentionDownsampleRate);
            this.norm_final_attn = LayerNorm(embeddingDim);

            RegisterComponents();
        }

        public override (Tensor, Tensor) forward(Tensor imageEmbedding, Tensor imagePe, Tensor pointEmbedding)
        {
            // BxCxHxW -> BxHWxC == B x N_image_tokens x C
            var bs = imageEmbedding.size(0);
            var c = imageEmbedding.size(1);
            var h = imageEmbedding.size(2);
            var w = imageEmbedding.size(3);
            imageEmbedding = imageEmbedding.flatten(2).permute(0, 2, 1);
            imagePe = imagePe.flatten(2).permute(0, 2, 1);

            // Prepare queries
            Tensor queries = pointEmbedding;
            Tensor keys = imageEmbedding;
            // Apply transformer blocks and final layernorm
            foreach (var layer in layers)
            {
                (queries, keys) = layer.forward(queries, keys, pointEmbedding, imagePe);
            }

            // Apply the final attention layer from the points to the image
            Tensor q = queries.add(pointEmbedding);
            Tensor k = keys.add(imagePe);
            Tensor attnOut = final_attn_token_to_image.forward(q, k, keys);
            queries.add_(attnOut);
            queries = norm_final_attn.forward(queries);

            return (queries, keys);
        }
    }
}
