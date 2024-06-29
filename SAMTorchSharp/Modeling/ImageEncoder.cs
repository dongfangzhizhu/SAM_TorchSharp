using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.Activations;

namespace SAMTorchSharp.Modeling
{
    public static partial class Helper
    {
        public static (Tensor, (long, long)) WindowPartition(Tensor x, int windowSize)
        {
            var shape = x.shape;
            var (B, H, W, C) = (shape[0], shape[1], shape[2], shape[3]);

            long padH = (windowSize - H % windowSize) % windowSize;
            long padW = (windowSize - W % windowSize) % windowSize;
            if (padH > 0 || padW > 0)
            {
                x = functional.pad(x, new long[]{0, 0, 0, padW, 0, padH});
            }
            long Hp = H + padH, Wp = W + padW;

            x = x.view(B, Hp / windowSize, windowSize, Wp / windowSize, windowSize, C);
            Tensor windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, windowSize, windowSize, C);

            return (windows, (Hp, Wp));
        }

        public static Tensor WindowUnpartition(Tensor windows, int windowSize, (long, long) padHw, (long, long) hw)
        {
            var (H, W) = hw;
            var (Hp, Wp) = padHw;

            long B = windows.shape[0] / ((int)(Hp * Wp) / windowSize / windowSize);

            Tensor x = windows.view(B, (int)Hp / windowSize, (int)Wp / windowSize, windowSize, windowSize, -1);
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1);

            if (Hp > H || Wp > W)
            {
                //x = x.slice(1, null, H).slice(2, null, W).contiguous();
                x = x[TensorIndex.Colon, TensorIndex.Slice(null, H), TensorIndex.Slice(null, W), TensorIndex.Colon].contiguous();
            }

            return x;
        }
        public static Tensor GetRelPos(long qSize, long kSize, Tensor relPos)
        {
            long maxRelDist = (2 * Math.Max(qSize, kSize) - 1);
            Tensor relPosResized;

            // Interpolate rel pos if needed.
            if (relPos.shape[0] != maxRelDist)
            {
                // Interpolate rel pos.
                relPosResized = functional.interpolate(
                    relPos.reshape(1, relPos.shape[0], -1).permute(0, 2, 1),
                    size: new long[]{ maxRelDist},
                    mode:InterpolationMode.Linear
                ).reshape(-1, maxRelDist).permute(1, 0);
            }
            else
            {
                relPosResized = relPos;
            }

            // Scale the coords with short length if shapes for q and k are different.
            Tensor qCoords = torch.arange(qSize)[TensorIndex.Ellipsis, TensorIndex.None] * Math.Max(kSize / (double)qSize, 1.0);
            Tensor kCoords = torch.arange(kSize)[TensorIndex.None,TensorIndex.Ellipsis] * Math.Max(qSize / (double)kSize, 1.0);
            Tensor relativeCoords = (qCoords - kCoords) + (kSize - 1) * Math.Max(qSize / (double)kSize, 1.0);

            return relPosResized[relativeCoords.to_type(ScalarType.Int64)];
        }
        public static Tensor AddDecomposedRelPos(Tensor attn,Tensor q,Tensor relPosH,Tensor relPosW,(long, long) qSize,(long, long) kSize)
        {
            var (qH, qW) = qSize;
            var (kH, kW) = kSize;
            var shape = q.shape;
            var (b,dim) = (shape[0], shape[2]);
            var rq = q.reshape(new long[] { b, qH, qW, dim });

            Tensor rh = GetRelPos(qH, kH, relPosH);
            Tensor rw = GetRelPos(qW, kW, relPosW);

            Tensor relH = torch.einsum("bhwc,hkc->bhwk", rq, rh);
            Tensor relW = torch.einsum("bhwc,wkc->bhwk", rq, rw);

            attn = (
                attn.view(b, qH, qW, kH, kW) +
                relH[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Colon, TensorIndex.None] +
                relW[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Colon, TensorIndex.None, TensorIndex.Colon]
            ).view(b, qH * qW, kH * kW);

            return attn;
        }
    }
    
    public class PatchEmbed : Module<Tensor,Tensor>
    {
        private readonly Conv2d proj;

        public PatchEmbed(int  kernelSize = 16,int stride = 16,int padding = 0,int inChans = 3,int embedDim = 768,string name= "pos_embed") : base(name)
        {
            // Initialize the projection layer
            this.proj = nn.Conv2d(inChans, embedDim,kernelSize: kernelSize,stride: stride,padding: padding);
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            // Apply the projection layer
            x = this.proj.forward(x);

            // Permute the dimensions from B C H W to B H W C
            x = x.permute(0, 2, 3, 1);

            return x;
        }
    }

    public class Attention : Module<Tensor,Tensor>
    {
        private readonly int numHeads;
        private readonly double scale;
        private readonly Linear qkv;
        private readonly Linear proj;
        private readonly bool useRelPos;
        private readonly Parameter rel_pos_h;
        private readonly Parameter rel_pos_w;

        public Attention(
            int dim,
            int numHeads = 8,
            bool qkvBias = true,
            bool useRelPos = false,
            bool relPosZeroInit = true,
            (int, int)? inputSize = null
        ) : base("Attention")
        {
            this.numHeads = numHeads;
            int headDim = dim / numHeads;
            this.scale = Math.Pow(headDim, -0.5);

            this.qkv = Linear(dim, dim * 3, hasBias: qkvBias);
            this.proj = Linear(dim, dim);

            this.useRelPos = useRelPos;
            if (this.useRelPos)
            {
                if (!inputSize.HasValue)
                {
                    throw new ArgumentException("Input size must be provided if using relative positional encoding.");
                }
                // Initialize relative positional embeddings
                this.rel_pos_h = Parameter( torch.zeros(new long[] { 2 * inputSize.Value.Item1 - 1, headDim }));
                this.rel_pos_w = Parameter( torch.zeros(new long[] { 2 * inputSize.Value.Item2 - 1, headDim }));
               
            }
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            var shape = x.shape;
            var (B, H, W)  = (shape[0], shape[1], shape[2]);
            // qkv with shape (3, B, nHead, H * W, C)
            Tensor qkv = this.qkv.forward(x).reshape(B, H * W, 3, this.numHeads, -1).permute(2,0,3,1,4);
            // q, k, v with shape (B * nHead, H * W, C)
            qkv = qkv.reshape(3, B * numHeads, H * W, -1);
            var qkvs = qkv.unbind(0);
            var (q, k, v) = (qkvs[0], qkvs[1], qkvs[2]);

            Tensor attn = (q * this.scale).matmul(k.transpose(-2, -1));

            if (this.useRelPos)
            {
                // The add_decomposed_rel_pos function is not part of TorchSharp and would need to be implemented separately.
                 attn = Helper.AddDecomposedRelPos(attn, q, this.rel_pos_h, this.rel_pos_w, (H, W), (H, W));
            }

            attn = attn.softmax(dim: -1);
            x = attn.matmul(v).view(B, this.numHeads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1);
            x = this.proj.forward(x);

            return x;
        }
    }

    public class Block : Module<Tensor,Tensor>
    {
        private readonly Module<Tensor,Tensor> norm1;
        private readonly Attention attn;
        private readonly Module<Tensor, Tensor> norm2;
        private readonly MLPBlock mlp;
        private readonly int windowSize;

        public Block(
            int dim,
            int numHeads,
            float mlpRatio = 4.0f,
            bool qkvBias = true,
            Func<long, LayerNorm> normLayer = null,
            Func<GELU> actLayer = null,
            bool useRelPos = false,
            bool relPosZeroInit = true,
            int windowSize = 0,
            (int, int)? inputSize = null,
            string name= "Block"
        ) : base(name)
        {
            if (normLayer== null)
            {
                normLayer= (long ch) => { return nn.LayerNorm(ch); };
            }
            if (actLayer== null)
            {
                actLayer= GELU;
            }
            // Initialize normalization layers
            this.norm1 = normLayer(dim);
            this.norm2 = normLayer(dim);

            // Initialize attention layer
            this.attn = new Attention(
                dim,
                numHeads: numHeads,
                qkvBias: qkvBias,
                useRelPos: useRelPos,
                relPosZeroInit: relPosZeroInit,
                inputSize: windowSize == 0 ? inputSize : (windowSize, windowSize)
            );

            // Initialize MLP block
            this.mlp = new MLPBlock("",embeddingDim: dim,mlpDim: (int)(dim * mlpRatio),act: actLayer);

            this.windowSize = windowSize;
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            // Shortcut for residual connection
            Tensor shortcut = x;

            // First normalization
            x = this.norm1.forward(x);
            

            // Window partition if window size is greater than 0
            (long,long) padHw = (1,1);
            long H = 0;
            long W = 0;
            if (this.windowSize > 0)
            {
                H = x.shape[1];
                W = x.shape[2];
                 (x, padHw) = Helper.WindowPartition(x, this.windowSize);
            }

            // Attention
            x = this.attn.forward(x);

            // Reverse window partition if window size is greater than 0
            if (this.windowSize > 0)
            {
                x = Helper.WindowUnpartition(x, this.windowSize, padHw, (H, W));
            }

            // Add shortcut
            x = shortcut + x;

            // Second normalization and MLP
            x = x+ this.mlp.forward(this.norm2.forward(x));

            return x;
        }

    }
    public class ImageEncoderViT : ImageEncoderViTBase
    {
        private readonly PatchEmbed patch_embed;
        private readonly Parameter pos_embed;
        private readonly ModuleList<Module<Tensor, Tensor>> blocks;
        private readonly Sequential neck;
        //public readonly int imgSize;
        private readonly bool useAbsPos;

        public ImageEncoderViT(
            string name = "ImageEncoderViT",
            int imgSize = 1024,
            int patchSize = 16,
            int inChans = 3,
            int embedDim = 768,
            int depth = 12,
            int numHeads = 12,
            float mlpRatio = 4.0f,
            int outChans = 256,
            bool qkvBias = true,
            Func<long,LayerNorm> normLayer= null, 
            Func<GELU> actLayer= null,
            bool useAbsPos = true,
            bool useRelPos = false,
            bool relPosZeroInit = true,
            int windowSize = 0,
            params int[] globalAttnIndexes
        ) : base(imgSize,name)
        {
            //this.imgSize = imgSize;
            this.useAbsPos = useAbsPos;
            if (normLayer==null)
            {
                normLayer= (long ch) => { return nn.LayerNorm(ch,eps:1e-6); };
            }
            if(actLayer==null)
            {
                actLayer= GELU;
            }
            // Initialize patch embedding
            this.patch_embed = new PatchEmbed(kernelSize: patchSize,stride: patchSize,inChans: inChans,embedDim: embedDim);

            // Initialize position embedding
            this.pos_embed = useAbsPos ? Parameter(torch.zeros(new long[] { 1, imgSize / patchSize, imgSize / patchSize, embedDim })) : null;

            // Initialize blocks
            this.blocks = nn.ModuleList<Module<Tensor, Tensor>>();
            for (int i = 0; i < depth; i++)
            {
                bool isGlobalAttention = Array.IndexOf(globalAttnIndexes, i) >= 0;
                int effectiveWindowSize = isGlobalAttention ? 0 : windowSize;
                var block = new Block(
                    dim: embedDim,
                    numHeads: numHeads,
                    mlpRatio: mlpRatio,
                    qkvBias: qkvBias,
                    normLayer: normLayer,
                    actLayer: actLayer,
                    useRelPos: useRelPos,
                    relPosZeroInit: relPosZeroInit,
                    windowSize: effectiveWindowSize,
                    inputSize: (imgSize / patchSize, imgSize / patchSize)
                );
                this.blocks.Add(block);
            }

            // Initialize the neck layers
            this.neck = torch.nn.Sequential(
                nn.Conv2d(embedDim, outChans, 1, bias: false),
                new LayerNorm2d(outChans),
                nn.Conv2d(outChans, outChans, 3, padding: 1, bias: false),
                new LayerNorm2d(outChans)
            );
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            x = this.patch_embed.forward(x);
            if (useAbsPos)
            {
                x = x + this.pos_embed;
            }
            foreach (var blk in this.blocks)
            {
                x = blk.forward(x);
            }

            x = x.permute(0, 3, 1, 2);
            x = this.neck.forward(x);
            return x;
        }
    }
}
