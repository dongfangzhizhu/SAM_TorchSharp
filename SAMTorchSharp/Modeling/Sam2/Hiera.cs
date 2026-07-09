using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SAMTorchSharp.Modeling.Sam2
{
    /// <summary>
    /// 对应 sam2/modeling/backbones/utils.py 中的窗口工具函数与 PatchEmbed。
    /// </summary>
    public static class HieraUtils
    {
        /// <summary>
        /// 将 [B, H, W, C] 的输入划分为不重叠窗口，必要时做零填充。
        /// 返回 windows: [B*numWindows, windowSize, windowSize, C] 以及填充后的 (Hp, Wp)。
        /// </summary>
        public static (Tensor windows, (long Hp, long Wp) padHw) WindowPartition(Tensor x, int windowSize)
        {
            var shape = x.shape;
            var (B, H, W, C) = (shape[0], shape[1], shape[2], shape[3]);

            long padH = (windowSize - H % windowSize) % windowSize;
            long padW = (windowSize - W % windowSize) % windowSize;
            if (padH > 0 || padW > 0)
            {
                x = functional.pad(x, new long[] { 0, 0, 0, padW, 0, padH });
            }
            long Hp = H + padH, Wp = W + padW;

            x = x.view(B, Hp / windowSize, windowSize, Wp / windowSize, windowSize, C);
            Tensor windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, windowSize, windowSize, C);

            return (windows, (Hp, Wp));
        }

        /// <summary>
        /// window_partition 的逆操作，还原为 [B, H, W, C]，并去掉填充部分。
        /// </summary>
        public static Tensor WindowUnpartition(Tensor windows, int windowSize, (long Hp, long Wp) padHw, (long H, long W) hw)
        {
            var (Hp, Wp) = padHw;
            var (H, W) = hw;
            long B = windows.shape[0] / (Hp * Wp / windowSize / windowSize);

            Tensor x = windows.reshape(B, Hp / windowSize, Wp / windowSize, windowSize, windowSize, -1);
            x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1);

            if (Hp > H || Wp > W)
            {
                x = x[TensorIndex.Colon, TensorIndex.Slice(null, H), TensorIndex.Slice(null, W), TensorIndex.Colon].contiguous();
            }
            return x;
        }
    }

    /// <summary>
    /// 对应 sam2/modeling/backbones/utils.py: PatchEmbed。
    /// 输入 [B,C,H,W] -> conv -> 输出 [B,H',W',C]（注意通道放在最后一维，与 SAM1 的 PatchEmbed 相同布局）。
    /// </summary>
    public class HieraPatchEmbed : Module<Tensor, Tensor>
    {
        // 字段名 proj 对应 Python 端 PatchEmbed.proj，保持 state_dict key 一致（如 patch_embed.proj.weight）。
        private readonly Conv2d proj;

        public HieraPatchEmbed(
            long inChans = 3,
            long embedDim = 768,
            long kernelSize = 7,
            long stride = 4,
            long padding = 3,
            string name = "PatchEmbed") : base(name)
        {
            proj = Conv2d(inChans, embedDim, kernelSize: kernelSize, stride: stride, padding: padding);
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            x = proj.forward(x);
            // B C H W -> B H W C
            return x.permute(0, 2, 3, 1);
        }
    }

    /// <summary>
    /// do_pool: 对 [B,H,W,C] 输入做池化（MaxPool2d），期间需要转换到 [B,C,H,W] 布局。
    /// </summary>
    internal static class HieraOps
    {
        public static Tensor DoPool(Tensor x, Module<Tensor, Tensor>? pool, Module<Tensor, Tensor>? norm = null)
        {
            if (pool is null) return x;
            // (B, H, W, C) -> (B, C, H, W)
            x = x.permute(0, 3, 1, 2);
            x = pool.forward(x);
            // (B, C, H', W') -> (B, H', W', C)
            x = x.permute(0, 2, 3, 1);
            if (norm is not null)
            {
                x = norm.forward(x);
            }
            return x;
        }
    }

    /// <summary>
    /// 对应 hieradet.py: MultiScaleAttention。
    /// qkv 用单个 Linear (dim -> dim_out*3)，可选 q_pool 在 stage 切换处对 Q 做下采样。
    /// </summary>
    public class MultiScaleAttention : Module<Tensor, Tensor>
    {
        private readonly long dim;
        private readonly long dimOut;
        private readonly int numHeads;
        private readonly Module<Tensor, Tensor>? qPool;
        private readonly Linear qkv;
        private readonly Linear proj;

        public MultiScaleAttention(
            long dim,
            long dimOut,
            int numHeads,
            Module<Tensor, Tensor>? qPool = null,
            string name = "MultiScaleAttention") : base(name)
        {
            this.dim = dim;
            this.dimOut = dimOut;
            this.numHeads = numHeads;
            this.qPool = qPool;

            qkv = Linear(dim, dimOut * 3);
            proj = Linear(dimOut, dimOut);

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            var shape = x.shape;
            var (B, H, W) = (shape[0], shape[1], shape[2]);

            // qkv: (B, H*W, 3, nHead, C)
            Tensor qkvOut = qkv.forward(x).reshape(B, H * W, 3, numHeads, -1);
            var unbound = qkvOut.unbind(2);
            var (q, k, v) = (unbound[0], unbound[1], unbound[2]);

            if (qPool is not null)
            {
                Tensor qImg = q.reshape(B, H, W, -1);
                qImg = HieraOps.DoPool(qImg, qPool);
                H = qImg.shape[1];
                W = qImg.shape[2];
                q = qImg.reshape(B, H * W, numHeads, -1);
            }

            // -> [B, nHead, N, C]
            q = q.transpose(1, 2);
            k = k.transpose(1, 2);
            v = v.transpose(1, 2);

            double scale = 1.0 / Math.Sqrt(q.size(-1));
            Tensor attn = q.matmul(k.transpose(-2, -1)) * scale;
            attn = functional.softmax(attn, dim: -1);
            Tensor outT = attn.matmul(v);

            outT = outT.transpose(1, 2);
            outT = outT.reshape(B, H, W, -1);

            outT = proj.forward(outT);
            return outT;
        }
    }

    /// <summary>
    /// 对应 hieradet.py: MultiScaleBlock。
    /// </summary>
    public class MultiScaleBlock : Module<Tensor, Tensor>
    {
        private readonly long dim;
        private readonly long dimOut;
        private readonly int windowSize;
        private readonly (int, int)? qStride;
        private readonly LayerNorm norm1;
        private readonly MaxPool2d? pool;
        private readonly MultiScaleAttention attn;
        private readonly Module<Tensor, Tensor> dropPath;
        private readonly LayerNorm norm2;
        private readonly MLP mlp;
        // 字段名 proj 对应 Python 端 MultiScaleBlock.proj（仅 dim!=dim_out 时存在）。
        private readonly Linear? proj;

        public long GetDimOut() => dimOut;

        public MultiScaleBlock(
            long dim,
            long dimOut,
            int numHeads,
            double mlpRatio = 4.0,
            double dropPath = 0.0,
            (int, int)? qStride = null,
            int windowSize = 0,
            string name = "MultiScaleBlock") : base(name)
        {
            this.dim = dim;
            this.dimOut = dimOut;
            this.windowSize = windowSize;
            this.qStride = qStride;

            norm1 = LayerNorm(new long[] { dim }, eps: 1e-6);

            if (qStride.HasValue)
            {
                var (sh, sw) = qStride.Value;
                pool = MaxPool2d(kernelSize: (sh, sw), stride: (sh, sw));
            }

            attn = new MultiScaleAttention(dim, dimOut, numHeads, pool);
            this.dropPath = dropPath > 0.0 ? new DropPath(dropPath) : Identity();

            norm2 = LayerNorm(new long[] { dimOut }, eps: 1e-6);
            mlp = new MLP((int)dimOut, (int)(dimOut * mlpRatio), (int)dimOut, 2, activation: () => GELU());

            if (dim != dimOut)
            {
                proj = Linear(dim, dimOut);
            }

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            Tensor shortcut = x; // B H W C
            x = norm1.forward(x);

            if (dim != dimOut)
            {
                Tensor projected = proj!.forward(x);
                shortcut = HieraOps.DoPool(projected, pool);
            }

            int windowSizeLocal = windowSize;
            (long Hp, long Wp) padHw = default;
            long H = 0, W = 0;
            if (windowSizeLocal > 0)
            {
                H = x.shape[1];
                W = x.shape[2];
                (x, padHw) = HieraUtils.WindowPartition(x, windowSizeLocal);
            }

            x = attn.forward(x);

            if (qStride.HasValue)
            {
                windowSizeLocal = windowSize / qStride.Value.Item1;
                H = shortcut.shape[1];
                W = shortcut.shape[2];

                long padH = (windowSizeLocal - H % windowSizeLocal) % windowSizeLocal;
                long padW = (windowSizeLocal - W % windowSizeLocal) % windowSizeLocal;
                padHw = (H + padH, W + padW);
            }

            if (windowSize > 0)
            {
                x = HieraUtils.WindowUnpartition(x, windowSizeLocal, padHw, (H, W));
            }

            x = shortcut + dropPath.forward(x);
            x = x + dropPath.forward(mlp.forward(norm2.forward(x)));
            return x;
        }
    }

    /// <summary>
    /// 对应 hieradet.py: Hiera（SAM2 图像 backbone 主干网络）。
    /// 返回各 stage 末尾的多尺度特征（[B,C,H,W] 布局），用于 FpnNeck。
    /// </summary>
    public class Hiera : Module<Tensor, IList<Tensor>>
    {
        private readonly (int, int) qStride;
        private readonly int[] stageEnds;
        private readonly HashSet<int> qPoolBlocks;
        private readonly bool returnIntermLayers;
        // 以下字段名(patch_embed/pos_embed/pos_embed_window/blocks)必须与 Python 端 Hiera 的属性名一致，
        // 以保证 safetensors/state_dict 的 key 能够直接对应。
        private readonly HieraPatchEmbed patch_embed;
        private readonly int[] globalAttBlocks;
        private readonly (int, int) windowPosEmbedBkgSpatialSize;
        private readonly Parameter pos_embed;
        private readonly Parameter pos_embed_window;
        private readonly ModuleList<MultiScaleBlock> blocks;
        private readonly int[] windowSpec;

        public long[] ChannelList { get; }

        public Hiera(
            long embedDim = 96,
            int numHeads = 1,
            double dropPathRate = 0.0,
            int qPool = 3,
            (int, int)? qStride = null,
            int[]? stages = null,
            double dimMul = 2.0,
            double headMul = 2.0,
            (int, int)? windowPosEmbedBkgSpatialSize = null,
            int[]? windowSpec = null,
            int[]? globalAttBlocks = null,
            bool returnIntermLayers = true,
            string name = "Hiera") : base(name)
        {
            stages ??= new[] { 2, 3, 16, 3 };
            this.windowSpec = windowSpec ?? new[] { 8, 4, 14, 7 };
            this.qStride = qStride ?? (2, 2);
            this.globalAttBlocks = globalAttBlocks ?? new[] { 12, 16, 20 };
            this.windowPosEmbedBkgSpatialSize = windowPosEmbedBkgSpatialSize ?? (14, 14);
            this.returnIntermLayers = returnIntermLayers;

            if (stages.Length != this.windowSpec.Length)
            {
                throw new ArgumentException("stages 和 window_spec 长度必须一致");
            }

            int depth = stages.Sum();
            stageEnds = new int[stages.Length];
            int acc = 0;
            for (int i = 0; i < stages.Length; i++)
            {
                acc += stages[i];
                stageEnds[i] = acc - 1;
            }

            var qPoolBlocksList = stageEnds.Take(stageEnds.Length - 1).Select(x => x + 1).Take(qPool).ToList();
            qPoolBlocks = new HashSet<int>(qPoolBlocksList);

            patch_embed = new HieraPatchEmbed(inChans: 3, embedDim: embedDim);

            pos_embed = Parameter(zeros(1, embedDim, this.windowPosEmbedBkgSpatialSize.Item1, this.windowPosEmbedBkgSpatialSize.Item2));
            pos_embed_window = Parameter(zeros(1, embedDim, this.windowSpec[0], this.windowSpec[0]));

            float[] dpr = linspace(0, dropPathRate, depth).data<float>().ToArray();

            blocks = new ModuleList<MultiScaleBlock>();
            long curEmbedDim = embedDim;
            int curNumHeads = numHeads;
            int curStage = 1;
            var channelListDesc = new List<long>();

            for (int i = 0; i < depth; i++)
            {
                long dimOut = curEmbedDim;
                int windowSizeForBlock = this.windowSpec[curStage - 1];
                if (Array.IndexOf(this.globalAttBlocks, i) >= 0)
                {
                    windowSizeForBlock = 0;
                }

                if (Array.IndexOf(stageEnds, i - 1) >= 0)
                {
                    dimOut = (long)(curEmbedDim * dimMul);
                    curNumHeads = (int)(curNumHeads * headMul);
                    curStage += 1;
                }

                (int, int)? blockQStride = qPoolBlocks.Contains(i) ? this.qStride : null;

                var block = new MultiScaleBlock(
                    dim: curEmbedDim,
                    dimOut: dimOut,
                    numHeads: curNumHeads,
                    dropPath: dpr[i],
                    qStride: blockQStride,
                    windowSize: windowSizeForBlock);

                curEmbedDim = dimOut;
                blocks.Add(block);
            }

            if (returnIntermLayers)
            {
                for (int i = stageEnds.Length - 1; i >= 0; i--)
                {
                    channelListDesc.Add(blocks[stageEnds[i]].GetDimOut());
                }
            }
            else
            {
                channelListDesc.Add(blocks[^1].GetDimOut());
            }
            ChannelList = channelListDesc.ToArray();

            RegisterComponents();
        }

        private Tensor GetPosEmbed((long H, long W) hw)
        {
            var (h, w) = hw;
            Tensor windowEmbed = pos_embed_window;
            Tensor pe = functional.interpolate(pos_embed, size: new long[] { h, w }, mode: InterpolationMode.Bicubic);

            long[] tileFactors = new long[pe.shape.Length];
            for (int i = 0; i < pe.shape.Length; i++)
            {
                tileFactors[i] = pe.shape[i] / windowEmbed.shape[i];
            }
            Tensor tiled = windowEmbed.tile(tileFactors);
            pe = pe + tiled;
            pe = pe.permute(0, 2, 3, 1);
            return pe;
        }

        public override IList<Tensor> forward(Tensor x)
        {
            x = patch_embed.forward(x);
            // x: (B, H, W, C)
            x = x + GetPosEmbed((x.shape[1], x.shape[2]));

            var outputs = new List<Tensor>();
            for (int i = 0; i < blocks.Count; i++)
            {
                x = blocks[i].forward(x);
                bool isLastStage = i == stageEnds[^1];
                bool isStageEnd = Array.IndexOf(stageEnds, i) >= 0;
                if (isLastStage || (isStageEnd && returnIntermLayers))
                {
                    Tensor feats = x.permute(0, 3, 1, 2);
                    outputs.Add(feats);
                }
            }

            return outputs;
        }
    }
}
