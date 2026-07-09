using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SAMTorchSharp.Modeling.Sam2
{
    /// <summary>
    /// 对应 sam2/modeling/memory_encoder.py: MaskDownSampler。
    /// 逐层 conv+LayerNorm2d+GELU 把 mask 从原始分辨率下采样 total_stride 倍(默认16=2^4)，
    /// 每次下采样通道数按 stride^2 增长，最后 1x1 conv 投影到 embed_dim。
    /// </summary>
    public class MaskDownSampler : Module<Tensor, Tensor>
    {
        public readonly Sequential encoder;

        public MaskDownSampler(
            long embedDim = 256,
            long kernelSize = 4,
            long stride = 4,
            long padding = 0,
            long totalStride = 16,
            string name = "MaskDownSampler") : base(name)
        {
            int numLayers = (int)(Math.Log2(totalStride) / Math.Log2(stride));
            if ((long)Math.Pow(stride, numLayers) != totalStride)
            {
                throw new ArgumentException("stride^numLayers 必须等于 totalStride");
            }

            var modules = new List<(string, Module<Tensor, Tensor>)>();
            long maskInChans = 1, maskOutChans = 1;
            for (int i = 0; i < numLayers; i++)
            {
                maskOutChans = maskInChans * (stride * stride);
                modules.Add(($"{modules.Count}", Conv2d(maskInChans, maskOutChans, kernelSize: kernelSize, stride: stride, padding: padding)));
                modules.Add(($"{modules.Count}", new LayerNorm2d(maskOutChans)));
                modules.Add(($"{modules.Count}", GELU()));
                maskInChans = maskOutChans;
            }
            modules.Add(($"{modules.Count}", Conv2d(maskOutChans, embedDim, kernelSize: 1)));

            encoder = Sequential(modules);

            RegisterComponents();
        }

        public override Tensor forward(Tensor x) => encoder.forward(x);
    }

    /// <summary>
    /// 对应 memory_encoder.py: CXBlock（轻量改写自 ConvNeXt Block）。
    /// DwConv -> LayerNorm2d -> permute(NCHW->NHWC) -> Linear(4x) -> GELU -> Linear -> gamma缩放 -> permute back -> 残差。
    /// </summary>
    public class CXBlock : Module<Tensor, Tensor>
    {
        public readonly Conv2d dwconv;
        public readonly LayerNorm2d norm;
        public readonly Linear pwconv1;
        public readonly Module<Tensor, Tensor> act;
        public readonly Linear pwconv2;
        public readonly Parameter? gamma;
        public readonly Module<Tensor, Tensor> drop_path;

        public CXBlock(
            long dim,
            long kernelSize = 7,
            long padding = 3,
            double dropPath = 0.0,
            double layerScaleInitValue = 1e-6,
            bool useDwconv = true,
            string name = "CXBlock") : base(name)
        {
            dwconv = Conv2d(dim, dim, kernelSize: kernelSize, padding: padding, groups: useDwconv ? dim : 1);
            norm = new LayerNorm2d(dim, eps: 1e-6);
            pwconv1 = Linear(dim, 4 * dim);
            act = GELU();
            pwconv2 = Linear(4 * dim, dim);

            if (layerScaleInitValue > 0)
            {
                gamma = Parameter(layerScaleInitValue * ones(dim));
            }

            drop_path = dropPath > 0.0 ? new DropPath(dropPath) : Identity();

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            Tensor input = x;
            x = dwconv.forward(x);
            x = norm.forward(x);
            x = x.permute(0, 2, 3, 1); // NCHW -> NHWC
            x = pwconv1.forward(x);
            x = act.forward(x);
            x = pwconv2.forward(x);
            if (gamma is not null)
            {
                x = gamma * x;
            }
            x = x.permute(0, 3, 1, 2); // NHWC -> NCHW

            x = input + drop_path.forward(x);
            return x;
        }
    }

    /// <summary>
    /// 对应 memory_encoder.py: Fuser。按顺序执行若干个（默认2个）CXBlock，可选输入投影层。
    /// </summary>
    public class Fuser : Module<Tensor, Tensor>
    {
        public readonly Module<Tensor, Tensor> proj;
        public readonly ModuleList<CXBlock> layers;

        public Fuser(
            CXBlock layer,
            int numLayers,
            long? dim = null,
            bool inputProjection = false,
            string name = "Fuser") : base(name)
        {
            proj = Identity();
            if (inputProjection)
            {
                if (dim is null) throw new ArgumentException("inputProjection=true 时必须提供 dim");
                proj = Conv2d(dim.Value, dim.Value, kernelSize: 1);
            }

            layers = new ModuleList<CXBlock>();
            for (int i = 0; i < numLayers; i++)
            {
                layers.Add(CloneCXBlock(layer));
            }

            RegisterComponents();
        }

        private static CXBlock CloneCXBlock(CXBlock template)
        {
            // 对应 Python 端 get_clones(copy.deepcopy)：结构相同、参数独立初始化的新实例。
            // 由于 CXBlock 的构造参数在推理阶段（加载权重后）不影响数值结果，这里按默认参数创建，
            // 依赖上层通过 load_safetensors 用真实权重整体覆盖。
            long dim = template.dwconv.weight!.shape[0];
            long kernelSize = template.dwconv.weight!.shape[2];
            long padding = (kernelSize - 1) / 2;
            bool useDwconv = template.dwconv.weight!.shape[1] == 1;
            return new CXBlock(dim, kernelSize, padding, 0.0, template.gamma is not null ? 1e-6 : 0.0, useDwconv);
        }

        public override Tensor forward(Tensor x)
        {
            x = proj.forward(x);
            for (int i = 0; i < layers.Count; i++)
            {
                x = layers[i].forward(x);
            }
            return x;
        }
    }

    public class MemoryEncoderOutput
    {
        public Tensor VisionFeatures { get; init; } = null!;
        public IList<Tensor> VisionPosEnc { get; init; } = null!;
    }

    /// <summary>
    /// 对应 memory_encoder.py: MemoryEncoder。
    /// 将当前帧的视觉特征与（经过 sigmoid 的）预测 mask 融合，编码成供未来帧 cross-attend 的记忆特征。
    /// </summary>
    public class MemoryEncoder : Module
    {
        public readonly MaskDownSampler mask_downsampler;
        public readonly Conv2d pix_feat_proj;
        public readonly Fuser fuser;
        public readonly PositionEmbeddingSine position_encoding;
        public readonly Module<Tensor, Tensor> out_proj;

        public MemoryEncoder(
            long outDim,
            MaskDownSampler maskDownsampler,
            Fuser fuser,
            PositionEmbeddingSine positionEncoding,
            long inDim = 256,
            string name = "MemoryEncoder") : base(name)
        {
            mask_downsampler = maskDownsampler;
            pix_feat_proj = Conv2d(inDim, inDim, kernelSize: 1);
            this.fuser = fuser;
            position_encoding = positionEncoding;

            out_proj = Identity();
            if (outDim != inDim)
            {
                out_proj = Conv2d(inDim, outDim, kernelSize: 1);
            }

            RegisterComponents();
        }

        public MemoryEncoderOutput forward(Tensor pixFeat, Tensor masks, bool skipMaskSigmoid = false)
        {
            if (!skipMaskSigmoid)
            {
                masks = sigmoid(masks);
            }
            masks = mask_downsampler.forward(masks);

            pixFeat = pixFeat.to(masks.device);

            Tensor x = pix_feat_proj.forward(pixFeat);
            x = x + masks;
            x = fuser.forward(x);
            x = out_proj.forward(x);

            Tensor pos = position_encoding.forward(x).to(x.dtype);

            return new MemoryEncoderOutput
            {
                VisionFeatures = x,
                VisionPosEnc = new List<Tensor> { pos },
            };
        }
    }
}
