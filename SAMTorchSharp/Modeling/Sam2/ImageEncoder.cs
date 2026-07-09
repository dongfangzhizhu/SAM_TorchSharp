using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SAMTorchSharp.Modeling.Sam2
{
    /// <summary>
    /// 对应 sam2/modeling/backbones/image_encoder.py: FpnNeck。
    /// 对 Hiera 各 stage 的多尺度特征做 1x1 conv 统一到 d_model 通道，并从低分辨率向高分辨率做
    /// top-down 融合（仅 fpnTopDownLevels 指定的层参与融合，其余层只使用 lateral 特征）。
    /// </summary>
    public class FpnNeck : Module<IList<Tensor>, (IList<Tensor> Features, IList<Tensor> Pos)>
    {
        // 字段名需与 Python 端保持一致，便于 safetensors 权重互通。
        public readonly PositionEmbeddingSine position_encoding;
        private readonly ModuleList<Sequential> convs;
        public readonly long[] backbone_channel_list;
        public readonly long d_model;
        private readonly InterpolationMode fpnInterpModel;
        private readonly string fuseType;
        private readonly HashSet<int> fpnTopDownLevels;

        public FpnNeck(
            PositionEmbeddingSine positionEncoding,
            long dModel,
            long[] backboneChannelList,
            long kernelSize = 1,
            long stride = 1,
            long padding = 0,
            string fpnInterpModel = "bilinear",
            string fuseType = "sum",
            int[]? fpnTopDownLevels = null,
            string name = "FpnNeck") : base(name)
        {
            position_encoding = positionEncoding;
            backbone_channel_list = backboneChannelList;
            d_model = dModel;
            this.fpnInterpModel = fpnInterpModel == "nearest" ? InterpolationMode.Nearest : InterpolationMode.Bilinear;

            if (fuseType != "sum" && fuseType != "avg")
            {
                throw new ArgumentException("fuse_type 必须是 sum 或 avg");
            }
            this.fuseType = fuseType;

            convs = new ModuleList<Sequential>();
            foreach (long dim in backboneChannelList)
            {
                var seq = Sequential(("conv", Conv2d(dim, dModel, kernelSize: kernelSize, stride: stride, padding: padding)));
                convs.Add(seq);
            }

            var levels = fpnTopDownLevels ?? Enumerable.Range(0, convs.Count).ToArray();
            this.fpnTopDownLevels = new HashSet<int>(levels);

            RegisterComponents();
        }

        public override (IList<Tensor> Features, IList<Tensor> Pos) forward(IList<Tensor> xs)
        {
            if (xs.Count != convs.Count)
            {
                throw new ArgumentException("输入特征数量必须与 convs 数量一致");
            }

            var outFeat = new Tensor?[convs.Count];
            var outPos = new Tensor?[convs.Count];

            Tensor? prevFeatures = null;
            int n = convs.Count - 1;
            for (int i = n; i >= 0; i--)
            {
                Tensor x = xs[i];
                Tensor lateralFeatures = convs[n - i].forward(x);

                Tensor curFeatures;
                if (fpnTopDownLevels.Contains(i) && prevFeatures is not null)
                {
                    Tensor topDownFeatures = functional.interpolate(
                        prevFeatures.to(ScalarType.Float32),
                        scale_factor: new double[] { 2.0, 2.0 },
                        mode: fpnInterpModel,
                        align_corners: fpnInterpModel == InterpolationMode.Nearest ? null : false);

                    curFeatures = lateralFeatures + topDownFeatures;
                    if (fuseType == "avg")
                    {
                        curFeatures = curFeatures / 2;
                    }
                }
                else
                {
                    curFeatures = lateralFeatures;
                }

                prevFeatures = curFeatures;
                outFeat[i] = curFeatures;
                outPos[i] = position_encoding.forward(curFeatures).to(curFeatures.dtype);
            }

            return (outFeat.Select(t => t!).ToList(), outPos.Select(t => t!).ToList());
        }
    }

    /// <summary>
    /// 对应 sam2/modeling/backbones/image_encoder.py: ImageEncoder。
    /// 将 trunk（Hiera）与 neck（FpnNeck）串联，并支持 scalp（丢弃最低分辨率输出）。
    /// </summary>
    public class ImageEncoderOutput
    {
        public Tensor VisionFeatures { get; init; } = null!;
        public IList<Tensor> VisionPosEnc { get; init; } = null!;
        public IList<Tensor> BackboneFpn { get; init; } = null!;
    }

    public class ImageEncoder : Module<Tensor, ImageEncoderOutput>
    {
        public readonly Hiera trunk;
        public readonly FpnNeck neck;
        private readonly int scalp;

        public ImageEncoder(Hiera trunk, FpnNeck neck, int scalp = 0, string name = "ImageEncoder") : base(name)
        {
            this.trunk = trunk;
            this.neck = neck;
            this.scalp = scalp;

            if (!trunk.ChannelList.SequenceEqual(neck.backbone_channel_list))
            {
                throw new ArgumentException(
                    $"Channel dims of trunk and neck do not match. Trunk: [{string.Join(',', trunk.ChannelList)}], neck: [{string.Join(',', neck.backbone_channel_list)}]");
            }

            RegisterComponents();
        }

        public override ImageEncoderOutput forward(Tensor sample)
        {
            var (features, pos) = neck.forward(trunk.forward(sample));

            if (scalp > 0)
            {
                features = features.Take(features.Count - scalp).ToList();
                pos = pos.Take(pos.Count - scalp).ToList();
            }

            Tensor src = features[^1];

            return new ImageEncoderOutput
            {
                VisionFeatures = src,
                VisionPosEnc = pos,
                BackboneFpn = features,
            };
        }
    }
}
