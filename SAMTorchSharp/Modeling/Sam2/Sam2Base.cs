using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SAMTorchSharp.Modeling.Sam2
{
    public class BackboneOut
    {
        public IList<Tensor> BackboneFpn { get; init; } = null!;
        public IList<Tensor> VisionPosEnc { get; init; } = null!;
    }

    /// <summary>
    /// 对应 sam2/modeling/sam2_base.py: SAM2Base，本类仅实现图片推理所需的子集
    /// （对应 SAM2ImagePredictor 使用到的部分）：image_encoder、sam_prompt_encoder、
    /// sam_mask_decoder、no_mem_embed。memory_attention/memory_encoder/track_step 等
    /// 视频推理相关逻辑留给 Phase2。
    ///
    /// 字段命名（image_encoder/sam_prompt_encoder/sam_mask_decoder/no_mem_embed）与 Python 端
    /// SAM2Base 保持一致，便于从完整 checkpoint 中以相同 key 加载这部分权重。
    /// </summary>
    public class Sam2Base : torch.nn.Module
    {
        public readonly ImageEncoder image_encoder;
        public readonly PromptEncoder sam_prompt_encoder;
        public readonly MaskDecoder sam_mask_decoder;
        public readonly Parameter no_mem_embed;

        public readonly long hidden_dim;
        public readonly int image_size;
        public readonly int backbone_stride;
        public readonly bool use_high_res_features_in_sam;
        public readonly bool directly_add_no_mem_embed;
        public readonly int num_feature_levels;

        public Sam2Base(
            ImageEncoder imageEncoder,
            PromptEncoder samPromptEncoder,
            MaskDecoder samMaskDecoder,
            int imageSize = 1024,
            int backboneStride = 16,
            bool useHighResFeaturesInSam = true,
            bool directlyAddNoMemEmbed = true,
            string name = "Sam2Base") : base(name)
        {
            image_encoder = imageEncoder;
            sam_prompt_encoder = samPromptEncoder;
            sam_mask_decoder = samMaskDecoder;

            hidden_dim = imageEncoder.neck.d_model;
            image_size = imageSize;
            backbone_stride = backboneStride;
            use_high_res_features_in_sam = useHighResFeaturesInSam;
            directly_add_no_mem_embed = directlyAddNoMemEmbed;
            num_feature_levels = useHighResFeaturesInSam ? 3 : 1;

            no_mem_embed = Parameter(zeros(1, 1, hidden_dim));

            RegisterComponents();
        }

        /// <summary>
        /// 对应 Python 端 SAM2Base.forward_image：跑一次 image_encoder，若开启高分辨率特征，
        /// 预先用 sam_mask_decoder.conv_s0/conv_s1 对 backbone_fpn[0]/[1] 做投影，
        /// 避免每次点击都重复计算。
        /// </summary>
        public BackboneOut ForwardImage(Tensor imgBatch)
        {
            var encoded = image_encoder.forward(imgBatch);
            var backboneFpn = encoded.BackboneFpn.ToList();

            if (use_high_res_features_in_sam)
            {
                backboneFpn[0] = sam_mask_decoder.conv_s0!.forward(backboneFpn[0]);
                backboneFpn[1] = sam_mask_decoder.conv_s1!.forward(backboneFpn[1]);
            }

            return new BackboneOut
            {
                BackboneFpn = backboneFpn,
                VisionPosEnc = encoded.VisionPosEnc,
            };
        }

        /// <summary>
        /// 对应 Python 端 SAM2Base._prepare_backbone_features：取最后 num_feature_levels 个特征层，
        /// 展平为 (H*W, N, C) 布局，同时返回各层的空间尺寸 (H, W)。
        /// </summary>
        public (IList<Tensor> VisionFeats, IList<(long H, long W)> FeatSizes) PrepareBackboneFeatures(BackboneOut backboneOut)
        {
            int n = backboneOut.BackboneFpn.Count;
            var featureMaps = backboneOut.BackboneFpn.Skip(n - num_feature_levels).ToList();
            var visionPosEmbeds = backboneOut.VisionPosEnc.Skip(backboneOut.VisionPosEnc.Count - num_feature_levels).ToList();

            var featSizes = visionPosEmbeds.Select(x => (x.shape[^2], x.shape[^1])).ToList();
            var visionFeats = featureMaps.Select(x => x.flatten(2).permute(2, 0, 1)).ToList();

            return (visionFeats, featSizes);
        }
    }
}
