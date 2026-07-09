using SAMTorchSharp.Modeling.Sam2;
using TorchSharp;
using TorchSharp.PyBridge;

namespace SAMTorchSharp
{
    /// <summary>
    /// SAM2 图片推理管线的构建器，对应 Python 端 sam2/build_sam.py 中 build_sam2 + sam2.1_hiera_*.yaml 配置。
    /// 注意：当前仅实现图片推理所需的子集（image_encoder + sam_prompt_encoder + sam_mask_decoder），
    /// 不包含 memory_attention/memory_encoder（视频推理相关，留给 Phase2）。
    /// 因此从官方完整 checkpoint 加载权重时，需要使用非严格模式（strict=false），
    /// memory_attention.*/memory_encoder.* 等 key 会被跳过，仅加载图片推理用到的权重。
    /// </summary>
    public static class BuildSam2
    {
        public static Sam2Base BuildSam2HieraTiny(string? checkpoint = null, int imageSize = 1024)
        {
            return _BuildSam2(
                embedDim: 96,
                numHeads: 1,
                stages: new[] { 1, 2, 7, 2 },
                globalAttBlocks: new[] { 5, 7, 9 },
                windowSpec: new[] { 8, 4, 14, 7 },
                backboneChannelList: new long[] { 768, 384, 192, 96 },
                imageSize: imageSize,
                checkpoint: checkpoint);
        }

        public static Sam2Base BuildSam2HieraSmall(string? checkpoint = null, int imageSize = 1024)
        {
            return _BuildSam2(
                embedDim: 96,
                numHeads: 1,
                stages: new[] { 1, 2, 11, 2 },
                globalAttBlocks: new[] { 7, 10, 13 },
                windowSpec: new[] { 8, 4, 14, 7 },
                backboneChannelList: new long[] { 768, 384, 192, 96 },
                imageSize: imageSize,
                checkpoint: checkpoint);
        }

        public static Sam2Base BuildSam2HieraBasePlus(string? checkpoint = null, int imageSize = 1024)
        {
            return _BuildSam2(
                embedDim: 112,
                numHeads: 2,
                stages: new[] { 2, 3, 16, 3 },
                globalAttBlocks: new[] { 12, 16, 20 },
                windowSpec: new[] { 8, 4, 14, 7 },
                backboneChannelList: new long[] { 896, 448, 224, 112 },
                imageSize: imageSize,
                checkpoint: checkpoint);
        }

        public static Sam2Base BuildSam2HieraLarge(string? checkpoint = null, int imageSize = 1024)
        {
            return _BuildSam2(
                embedDim: 144,
                numHeads: 2,
                stages: new[] { 2, 6, 36, 4 },
                globalAttBlocks: new[] { 23, 33, 43 },
                windowSpec: new[] { 8, 4, 16, 8 },
                backboneChannelList: new long[] { 1152, 576, 288, 144 },
                imageSize: imageSize,
                checkpoint: checkpoint);
        }

        private static readonly Dictionary<string, Func<string?, int, Sam2Base>> Sam2ModelRegistry = new()
        {
            { "tiny", (ckpt, size) => BuildSam2HieraTiny(ckpt, size) },
            { "small", (ckpt, size) => BuildSam2HieraSmall(ckpt, size) },
            { "base_plus", (ckpt, size) => BuildSam2HieraBasePlus(ckpt, size) },
            { "large", (ckpt, size) => BuildSam2HieraLarge(ckpt, size) },
        };

        public static Sam2Base Build(string modelType, string? checkpoint = null, int imageSize = 1024)
        {
            if (Sam2ModelRegistry.TryGetValue(modelType, out var builder))
            {
                return builder(checkpoint, imageSize);
            }
            throw new ArgumentException($"Invalid SAM2 model type: {modelType}");
        }

        private static Sam2Base _BuildSam2(
            long embedDim,
            int numHeads,
            int[] stages,
            int[] globalAttBlocks,
            int[] windowSpec,
            long[] backboneChannelList,
            int imageSize,
            string? checkpoint)
        {
            const long dModel = 256;

            var trunk = new Hiera(
                embedDim: embedDim,
                numHeads: numHeads,
                stages: stages,
                globalAttBlocks: globalAttBlocks,
                windowPosEmbedBkgSpatialSize: (7, 7),
                windowSpec: windowSpec);

            var positionEncoding = new PositionEmbeddingSine(numPosFeats: dModel, normalize: true);
            var neck = new FpnNeck(
                positionEncoding: positionEncoding,
                dModel: dModel,
                backboneChannelList: backboneChannelList,
                fpnTopDownLevels: new[] { 2, 3 },
                fpnInterpModel: "nearest");
            var imageEncoder = new ImageEncoder(trunk, neck, scalp: 1);

            var promptEncoder = new PromptEncoder(
                embed_dim: (int)dModel,
                image_embedding_size: (imageSize / 16, imageSize / 16),
                input_image_size: (imageSize, imageSize),
                mask_in_chans: 16);

            var transformer = new TwoWayTransformer(depth: 2, embeddingDim: dModel, numHeads: 8, mlpDim: 2048);
            var maskDecoder = new MaskDecoder(
                transformerDim: dModel,
                transformer: transformer,
                numMultimaskOutputs: 3,
                iouHeadDepth: 3,
                iouHeadHiddenDim: 256,
                useHighResFeatures: true,
                iouPredictionUseSigmoid: true,
                dynamicMultimaskViaStability: true,
                dynamicMultimaskStabilityDelta: 0.05,
                dynamicMultimaskStabilityThresh: 0.98,
                predObjScores: true,
                predObjScoresMlp: true,
                useMultimaskTokenForObjPtr: true);

            var model = new Sam2Base(
                imageEncoder,
                promptEncoder,
                maskDecoder,
                imageSize: imageSize,
                backboneStride: 16,
                useHighResFeaturesInSam: true,
                directlyAddNoMemEmbed: true);

            if (!string.IsNullOrEmpty(checkpoint))
            {
                string ext = Path.GetExtension(checkpoint);
                // 官方完整 checkpoint 包含 memory_attention/memory_encoder 等本类未实现的模块权重，
                // 因此使用非严格模式加载：仅匹配 image_encoder/sam_prompt_encoder/sam_mask_decoder/no_mem_embed。
                if (ext.Equals(".pth", StringComparison.InvariantCultureIgnoreCase) || ext.Equals(".pt", StringComparison.InvariantCultureIgnoreCase))
                {
                    model.load_py(checkpoint, strict: false);
                }
                else if (ext.Equals(".safetensors", StringComparison.InvariantCultureIgnoreCase))
                {
                    model.load_safetensors(checkpoint, strict: false);
                }
            }

            return model;
        }
    }
}
