using SAMTorchSharp.Modeling.Sam2;
using TorchSharp;

namespace SAMTorchSharp
{
    /// <summary>
    /// SAM2 完整推理管线（图片 + 视频）的构建器，对应 Python 端 sam2/build_sam.py 中
    /// build_sam2 + sam2.1_hiera_*.yaml 配置。包含 image_encoder、sam_prompt_encoder、
    /// sam_mask_decoder、memory_attention、memory_encoder，可直接从官方完整 checkpoint 严格加载权重。
    /// </summary>
    public static class BuildSam2
    {
        public static Sam2Base BuildSam2HieraTiny(string? checkpoint = null, int imageSize = 1024)
            => BuildRegistered(Sam2ModelVariant.Sam21Tiny, checkpoint, imageSize);

        public static Sam2Base BuildSam2HieraSmall(string? checkpoint = null, int imageSize = 1024)
            => BuildRegistered(Sam2ModelVariant.Sam21Small, checkpoint, imageSize);

        public static Sam2Base BuildSam2Tiny(string? checkpoint = null, int imageSize = 1024)
            => BuildRegistered(Sam2ModelVariant.Sam2Tiny, checkpoint, imageSize);

        public static Sam2Base BuildSam2Small(string? checkpoint = null, int imageSize = 1024)
            => BuildRegistered(Sam2ModelVariant.Sam2Small, checkpoint, imageSize);

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

        private static readonly Dictionary<string, Func<string?, int, Sam2Base>> Builders = new(StringComparer.OrdinalIgnoreCase)
        {
            { "tiny", (ckpt, size) => BuildSam2HieraTiny(ckpt, size) },
            { "small", (ckpt, size) => BuildSam2HieraSmall(ckpt, size) },
            { "sam2-tiny", (ckpt, size) => BuildSam2Tiny(ckpt, size) },
            { "sam2-small", (ckpt, size) => BuildSam2Small(ckpt, size) },
            { "sam2.1-tiny", (ckpt, size) => BuildSam2HieraTiny(ckpt, size) },
            { "sam2.1-small", (ckpt, size) => BuildSam2HieraSmall(ckpt, size) },
            { "base_plus", (ckpt, size) => BuildSam2HieraBasePlus(ckpt, size) },
            { "large", (ckpt, size) => BuildSam2HieraLarge(ckpt, size) },
        };

        public static Sam2Base Build(string modelType, string? checkpoint = null, int imageSize = 1024)
        {
            if (Builders.TryGetValue(modelType, out var builder))
            {
                return builder(checkpoint, imageSize);
            }
            throw new ArgumentException($"Invalid SAM2 model type: {modelType}");
        }

        private static Sam2Base BuildRegistered(Sam2ModelVariant variant, string? checkpoint, int imageSize)
        {
            var options = Sam2ModelRegistry.Get(variant) with { ImageSize = imageSize };
            var model = Sam2ModelBuilder.Build(options);
            if (!string.IsNullOrWhiteSpace(checkpoint))
                Sam2CheckpointLoader.Load(model, checkpoint, strict: true);
            return model;
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
            const long memDim = 64;

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

            // memory_attention: 4 层，self_attn/cross_attn_image 均为 RoPEAttention
            // (cross_attn_image 的 kv_in_dim=mem_dim=64，对应 memory_encoder.out_dim)
            var selfAttn = new RoPEAttention(embeddingDim: dModel, numHeads: 1, downsampleRate: 1, ropeTheta: 10000.0, featSizes: (64, 64));
            var crossAttn = new RoPEAttention(
                embeddingDim: dModel, numHeads: 1, downsampleRate: 1, ropeTheta: 10000.0,
                featSizes: (64, 64), ropeKRepeat: true, kvInDim: memDim);
            var memAttnLayer = new MemoryAttentionLayer(
                activation: "relu", crossAttention: crossAttn, dModel: dModel, dimFeedforward: 2048, dropout: 0.1,
                posEncAtAttn: false, posEncAtCrossAttnKeys: true, posEncAtCrossAttnQueries: false, selfAttention: selfAttn);
            var memoryAttention = new MemoryAttention(dModel: dModel, posEncAtInput: true, layer: memAttnLayer, numLayers: 4);

            // memory_encoder: mask_downsampler(stride16=4层) + fuser(2层CXBlock) + out_proj(256->64)
            var maskDownsampler = new MaskDownSampler(kernelSize: 3, stride: 2, padding: 1);
            var fuser = new Fuser(
                layer: new CXBlock(dim: dModel, kernelSize: 7, padding: 3, layerScaleInitValue: 1e-6, useDwconv: true),
                numLayers: 2);
            var memPositionEncoding = new PositionEmbeddingSine(numPosFeats: memDim, normalize: true);
            var memoryEncoder = new MemoryEncoder(outDim: memDim, maskDownsampler: maskDownsampler, fuser: fuser, positionEncoding: memPositionEncoding);

            var model = new Sam2Base(
                imageEncoder,
                promptEncoder,
                maskDecoder,
                memoryAttention: memoryAttention,
                memoryEncoder: memoryEncoder,
                numMaskmem: 7,
                imageSize: imageSize,
                backboneStride: 16,
                sigmoidScaleForMemEnc: 20.0,
                sigmoidBiasForMemEnc: -10.0,
                useMaskInputAsOutputWithoutSam: true,
                directlyAddNoMemEmbed: true,
                useHighResFeaturesInSam: true,
                multimaskOutputInSam: true,
                multimaskMinPtNum: 0,
                multimaskMaxPtNum: 1,
                multimaskOutputForTracking: true,
                useObjPtrsInEncoder: true,
                addTposEncToObjPtrs: true,
                projTposEncInObjPtrs: true,
                useSignedTposEncToObjPtrs: true,
                onlyObjPtrsInThePastForEval: true,
                predObjScores: true,
                fixedNoObjPtr: true,
                useMlpForObjPtrProj: true,
                noObjEmbedSpatial: true);

            if (!string.IsNullOrEmpty(checkpoint))
            {
                string ext = Path.GetExtension(checkpoint);
                if (ext.Equals(".pth", StringComparison.InvariantCultureIgnoreCase) || ext.Equals(".pt", StringComparison.InvariantCultureIgnoreCase))
                {
                    throw new NotSupportedException("Legacy base_plus/large .pt loading is not supported. Use a safetensors state dictionary.");
                }
                else if (ext.Equals(".safetensors", StringComparison.InvariantCultureIgnoreCase))
                {
                    Sam2CheckpointLoader.Load(model, checkpoint, strict: true);
                }
            }

            return model;
        }
    }
}
