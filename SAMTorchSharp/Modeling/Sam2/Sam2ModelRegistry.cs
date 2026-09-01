namespace SAMTorchSharp.Modeling.Sam2;

public enum Sam2ModelVariant
{
    Sam2Tiny,
    Sam2Small,
    Sam21Tiny,
    Sam21Small,
}

public sealed record Sam2ModelOptions(
    Sam2ModelVariant Variant,
    string Name,
    int ImageSize,
    int[] Stages,
    int[] GlobalAttentionBlocks,
    bool AddTemporalPositionEncodingToObjectPointers,
    bool ProjectTemporalPositionEncodingInObjectPointers,
    bool UseSignedTemporalPositionEncodingForObjectPointers,
    bool UseNoObjectSpatialEmbedding);

public static class Sam2ModelRegistry
{
    public static IReadOnlyList<Sam2ModelVariant> SupportedVariants { get; } =
        Enum.GetValues<Sam2ModelVariant>();

    public static Sam2ModelOptions Get(Sam2ModelVariant variant)
    {
        var isSmall = variant is Sam2ModelVariant.Sam2Small or Sam2ModelVariant.Sam21Small;
        var isSam21 = variant is Sam2ModelVariant.Sam21Tiny or Sam2ModelVariant.Sam21Small;
        if (!Enum.IsDefined(variant))
            throw new ArgumentOutOfRangeException(nameof(variant), variant, "Unsupported SAM2 model variant.");

        return new Sam2ModelOptions(
            variant,
            variant switch
            {
                Sam2ModelVariant.Sam2Tiny => "sam2_hiera_tiny",
                Sam2ModelVariant.Sam2Small => "sam2_hiera_small",
                Sam2ModelVariant.Sam21Tiny => "sam2.1_hiera_tiny",
                Sam2ModelVariant.Sam21Small => "sam2.1_hiera_small",
                _ => throw new ArgumentOutOfRangeException(nameof(variant), variant, null),
            },
            ImageSize: 1024,
            Stages: isSmall ? [1, 2, 11, 2] : [1, 2, 7, 2],
            GlobalAttentionBlocks: isSmall ? [7, 10, 13] : [5, 7, 9],
            AddTemporalPositionEncodingToObjectPointers: isSam21,
            ProjectTemporalPositionEncodingInObjectPointers: isSam21,
            UseSignedTemporalPositionEncodingForObjectPointers: isSam21,
            UseNoObjectSpatialEmbedding: isSam21);
    }
}

public static class Sam2ModelBuilder
{
    public static Sam2Base Build(Sam2ModelVariant variant) => Build(Sam2ModelRegistry.Get(variant));

    public static Sam2Base Build(Sam2ModelOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        const int dModel = 256;
        const int memoryDimension = 64;

        var trunk = new Hiera(
            embedDim: 96,
            numHeads: 1,
            stages: options.Stages,
            globalAttBlocks: options.GlobalAttentionBlocks,
            windowPosEmbedBkgSpatialSize: (7, 7),
            windowSpec: [8, 4, 14, 7]);
        var imageEncoder = new ImageEncoder(
            trunk,
            new FpnNeck(
                new PositionEmbeddingSine(numPosFeats: dModel, normalize: true),
                dModel,
                trunk.ChannelList,
                fpnTopDownLevels: [2, 3],
                fpnInterpModel: "nearest"),
            scalp: 1);
        var promptEncoder = new PromptEncoder(
            embed_dim: dModel,
            image_embedding_size: (options.ImageSize / 16, options.ImageSize / 16),
            input_image_size: (options.ImageSize, options.ImageSize),
            mask_in_chans: 16);
        var maskDecoder = new MaskDecoder(
            transformerDim: dModel,
            transformer: new TwoWayTransformer(depth: 2, embeddingDim: dModel, numHeads: 8, mlpDim: 2048),
            numMultimaskOutputs: 3,
            iouHeadDepth: 3,
            iouHeadHiddenDim: 256,
            useHighResFeatures: true,
            iouPredictionUseSigmoid: true,
            dynamicMultimaskViaStability: true,
            predObjScores: true,
            predObjScoresMlp: true,
            useMultimaskTokenForObjPtr: true);

        var selfAttention = new RoPEAttention(dModel, 1, 1, ropeTheta: 10000, featSizes: (64, 64));
        var crossAttention = new RoPEAttention(
            dModel, 1, 1, kvInDim: memoryDimension, ropeTheta: 10000, featSizes: (64, 64), ropeKRepeat: true);
        var memoryAttention = new MemoryAttention(
            dModel,
            posEncAtInput: true,
            new MemoryAttentionLayer(
                "relu", crossAttention, dModel, 2048, 0.1,
                posEncAtAttn: false,
                posEncAtCrossAttnKeys: true,
                posEncAtCrossAttnQueries: false,
                selfAttention),
            numLayers: 4);
        var memoryEncoder = new MemoryEncoder(
            memoryDimension,
            new MaskDownSampler(kernelSize: 3, stride: 2, padding: 1),
            new Fuser(new CXBlock(dModel, 7, 3, layerScaleInitValue: 1e-6, useDwconv: true), 2),
            new PositionEmbeddingSine(numPosFeats: memoryDimension, normalize: true));

        var model = new Sam2Base(
            imageEncoder,
            promptEncoder,
            maskDecoder,
            memoryAttention,
            memoryEncoder,
            numMaskmem: 7,
            imageSize: options.ImageSize,
            backboneStride: 16,
            sigmoidScaleForMemEnc: 20,
            sigmoidBiasForMemEnc: -10,
            binarizeMaskFromPointsForMemoryEncoder: true,
            useMaskInputAsOutputWithoutSam: true,
            directlyAddNoMemEmbed: true,
            useHighResFeaturesInSam: true,
            multimaskOutputInSam: true,
            multimaskMinPtNum: 0,
            multimaskMaxPtNum: 1,
            multimaskOutputForTracking: true,
            useObjPtrsInEncoder: true,
            addTposEncToObjPtrs: options.AddTemporalPositionEncodingToObjectPointers,
            projTposEncInObjPtrs: options.ProjectTemporalPositionEncodingInObjectPointers,
            useSignedTposEncToObjPtrs: options.UseSignedTemporalPositionEncodingForObjectPointers,
            onlyObjPtrsInThePastForEval: true,
            predObjScores: true,
            fixedNoObjPtr: true,
            useMlpForObjPtrProj: true,
            noObjEmbedSpatial: options.UseNoObjectSpatialEmbedding,
            name: options.Name);
        model.eval();
        return model;
    }
}