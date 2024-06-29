using SAMTorchSharp.Modeling;
using SAMTorchSharp.Modeling.TinyVitSam;
using SAMTorchSharp.Modeling.Transformer;
using TorchSharp;
using TorchSharp.PyBridge;

namespace SAMTorchSharp
{
    public static class BuildSam
    {
        public static Sam BuildSAMVitH(string checkpoint = null)
        {
            return _BuildSAM(
                encoderEmbedDim: 1280,
                encoderDepth: 32,
                encoderNumHeads: 16,
                encoderGlobalAttnIndexes: new[] { 7, 15, 23, 31 },
                checkpoint: checkpoint);
        }

        public static Sam BuildSAMVitL(string checkpoint = null)
        {
            return _BuildSAM(
                encoderEmbedDim: 1024,
                encoderDepth: 24,
                encoderNumHeads: 16,
                encoderGlobalAttnIndexes: new[] { 5, 11, 17, 23 },
                checkpoint: checkpoint);
        }

        public static Sam BuildSAMVitB(string checkpoint = null)
        {
            return _BuildSAM(
                encoderEmbedDim: 768,
                encoderDepth: 12,
                encoderNumHeads: 12,
                encoderGlobalAttnIndexes: new[] { 2, 5, 8, 11 },
                checkpoint: checkpoint);
        }

        public static Sam BuildSAMVitT(string checkpoint = null)
        {
            int promptEmbedDim = 256;
            int imageSize = 1024;
            int vitPatchSize = 16;
            int imageEmbeddingSize = imageSize / vitPatchSize;

            var mobileSam = new Sam(
                image_encoder: new TinyViT(
                    imgSize: 1024,
                    inChans: 3,
                    numClasses: 1000,
                    embedDims: new[] { 64, 128, 160, 320 },
                    depths: new[] { 2, 2, 6, 2 },
                    numHeads: new[] { 2, 4, 5, 10 },
                    windowSizes: new[] { 7, 7, 14, 7 },
                    mlpRatio: 4.0f,
                    dropRate: 0.0f,
                    dropPathRate: 0.0,
                    useCheckpoint: false,
                    mbconvExpandRatio: 4.0f,
                    localConvSize: 3,
                    layerLrDecay: 0.8f),
                prompt_encoder: new PromptEncoder(
                    embed_dim: promptEmbedDim,
                    image_embedding_size: Tuple.Create(imageEmbeddingSize, imageEmbeddingSize),
                    input_image_size: Tuple.Create(imageSize, imageSize),
                    mask_in_chans: 16),
                mask_decoder: new MaskDecoder(
                    numMultimaskOutputs: 3,
                    transformer: new TwoWayTransformer(
                        depth: 2,
                        embeddingDim: promptEmbedDim,
                        mlpDim: 2048,
                        numHeads: 8),
                    transformerDim: promptEmbedDim,
                    iouHeadDepth: 3,
                    iouHeadHiddenDim: 256),
                pixel_mean: new[] { 123.675f, 116.28f, 103.53f },
                pixel_std: new[] { 58.395f, 57.12f, 57.375f });
            //mobileSam.eval();
            if (!string.IsNullOrEmpty(checkpoint))
            {
                var ext = Path.GetExtension(checkpoint);
                if (ext.Equals(".pth",StringComparison.InvariantCultureIgnoreCase)|| ext.Equals(".pt", StringComparison.InvariantCultureIgnoreCase))
                {
                    mobileSam.load_py(checkpoint);
                }
                else if(ext.Equals(".safetensors", StringComparison.InvariantCultureIgnoreCase))
                {
                    mobileSam.load_safetensors(checkpoint);
                }
            }

            return mobileSam;
        }

        private static readonly Dictionary<string, Func<string, Sam>> SamModelRegistry = new Dictionary<string, Func<string, Sam>>
    {
        { "default", BuildSAMVitH },
        { "vit_h", BuildSAMVitH },
        { "vit_l", BuildSAMVitL },
        { "vit_b", BuildSAMVitB },
        { "vit_t", BuildSAMVitT },
    };

        private static Sam BuildSAM(string modelType, string checkpoint = null)
        {
            if (SamModelRegistry.TryGetValue(modelType, out var builder))
            {
                return builder(checkpoint);
            }
            throw new ArgumentException($"Invalid model type: {modelType}");
        }

        private static Sam _BuildSAM(
            int encoderEmbedDim,
            int encoderDepth,
            int encoderNumHeads,
            int[] encoderGlobalAttnIndexes,
            string checkpoint = null)
        {
            int promptEmbedDim = 256;
            int imageSize = 1024;
            int vitPatchSize = 16;
            int imageEmbeddingSize = imageSize / vitPatchSize;

            var sam = new Sam(
                image_encoder: new ImageEncoderViT(
                    depth: encoderDepth,
                    embedDim: encoderEmbedDim,
                    imgSize: imageSize,
                    mlpRatio: 4,
                    normLayer: (x) => torch.nn.LayerNorm(x, eps: 1e-6f),
                    numHeads: encoderNumHeads,
                    patchSize: vitPatchSize,
                    qkvBias: true,
                    useRelPos: true,
                    globalAttnIndexes: encoderGlobalAttnIndexes,
                    windowSize: 14,
                    outChans: promptEmbedDim),
                prompt_encoder: new PromptEncoder(
                    embed_dim: promptEmbedDim,
                    image_embedding_size: Tuple.Create(imageEmbeddingSize, imageEmbeddingSize),
                    input_image_size: Tuple.Create(imageSize, imageSize),
                    mask_in_chans: 16),
                mask_decoder: new MaskDecoder(
                    numMultimaskOutputs: 3,
                    transformer: new TwoWayTransformer(
                        depth: 2,
                        embeddingDim: promptEmbedDim,
                        mlpDim: 2048,
                        numHeads: 8),
                    transformerDim: promptEmbedDim,
                    iouHeadDepth: 3,
                    iouHeadHiddenDim: 256),
                pixel_mean: new[] { 123.675f, 116.28f, 103.53f },
                pixel_std: new[] { 58.395f, 57.12f, 57.375f });

            //sam.eval();
            if (!string.IsNullOrEmpty(checkpoint))
            {
                if (checkpoint.EndsWith(".pth"))
                {
                    sam.load_py(checkpoint);
                }
                else if (checkpoint.EndsWith(".safetensors"))
                {
                    sam.load_safetensors(checkpoint);
                }
            }

            return sam;
        }

    }
}
