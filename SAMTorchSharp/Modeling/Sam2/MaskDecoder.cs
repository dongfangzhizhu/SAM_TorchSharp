using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SAMTorchSharp.Modeling.Sam2
{
    /// <summary>
    /// 对应 sam2/modeling/sam/mask_decoder.py: MaskDecoder。
    /// 相比 SAM1 版本新增：obj_score_token/pred_obj_score_head（物体存在性打分）、
    /// use_high_res_features（融合 conv_s0/conv_s1 高分辨率特征做上采样）、
    /// use_multimask_token_for_obj_ptr、dynamic_multimask_via_stability（单mask输出时按稳定性回退到多mask最佳结果）。
    /// </summary>
    public class MaskDecoder : Module
    {
        public readonly long transformer_dim;
        public readonly TwoWayTransformer transformer;
        public readonly int num_multimask_outputs;
        public readonly Embedding iou_token;
        public readonly int num_mask_tokens;
        public readonly Embedding mask_tokens;
        public readonly bool pred_obj_scores;
        public readonly Embedding? obj_score_token;
        public readonly bool use_multimask_token_for_obj_ptr;
        public readonly Sequential output_upscaling;
        public readonly bool use_high_res_features;
        public readonly Conv2d? conv_s0;
        public readonly Conv2d? conv_s1;
        public readonly ModuleList<MLP> output_hypernetworks_mlps;
        public readonly MLP iou_prediction_head;
        public readonly Module<Tensor, Tensor>? pred_obj_score_head;

        private readonly bool dynamic_multimask_via_stability;
        private readonly double dynamic_multimask_stability_delta;
        private readonly double dynamic_multimask_stability_thresh;

        public MaskDecoder(
            long transformerDim,
            TwoWayTransformer transformer,
            int numMultimaskOutputs = 3,
            Func<Module<Tensor, Tensor>>? activation = null,
            int iouHeadDepth = 3,
            int iouHeadHiddenDim = 256,
            bool useHighResFeatures = false,
            bool iouPredictionUseSigmoid = false,
            bool dynamicMultimaskViaStability = false,
            double dynamicMultimaskStabilityDelta = 0.05,
            double dynamicMultimaskStabilityThresh = 0.98,
            bool predObjScores = false,
            bool predObjScoresMlp = false,
            bool useMultimaskTokenForObjPtr = false,
            string name = "MaskDecoder") : base(name)
        {
            var act = activation ?? (() => GELU());

            transformer_dim = transformerDim;
            this.transformer = transformer;
            num_multimask_outputs = numMultimaskOutputs;

            iou_token = Embedding(1, transformerDim);
            num_mask_tokens = numMultimaskOutputs + 1;
            mask_tokens = Embedding(num_mask_tokens, transformerDim);

            pred_obj_scores = predObjScores;
            if (pred_obj_scores)
            {
                obj_score_token = Embedding(1, transformerDim);
            }
            use_multimask_token_for_obj_ptr = useMultimaskTokenForObjPtr;

            output_upscaling = Sequential(
                ConvTranspose2d(transformerDim, transformerDim / 4, kernelSize: 2, stride: 2),
                new LayerNorm2d(transformerDim / 4),
                act(),
                ConvTranspose2d(transformerDim / 4, transformerDim / 8, kernelSize: 2, stride: 2),
                act());

            use_high_res_features = useHighResFeatures;
            if (use_high_res_features)
            {
                conv_s0 = Conv2d(transformerDim, transformerDim / 8, kernelSize: 1, stride: 1);
                conv_s1 = Conv2d(transformerDim, transformerDim / 4, kernelSize: 1, stride: 1);
            }

            output_hypernetworks_mlps = new ModuleList<MLP>();
            for (int i = 0; i < num_mask_tokens; i++)
            {
                output_hypernetworks_mlps.Add(new MLP((int)transformerDim, (int)transformerDim, (int)(transformerDim / 8), 3));
            }

            iou_prediction_head = new MLP((int)transformerDim, iouHeadHiddenDim, num_mask_tokens, iouHeadDepth, sigmoidOutput: iouPredictionUseSigmoid);

            if (pred_obj_scores)
            {
                if (predObjScoresMlp)
                {
                    pred_obj_score_head = new MLP((int)transformerDim, (int)transformerDim, 1, 3);
                }
                else
                {
                    pred_obj_score_head = Linear(transformerDim, 1);
                }
            }

            dynamic_multimask_via_stability = dynamicMultimaskViaStability;
            dynamic_multimask_stability_delta = dynamicMultimaskStabilityDelta;
            dynamic_multimask_stability_thresh = dynamicMultimaskStabilityThresh;

            RegisterComponents();
        }

        public (Tensor Masks, Tensor IouPred, Tensor SamTokensOut, Tensor ObjectScoreLogits) forward(
            Tensor imageEmbeddings,
            Tensor imagePe,
            Tensor sparsePromptEmbeddings,
            Tensor densePromptEmbeddings,
            bool multimaskOutput,
            bool repeatImage,
            IList<Tensor>? highResFeatures = null)
        {
            var (masks, iouPred, maskTokensOut, objectScoreLogits) = PredictMasks(
                imageEmbeddings, imagePe, sparsePromptEmbeddings, densePromptEmbeddings, repeatImage, highResFeatures);

            if (multimaskOutput)
            {
                masks = masks[TensorIndex.Colon, TensorIndex.Slice(1, null), TensorIndex.Colon, TensorIndex.Colon];
                iouPred = iouPred[TensorIndex.Colon, TensorIndex.Slice(1, null)];
            }
            else if (dynamic_multimask_via_stability && !training)
            {
                (masks, iouPred) = DynamicMultimaskViaStability(masks, iouPred);
            }
            else
            {
                masks = masks[TensorIndex.Colon, TensorIndex.Slice(0, 1), TensorIndex.Colon, TensorIndex.Colon];
                iouPred = iouPred[TensorIndex.Colon, TensorIndex.Slice(0, 1)];
            }

            Tensor samTokensOut;
            if (multimaskOutput && use_multimask_token_for_obj_ptr)
            {
                samTokensOut = maskTokensOut[TensorIndex.Colon, TensorIndex.Slice(1, null)];
            }
            else
            {
                samTokensOut = maskTokensOut[TensorIndex.Colon, TensorIndex.Slice(0, 1)];
            }

            return (masks, iouPred, samTokensOut, objectScoreLogits);
        }

        private (Tensor Masks, Tensor IouPred, Tensor MaskTokensOut, Tensor ObjectScoreLogits) PredictMasks(
            Tensor imageEmbeddings,
            Tensor imagePe,
            Tensor sparsePromptEmbeddings,
            Tensor densePromptEmbeddings,
            bool repeatImage,
            IList<Tensor>? highResFeatures)
        {
            int s = 0;
            Tensor outputTokens;
            if (pred_obj_scores)
            {
                outputTokens = cat(new[] { obj_score_token!.weight!, iou_token.weight!, mask_tokens.weight! }, dim: 0);
                s = 1;
            }
            else
            {
                outputTokens = cat(new[] { iou_token.weight!, mask_tokens.weight! }, dim: 0);
            }
            outputTokens = outputTokens.unsqueeze(0).expand(sparsePromptEmbeddings.size(0), -1, -1);
            Tensor tokens = cat(new[] { outputTokens, sparsePromptEmbeddings }, dim: 1);

            Tensor src;
            if (repeatImage)
            {
                src = imageEmbeddings.repeat_interleave(tokens.shape[0], dim: 0);
            }
            else
            {
                if (imageEmbeddings.shape[0] != tokens.shape[0])
                {
                    throw new ArgumentException("image_embeddings 的 batch 维必须与 tokens 一致");
                }
                src = imageEmbeddings;
            }
            src = src + densePromptEmbeddings;

            if (imagePe.size(0) != 1)
            {
                throw new ArgumentException("image_pe 的 batch 维必须为 1（来自 get_dense_pe()）");
            }
            Tensor posSrc = imagePe.repeat_interleave(tokens.shape[0], dim: 0);

            var shape = src.shape;
            var (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);

            var (hs, transformerSrc) = transformer.forward(src, posSrc, tokens);
            Tensor iouTokenOut = hs[TensorIndex.Colon, TensorIndex.Single(s), TensorIndex.Colon];
            Tensor maskTokensOut = hs[TensorIndex.Colon, TensorIndex.Slice(s + 1, s + 1 + num_mask_tokens), TensorIndex.Colon];

            Tensor srcReshaped = transformerSrc.transpose(1, 2).view(b, c, h, w);

            Tensor upscaledEmbedding;
            if (!use_high_res_features)
            {
                upscaledEmbedding = output_upscaling.forward(srcReshaped);
            }
            else
            {
                if (highResFeatures is null || highResFeatures.Count != 2)
                {
                    throw new ArgumentException("use_high_res_features=true 时必须提供 [feat_s0, feat_s1] 两个高分辨率特征");
                }
                Tensor featS0 = highResFeatures[0];
                Tensor featS1 = highResFeatures[1];

                // output_upscaling = Sequential(dc1, ln1, act1, dc2, act2)
                var modules = output_upscaling.children().ToList();
                var dc1 = (Module<Tensor, Tensor>)modules[0];
                var ln1 = (Module<Tensor, Tensor>)modules[1];
                var act1 = (Module<Tensor, Tensor>)modules[2];
                var dc2 = (Module<Tensor, Tensor>)modules[3];
                var act2 = (Module<Tensor, Tensor>)modules[4];

                upscaledEmbedding = act1.forward(ln1.forward(dc1.forward(srcReshaped) + featS1));
                upscaledEmbedding = act2.forward(dc2.forward(upscaledEmbedding) + featS0);
            }

            var hyperInList = new List<Tensor>();
            for (int i = 0; i < num_mask_tokens; i++)
            {
                hyperInList.Add(output_hypernetworks_mlps[i].forward(maskTokensOut[TensorIndex.Colon, TensorIndex.Single(i), TensorIndex.Colon]));
            }
            Tensor hyperIn = stack(hyperInList, dim: 1);

            var upShape = upscaledEmbedding.shape;
            var (ub, uc, uh, uw) = (upShape[0], upShape[1], upShape[2], upShape[3]);
            Tensor masks = hyperIn.matmul(upscaledEmbedding.view(ub, uc, uh * uw)).view(ub, -1, uh, uw);

            Tensor iouPred = iou_prediction_head.forward(iouTokenOut);

            Tensor objectScoreLogits;
            if (pred_obj_scores)
            {
                objectScoreLogits = pred_obj_score_head!.forward(hs[TensorIndex.Colon, TensorIndex.Single(0), TensorIndex.Colon]);
            }
            else
            {
                objectScoreLogits = 10.0 * iouPred.new_ones(iouPred.shape[0], 1);
            }

            return (masks, iouPred, maskTokensOut, objectScoreLogits);
        }

        private Tensor GetStabilityScores(Tensor maskLogits)
        {
            maskLogits = maskLogits.flatten(-2);
            double delta = dynamic_multimask_stability_delta;
            Tensor areaI = (maskLogits > delta).sum(new long[] { -1 }).to_type(ScalarType.Float32);
            Tensor areaU = (maskLogits > -delta).sum(new long[] { -1 }).to_type(ScalarType.Float32);
            Tensor stabilityScores = where(areaU > 0, areaI / areaU, ones_like(areaI));
            return stabilityScores;
        }

        private (Tensor, Tensor) DynamicMultimaskViaStability(Tensor allMaskLogits, Tensor allIouScores)
        {
            Tensor multimaskLogits = allMaskLogits[TensorIndex.Colon, TensorIndex.Slice(1, null), TensorIndex.Colon, TensorIndex.Colon];
            Tensor multimaskIouScores = allIouScores[TensorIndex.Colon, TensorIndex.Slice(1, null)];
            Tensor bestScoresInds = multimaskIouScores.argmax(dim: -1);
            Tensor batchInds = arange(multimaskIouScores.size(0), device: allIouScores.device);

            Tensor bestMultimaskLogits = multimaskLogits.index(new TensorIndex[] { TensorIndex.Tensor(batchInds), TensorIndex.Tensor(bestScoresInds) });
            bestMultimaskLogits = bestMultimaskLogits.unsqueeze(1);
            Tensor bestMultimaskIouScores = multimaskIouScores.index(new TensorIndex[] { TensorIndex.Tensor(batchInds), TensorIndex.Tensor(bestScoresInds) });
            bestMultimaskIouScores = bestMultimaskIouScores.unsqueeze(1);

            Tensor singlemaskLogits = allMaskLogits[TensorIndex.Colon, TensorIndex.Slice(0, 1), TensorIndex.Colon, TensorIndex.Colon];
            Tensor singlemaskIouScores = allIouScores[TensorIndex.Colon, TensorIndex.Slice(0, 1)];
            Tensor stabilityScores = GetStabilityScores(singlemaskLogits);
            Tensor isStable = stabilityScores >= dynamic_multimask_stability_thresh;

            Tensor isStableExpandedLogits = isStable[TensorIndex.Ellipsis, TensorIndex.None, TensorIndex.None].expand(singlemaskLogits.shape);
            Tensor maskLogitsOut = where(isStableExpandedLogits, singlemaskLogits, bestMultimaskLogits);

            Tensor isStableExpandedIou = isStable.expand(singlemaskIouScores.shape);
            Tensor iouScoresOut = where(isStableExpandedIou, singlemaskIouScores, bestMultimaskIouScores);

            return (maskLogitsOut, iouScoresOut);
        }
    }
}
