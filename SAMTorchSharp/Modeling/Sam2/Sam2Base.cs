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
    /// 对应 Python 端 point_inputs 字典：{"point_coords": [B,P,2], "point_labels": [B,P]}。
    /// </summary>
    public class PointInputs
    {
        public Tensor PointCoords { get; init; } = null!;
        public Tensor PointLabels { get; init; } = null!;
    }

    /// <summary>
    /// 对应 Python 端 SAM2Base._forward_sam_heads / _use_mask_as_output 的返回元组
    /// (low_res_multimasks, high_res_multimasks, ious, low_res_masks, high_res_masks, obj_ptr, object_score_logits)。
    /// </summary>
    public class SamHeadsOutput
    {
        public Tensor LowResMultimasks { get; init; } = null!;
        public Tensor HighResMultimasks { get; init; } = null!;
        public Tensor Ious { get; init; } = null!;
        public Tensor LowResMasks { get; init; } = null!;
        public Tensor HighResMasks { get; init; } = null!;
        public Tensor ObjPtr { get; init; } = null!;
        public Tensor ObjectScoreLogits { get; init; } = null!;
    }

    /// <summary>
    /// 对应 Python 端 output_dict 中每帧的记录（cond_frame_outputs / non_cond_frame_outputs 的 value）。
    /// </summary>
    public class FrameOutput
    {
        public Tensor? MaskmemFeatures { get; set; }
        public IList<Tensor>? MaskmemPosEnc { get; set; }
        public Tensor ObjPtr { get; set; } = null!;
        public Tensor PredMasks { get; set; } = null!;
        public Tensor PredMasksHighRes { get; set; } = null!;
        public Tensor ObjectScoreLogits { get; set; } = null!;
    }

    /// <summary>
    /// 对应 Python 端 output_dict：{"cond_frame_outputs": {frame_idx: FrameOutput}, "non_cond_frame_outputs": {...}}。
    /// </summary>
    public class VideoOutputDict
    {
        public Dictionary<int, FrameOutput> CondFrameOutputs { get; } = new();
        public Dictionary<int, FrameOutput> NonCondFrameOutputs { get; } = new();
    }

    /// <summary>
    /// 对应 sam2/modeling/sam2_base.py: SAM2Base。
    /// 包含图片推理（Phase1）与视频推理（Phase2：memory_attention/memory_encoder/track_step）的完整逻辑。
    /// 字段命名与 Python 端保持一致，便于从完整官方 checkpoint 直接加载权重。
    /// </summary>
    public class Sam2Base : Module
    {
        private const double NoObjScore = -1024.0;

        public readonly ImageEncoder image_encoder;
        public readonly PromptEncoder sam_prompt_encoder;
        public readonly MaskDecoder sam_mask_decoder;
        public readonly MemoryAttention? memory_attention;
        public readonly MemoryEncoder? memory_encoder;
        public readonly Conv2d? mask_downsample;
        public readonly Module<Tensor, Tensor> obj_ptr_proj;
        public readonly Module<Tensor, Tensor> obj_ptr_tpos_proj;
        public readonly Parameter maskmem_tpos_enc;
        public readonly Parameter no_mem_embed;
        public readonly Parameter no_mem_pos_enc;
        public readonly Parameter? no_obj_ptr;
        public readonly Parameter? no_obj_embed_spatial;

        public readonly long hidden_dim;
        public readonly long mem_dim;
        public readonly int image_size;
        public readonly int backbone_stride;
        public readonly bool binarize_mask_from_pts_for_mem_enc;
        public readonly bool use_high_res_features_in_sam;
        public readonly bool directly_add_no_mem_embed;
        public readonly int num_feature_levels;
        public readonly int num_maskmem;

        public readonly bool use_obj_ptrs_in_encoder;
        public readonly int max_obj_ptrs_in_encoder;
        public readonly bool add_tpos_enc_to_obj_ptrs;
        public readonly bool proj_tpos_enc_in_obj_ptrs;
        public readonly bool use_signed_tpos_enc_to_obj_ptrs;
        public readonly bool only_obj_ptrs_in_the_past_for_eval;
        public readonly bool pred_obj_scores;
        public readonly bool fixed_no_obj_ptr;
        public readonly bool soft_no_obj_ptr;
        public readonly double sigmoid_scale_for_mem_enc;
        public readonly double sigmoid_bias_for_mem_enc;
        public readonly bool use_mask_input_as_output_without_sam;
        public readonly bool multimask_output_in_sam;
        public readonly int multimask_min_pt_num;
        public readonly int multimask_max_pt_num;
        public readonly bool multimask_output_for_tracking;
        public readonly int max_cond_frames_in_attn;
        public readonly int memory_temporal_stride_for_eval;

        public Sam2Base(
            ImageEncoder imageEncoder,
            PromptEncoder samPromptEncoder,
            MaskDecoder samMaskDecoder,
            MemoryAttention? memoryAttention = null,
            MemoryEncoder? memoryEncoder = null,
            int numMaskmem = 7,
            int imageSize = 1024,
            int backboneStride = 16,
            double sigmoidScaleForMemEnc = 1.0,
            double sigmoidBiasForMemEnc = 0.0,
            bool binarizeMaskFromPointsForMemoryEncoder = false,
            bool useMaskInputAsOutputWithoutSam = false,
            int maxCondFramesInAttn = -1,
            bool directlyAddNoMemEmbed = false,
            bool useHighResFeaturesInSam = false,
            bool multimaskOutputInSam = false,
            int multimaskMinPtNum = 1,
            int multimaskMaxPtNum = 1,
            bool multimaskOutputForTracking = false,
            int memoryTemporalStrideForEval = 1,
            bool useObjPtrsInEncoder = false,
            int maxObjPtrsInEncoder = 16,
            bool addTposEncToObjPtrs = true,
            bool projTposEncInObjPtrs = false,
            bool useSignedTposEncToObjPtrs = false,
            bool onlyObjPtrsInThePastForEval = false,
            bool predObjScores = false,
            bool fixedNoObjPtr = false,
            bool softNoObjPtr = false,
            bool useMlpForObjPtrProj = false,
            bool noObjEmbedSpatial = false,
            string name = "Sam2Base") : base(name)
        {
            image_encoder = imageEncoder;
            use_high_res_features_in_sam = useHighResFeaturesInSam;
            num_feature_levels = useHighResFeaturesInSam ? 3 : 1;
            use_obj_ptrs_in_encoder = useObjPtrsInEncoder;
            max_obj_ptrs_in_encoder = maxObjPtrsInEncoder;
            if (useObjPtrsInEncoder)
            {
                mask_downsample = Conv2d(1, 1, kernelSize: 4, stride: 4);
            }
            add_tpos_enc_to_obj_ptrs = addTposEncToObjPtrs;
            proj_tpos_enc_in_obj_ptrs = projTposEncInObjPtrs;
            use_signed_tpos_enc_to_obj_ptrs = useSignedTposEncToObjPtrs;
            only_obj_ptrs_in_the_past_for_eval = onlyObjPtrsInThePastForEval;

            memory_attention = memoryAttention;
            hidden_dim = imageEncoder.neck.d_model;

            memory_encoder = memoryEncoder;
            mem_dim = hidden_dim;
            if (memoryEncoder is not null && memoryEncoder.out_proj is Conv2d outProjConv)
            {
                mem_dim = outProjConv.weight!.shape[0];
            }
            num_maskmem = numMaskmem;
            maskmem_tpos_enc = Parameter(zeros(numMaskmem, 1, 1, mem_dim));
            init.trunc_normal_(maskmem_tpos_enc, std: 0.02);

            no_mem_embed = Parameter(zeros(1, 1, hidden_dim));
            no_mem_pos_enc = Parameter(zeros(1, 1, hidden_dim));
            init.trunc_normal_(no_mem_embed, std: 0.02);
            init.trunc_normal_(no_mem_pos_enc, std: 0.02);
            directly_add_no_mem_embed = directlyAddNoMemEmbed;

            sigmoid_scale_for_mem_enc = sigmoidScaleForMemEnc;
            sigmoid_bias_for_mem_enc = sigmoidBiasForMemEnc;
            binarize_mask_from_pts_for_mem_enc = binarizeMaskFromPointsForMemoryEncoder;
            memory_temporal_stride_for_eval = memoryTemporalStrideForEval;
            use_mask_input_as_output_without_sam = useMaskInputAsOutputWithoutSam;
            multimask_output_in_sam = multimaskOutputInSam;
            multimask_min_pt_num = multimaskMinPtNum;
            multimask_max_pt_num = multimaskMaxPtNum;
            multimask_output_for_tracking = multimaskOutputForTracking;

            image_size = imageSize;
            backbone_stride = backboneStride;
            pred_obj_scores = predObjScores;
            fixed_no_obj_ptr = fixedNoObjPtr;
            soft_no_obj_ptr = softNoObjPtr;
            if (pred_obj_scores && use_obj_ptrs_in_encoder)
            {
                no_obj_ptr = Parameter(zeros(1, hidden_dim));
                init.trunc_normal_(no_obj_ptr, std: 0.02);
            }
            if (noObjEmbedSpatial)
            {
                no_obj_embed_spatial = Parameter(zeros(1, mem_dim));
                init.trunc_normal_(no_obj_embed_spatial, std: 0.02);
            }

            sam_prompt_encoder = samPromptEncoder;
            sam_mask_decoder = samMaskDecoder;

            if (use_obj_ptrs_in_encoder)
            {
                obj_ptr_proj = useMlpForObjPtrProj
                    ? new MLP((int)hidden_dim, (int)hidden_dim, (int)hidden_dim, 3)
                    : Linear(hidden_dim, hidden_dim);
            }
            else
            {
                obj_ptr_proj = Identity();
            }

            if (proj_tpos_enc_in_obj_ptrs)
            {
                obj_ptr_tpos_proj = Linear(hidden_dim, mem_dim);
            }
            else
            {
                obj_ptr_tpos_proj = Identity();
            }

            max_cond_frames_in_attn = maxCondFramesInAttn;

            RegisterComponents();
        }

        // ------------------------------------------------------------------
        // Phase1: 图片推理相关方法
        // ------------------------------------------------------------------

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

        public (IList<Tensor> VisionFeats, IList<Tensor> VisionPosEmbeds, IList<(long H, long W)> FeatSizes) PrepareBackboneFeatures(BackboneOut backboneOut)
        {
            int n = backboneOut.BackboneFpn.Count;
            var featureMaps = backboneOut.BackboneFpn.Skip(n - num_feature_levels).ToList();
            var visionPosEmbeds = backboneOut.VisionPosEnc.Skip(backboneOut.VisionPosEnc.Count - num_feature_levels).ToList();

            var featSizes = visionPosEmbeds.Select(x => (x.shape[^2], x.shape[^1])).ToList();
            var visionFeats = featureMaps.Select(x => x.flatten(2).permute(2, 0, 1)).ToList();
            var visionPosFlat = visionPosEmbeds.Select(x => x.flatten(2).permute(2, 0, 1)).ToList();

            return (visionFeats, visionPosFlat, featSizes);
        }

        // ------------------------------------------------------------------
        // Phase2: 视频推理相关方法
        // ------------------------------------------------------------------

        /// <summary>
        /// 对应 Python 端 _forward_sam_heads。
        /// backboneFeatures: [B, C, H, W]（融合memory后的特征，H=W=sam_image_embedding_size）。
        /// </summary>
        public SamHeadsOutput ForwardSamHeads(
            Tensor backboneFeatures,
            PointInputs? pointInputs,
            Tensor? maskInputs,
            IList<Tensor>? highResFeatures,
            bool multimaskOutput)
        {
            long B = backboneFeatures.size(0);
            Device device = backboneFeatures.device;

            Tensor samPointCoords, samPointLabels;
            if (pointInputs is not null)
            {
                samPointCoords = pointInputs.PointCoords;
                samPointLabels = pointInputs.PointLabels;
            }
            else
            {
                samPointCoords = zeros(B, 1, 2, device: device);
                samPointLabels = -ones(new long[] { B, 1 }, ScalarType.Int32, device: device);
            }

            Tensor? samMaskPrompt = null;
            if (maskInputs is not null)
            {
                var maskInputSize = sam_prompt_encoder.mask_input_size;
                if (maskInputs.shape[^2] != maskInputSize.H || maskInputs.shape[^1] != maskInputSize.W)
                {
                    // 注：Python 端此处使用 antialias=True 做下采样；TorchSharp 的 interpolate
                    // 未暴露 antialias 参数，这里用标准双线性插值近似。仅在 mask_inputs 分辨率与
                    // mask_input_size 不一致时才会走到这个分支（大多数推理路径不会触发）。
                    samMaskPrompt = functional.interpolate(
                        maskInputs.to(ScalarType.Float32),
                        size: new long[] { maskInputSize.H, maskInputSize.W },
                        mode: InterpolationMode.Bilinear,
                        align_corners: false);
                }
                else
                {
                    samMaskPrompt = maskInputs;
                }
            }

            var (sparseEmbeddings, denseEmbeddings) = sam_prompt_encoder.forward(
                Tuple.Create(samPointCoords, samPointLabels), null, samMaskPrompt);

            var (lowResMultimasksRaw, ious, samOutputTokens, objectScoreLogits) = sam_mask_decoder.forward(
                backboneFeatures,
                sam_prompt_encoder.get_dense_pe(),
                sparseEmbeddings,
                denseEmbeddings,
                multimaskOutput,
                false,
                highResFeatures);

            Tensor lowResMultimasks = lowResMultimasksRaw;
            if (pred_obj_scores)
            {
                Tensor isObjAppearing = objectScoreLogits > 0;
                Tensor mask3d = isObjAppearing[TensorIndex.Colon, TensorIndex.None, TensorIndex.None];
                lowResMultimasks = where(mask3d, lowResMultimasks, full_like(lowResMultimasks, NoObjScore));
            }

            lowResMultimasks = lowResMultimasks.to(ScalarType.Float32);
            Tensor highResMultimasks = functional.interpolate(
                lowResMultimasks,
                size: new long[] { image_size, image_size },
                mode: InterpolationMode.Bilinear,
                align_corners: false);

            Tensor samOutputToken = samOutputTokens[TensorIndex.Colon, TensorIndex.Single(0), TensorIndex.Colon];
            Tensor lowResMasks, highResMasks;
            if (multimaskOutput)
            {
                Tensor bestIouInds = ious.argmax(dim: -1);
                Tensor batchInds = arange(B, device: device);
                lowResMasks = lowResMultimasks.index(new TensorIndex[] { TensorIndex.Tensor(batchInds), TensorIndex.Tensor(bestIouInds) }).unsqueeze(1);
                highResMasks = highResMultimasks.index(new TensorIndex[] { TensorIndex.Tensor(batchInds), TensorIndex.Tensor(bestIouInds) }).unsqueeze(1);
                if (samOutputTokens.shape[1] > 1)
                {
                    samOutputToken = samOutputTokens.index(new TensorIndex[] { TensorIndex.Tensor(batchInds), TensorIndex.Tensor(bestIouInds) });
                }
            }
            else
            {
                lowResMasks = lowResMultimasks;
                highResMasks = highResMultimasks;
            }

            Tensor objPtr = obj_ptr_proj.forward(samOutputToken);
            if (pred_obj_scores)
            {
                Tensor isObjAppearingF = (objectScoreLogits > 0).to(ScalarType.Float32);
                Tensor lambdaIsObjAppearing = soft_no_obj_ptr ? sigmoid(objectScoreLogits) : isObjAppearingF;

                if (fixed_no_obj_ptr)
                {
                    objPtr = lambdaIsObjAppearing * objPtr;
                }
                objPtr = objPtr + (1 - lambdaIsObjAppearing) * no_obj_ptr!;
            }

            return new SamHeadsOutput
            {
                LowResMultimasks = lowResMultimasks,
                HighResMultimasks = highResMultimasks,
                Ious = ious,
                LowResMasks = lowResMasks,
                HighResMasks = highResMasks,
                ObjPtr = objPtr,
                ObjectScoreLogits = objectScoreLogits,
            };
        }

        /// <summary>对应 Python 端 _use_mask_as_output：直接把输入 mask 当作输出，不经过 SAM 头。</summary>
        public SamHeadsOutput UseMaskAsOutput(Tensor backboneFeatures, IList<Tensor>? highResFeatures, Tensor maskInputs)
        {
            const double outScale = 20.0, outBias = -10.0;
            Tensor maskInputsFloat = maskInputs.to(ScalarType.Float32);
            Tensor highResMasks = maskInputsFloat * outScale + outBias;
            // 注：Python 端 antialias=True，TorchSharp 未暴露该参数，这里用标准双线性插值近似。
            Tensor lowResMasks = functional.interpolate(
                highResMasks,
                size: new long[] { highResMasks.shape[^2] / 4, highResMasks.shape[^1] / 4 },
                mode: InterpolationMode.Bilinear,
                align_corners: false);

            Tensor ious = maskInputs.new_ones(maskInputs.shape[0], 1).to(ScalarType.Float32);

            Tensor objPtr;
            if (!use_obj_ptrs_in_encoder)
            {
                objPtr = zeros(maskInputs.shape[0], hidden_dim, device: maskInputs.device);
            }
            else
            {
                Tensor downsampled = mask_downsample!.forward(maskInputsFloat);
                var samOut = ForwardSamHeads(backboneFeatures, null, downsampled, highResFeatures, false);
                objPtr = samOut.ObjPtr;
            }

            Tensor isObjAppearing = (maskInputs.flatten(1).to(ScalarType.Float32) > 0.0).any(dim: 1);
            isObjAppearing = isObjAppearing[TensorIndex.Ellipsis, TensorIndex.None];
            Tensor lambdaIsObjAppearing = isObjAppearing.to(ScalarType.Float32);
            Tensor objectScoreLogits = outScale * lambdaIsObjAppearing + outBias;

            if (pred_obj_scores)
            {
                if (fixed_no_obj_ptr)
                {
                    objPtr = lambdaIsObjAppearing * objPtr;
                }
                objPtr = objPtr + (1 - lambdaIsObjAppearing) * no_obj_ptr!;
            }

            return new SamHeadsOutput
            {
                LowResMultimasks = lowResMasks,
                HighResMultimasks = highResMasks,
                Ious = ious,
                LowResMasks = lowResMasks,
                HighResMasks = highResMasks,
                ObjPtr = objPtr,
                ObjectScoreLogits = objectScoreLogits,
            };
        }

        /// <summary>对应 Python 端 _use_multimask。</summary>
        public bool UseMultimask(bool isInitCondFrame, PointInputs? pointInputs)
        {
            long numPts = pointInputs is null ? 0 : pointInputs.PointLabels.shape[1];
            return multimask_output_in_sam
                && (isInitCondFrame || multimask_output_for_tracking)
                && (multimask_min_pt_num <= numPts && numPts <= multimask_max_pt_num);
        }

        /// <summary>
        /// 对应 Python 端 select_closest_cond_frames。max_cond_frames_in_attn=-1（默认/最常见配置）
        /// 时直接返回全部条件帧，不做筛选。仅在数量超过阈值时才做时间最近筛选。
        /// </summary>
        private (Dictionary<int, FrameOutput> Selected, Dictionary<int, FrameOutput> Unselected) SelectClosestCondFrames(
            int frameIdx, Dictionary<int, FrameOutput> condFrameOutputs, int maxCondFrameNum)
        {
            if (maxCondFrameNum == -1 || condFrameOutputs.Count <= maxCondFrameNum)
            {
                return (condFrameOutputs, new Dictionary<int, FrameOutput>());
            }

            var selected = new Dictionary<int, FrameOutput>();
            var before = condFrameOutputs.Keys.Where(t => t < frameIdx).DefaultIfEmpty(int.MinValue).Max();
            if (before != int.MinValue) selected[before] = condFrameOutputs[before];

            var after = condFrameOutputs.Keys.Where(t => t >= frameIdx).DefaultIfEmpty(int.MinValue).Min();
            if (after != int.MinValue && !selected.ContainsKey(after)) selected[after] = condFrameOutputs[after];

            int numRemain = maxCondFrameNum - selected.Count;
            var remainKeys = condFrameOutputs.Keys
                .Where(t => !selected.ContainsKey(t))
                .OrderBy(t => Math.Abs(t - frameIdx))
                .Take(Math.Max(0, numRemain));
            foreach (var t in remainKeys) selected[t] = condFrameOutputs[t];

            var unselected = condFrameOutputs.Where(kv => !selected.ContainsKey(kv.Key))
                .ToDictionary(kv => kv.Key, kv => kv.Value);
            return (selected, unselected);
        }

        /// <summary>对应 sam2_utils.get_1d_sine_pe。</summary>
        public static Tensor Get1dSinePe(Tensor posInds, long dim, double temperature = 10000.0)
        {
            long peDim = dim / 2;
            Tensor dimT = arange(peDim, dtype: ScalarType.Float32, device: posInds.device);
            Tensor dimTHalfFloor = (dimT / 2.0).floor() * 2.0;
            dimT = pow(temperature, dimTHalfFloor / peDim);

            Tensor posEmbed = posInds.unsqueeze(-1) / dimT;
            posEmbed = cat(new[] { posEmbed.sin(), posEmbed.cos() }, dim: -1);
            return posEmbed;
        }

        /// <summary>
        /// 对应 Python 端 _prepare_memory_conditioned_features。融合当前帧视觉特征与历史 memory。
        /// currentVisionFeats/currentVisionPosEmbeds 只需传入最后一层（最低分辨率）的单元素列表。
        /// </summary>
        public Tensor PrepareMemoryConditionedFeatures(
            int frameIdx,
            bool isInitCondFrame,
            IList<Tensor> currentVisionFeats,
            IList<Tensor> currentVisionPosEmbeds,
            IList<(long H, long W)> featSizes,
            VideoOutputDict outputDict,
            int numFrames,
            bool trackInReverse = false)
        {
            long B = currentVisionFeats[^1].shape[1];
            long C = hidden_dim;
            var (H, W) = featSizes[^1];
            Device device = currentVisionFeats[^1].device;

            if (num_maskmem == 0)
            {
                return currentVisionFeats[^1].permute(1, 2, 0).view(B, C, H, W);
            }

            long numObjPtrTokens = 0;
            int tposSignMul = trackInReverse ? -1 : 1;

            if (!isInitCondFrame)
            {
                var toCatMemory = new List<Tensor>();
                var toCatMemoryPos = new List<Tensor>();

                if (outputDict.CondFrameOutputs.Count == 0)
                {
                    throw new InvalidOperationException("output_dict.cond_frame_outputs 不能为空");
                }
                var (selectedCond, unselectedCond) = SelectClosestCondFrames(frameIdx, outputDict.CondFrameOutputs, max_cond_frames_in_attn);

                var tPosAndPrevs = new List<(int TPos, FrameOutput? Out)>();
                foreach (var kv in selectedCond)
                {
                    tPosAndPrevs.Add((0, kv.Value));
                }

                int stride = memory_temporal_stride_for_eval;
                for (int tPos = 1; tPos < num_maskmem; tPos++)
                {
                    int tRel = num_maskmem - tPos;
                    int prevFrameIdx;
                    if (tRel == 1)
                    {
                        prevFrameIdx = !trackInReverse ? frameIdx - tRel : frameIdx + tRel;
                    }
                    else
                    {
                        if (!trackInReverse)
                        {
                            prevFrameIdx = ((frameIdx - 2) / stride) * stride;
                            prevFrameIdx = prevFrameIdx - (tRel - 2) * stride;
                        }
                        else
                        {
                            prevFrameIdx = -(-(frameIdx + 2) / stride) * stride;
                            prevFrameIdx = prevFrameIdx + (tRel - 2) * stride;
                        }
                    }

                    FrameOutput? outFrame = outputDict.NonCondFrameOutputs.GetValueOrDefault(prevFrameIdx);
                    if (outFrame is null)
                    {
                        outFrame = unselectedCond.GetValueOrDefault(prevFrameIdx);
                    }
                    tPosAndPrevs.Add((tPos, outFrame));
                }

                foreach (var (tPos, prev) in tPosAndPrevs)
                {
                    if (prev is null) continue;

                    Tensor feats = prev.MaskmemFeatures!.to(device);
                    toCatMemory.Add(feats.flatten(2).permute(2, 0, 1));

                    Tensor maskmemEnc = prev.MaskmemPosEnc![^1].to(device);
                    maskmemEnc = maskmemEnc.flatten(2).permute(2, 0, 1);
                    maskmemEnc = maskmemEnc + maskmem_tpos_enc[num_maskmem - tPos - 1];
                    toCatMemoryPos.Add(maskmemEnc);
                }

                if (use_obj_ptrs_in_encoder)
                {
                    int maxObjPtrsInEncoder = Math.Min(numFrames, max_obj_ptrs_in_encoder);

                    IEnumerable<KeyValuePair<int, FrameOutput>> ptrCondOutputs;
                    if (only_obj_ptrs_in_the_past_for_eval)
                    {
                        ptrCondOutputs = selectedCond.Where(kv => trackInReverse ? kv.Key >= frameIdx : kv.Key <= frameIdx);
                    }
                    else
                    {
                        ptrCondOutputs = selectedCond;
                    }

                    var posAndPtrs = new List<(double Pos, Tensor Ptr)>();
                    foreach (var kv in ptrCondOutputs)
                    {
                        double pos = use_signed_tpos_enc_to_obj_ptrs
                            ? (frameIdx - kv.Key) * tposSignMul
                            : Math.Abs(frameIdx - kv.Key);
                        posAndPtrs.Add((pos, kv.Value.ObjPtr));
                    }

                    for (int tDiff = 1; tDiff < maxObjPtrsInEncoder; tDiff++)
                    {
                        int t = trackInReverse ? frameIdx + tDiff : frameIdx - tDiff;
                        if (t < 0 || t >= numFrames) break;

                        FrameOutput? outFrame = outputDict.NonCondFrameOutputs.GetValueOrDefault(t)
                            ?? unselectedCond.GetValueOrDefault(t);
                        if (outFrame is not null)
                        {
                            posAndPtrs.Add((tDiff, outFrame.ObjPtr));
                        }
                    }

                    if (posAndPtrs.Count > 0)
                    {
                        var ptrsList = posAndPtrs.Select(p => p.Ptr).ToList();
                        Tensor objPtrs = stack(ptrsList, dim: 0);

                        Tensor objPos;
                        if (add_tpos_enc_to_obj_ptrs)
                        {
                            double tDiffMax = maxObjPtrsInEncoder - 1;
                            long tposDim = proj_tpos_enc_in_obj_ptrs ? C : mem_dim;
                            Tensor objPosInds = tensor(posAndPtrs.Select(p => (float)p.Pos).ToArray(), device: device);
                            objPos = Get1dSinePe(objPosInds / tDiffMax, tposDim);
                            objPos = obj_ptr_tpos_proj.forward(objPos);
                            objPos = objPos.unsqueeze(1).expand(-1, B, mem_dim);
                        }
                        else
                        {
                            objPos = objPtrs.new_zeros(posAndPtrs.Count, B, mem_dim);
                        }

                        if (mem_dim < C)
                        {
                            long splitCount = C / mem_dim;
                            objPtrs = objPtrs.reshape(-1, B, splitCount, mem_dim);
                            objPtrs = objPtrs.permute(0, 2, 1, 3).flatten(0, 1);
                            objPos = objPos.repeat_interleave(splitCount, dim: 0);
                        }

                        toCatMemory.Add(objPtrs);
                        toCatMemoryPos.Add(objPos);
                        numObjPtrTokens = objPtrs.shape[0];
                    }
                }

                Tensor memory = cat(toCatMemory, dim: 0);
                Tensor memoryPosEmbed = cat(toCatMemoryPos, dim: 0);

                Tensor pixFeatWithMem = memory_attention!.forward(
                    currentVisionFeats[^1], memory, currentVisionPosEmbeds[^1], memoryPosEmbed, numObjPtrTokens);
                pixFeatWithMem = pixFeatWithMem.permute(1, 2, 0).view(B, C, H, W);
                return pixFeatWithMem;
            }
            else
            {
                if (directly_add_no_mem_embed)
                {
                    Tensor pixFeatWithMem = currentVisionFeats[^1] + no_mem_embed;
                    pixFeatWithMem = pixFeatWithMem.permute(1, 2, 0).view(B, C, H, W);
                    return pixFeatWithMem;
                }

                Tensor memory0 = no_mem_embed.expand(1, B, mem_dim);
                Tensor memoryPos0 = no_mem_pos_enc.expand(1, B, mem_dim);

                Tensor pixFeatWithMem2 = memory_attention!.forward(
                    currentVisionFeats[^1], memory0, currentVisionPosEmbeds[^1], memoryPos0, 0);
                pixFeatWithMem2 = pixFeatWithMem2.permute(1, 2, 0).view(B, C, H, W);
                return pixFeatWithMem2;
            }
        }

        /// <summary>对应 Python 端 _encode_new_memory。</summary>
        public (Tensor MaskmemFeatures, IList<Tensor> MaskmemPosEnc) EncodeNewMemory(
            IList<Tensor> currentVisionFeats,
            IList<(long H, long W)> featSizes,
            Tensor predMasksHighRes,
            Tensor objectScoreLogits,
            bool isMaskFromPoints = false)
        {
            long B = currentVisionFeats[^1].shape[1];
            long C = hidden_dim;
            var (H, W) = featSizes[^1];
            Tensor pixFeat = currentVisionFeats[^1].permute(1, 2, 0).view(B, C, H, W);

            Tensor maskForMem = binarize_mask_from_pts_for_mem_enc && isMaskFromPoints && !training
                ? (predMasksHighRes > 0).to(ScalarType.Float32)
                : sigmoid(predMasksHighRes);
            if (sigmoid_scale_for_mem_enc != 1.0)
            {
                maskForMem = maskForMem * sigmoid_scale_for_mem_enc;
            }
            if (sigmoid_bias_for_mem_enc != 0.0)
            {
                maskForMem = maskForMem + sigmoid_bias_for_mem_enc;
            }

            var maskmemOut = memory_encoder!.forward(pixFeat, maskForMem, skipMaskSigmoid: true);
            Tensor maskmemFeatures = maskmemOut.VisionFeatures;
            IList<Tensor> maskmemPosEnc = maskmemOut.VisionPosEnc;

            if (no_obj_embed_spatial is not null)
            {
                Tensor isObjAppearing = (objectScoreLogits > 0).to(ScalarType.Float32);
                Tensor noObjTerm = (1 - isObjAppearing[TensorIndex.Ellipsis, TensorIndex.None, TensorIndex.None])
                    * no_obj_embed_spatial[TensorIndex.Ellipsis, TensorIndex.None, TensorIndex.None].expand(maskmemFeatures.shape);
                maskmemFeatures = maskmemFeatures + noObjTerm;
            }

            return (maskmemFeatures, maskmemPosEnc);
        }

        /// <summary>
        /// 对应 Python 端 _track_step + track_step 的合并版本（推理场景不区分二者的中间态）。
        /// currentVisionFeats/currentVisionPosEmbeds/featSizes 需包含全部 num_feature_levels 层
        /// （从高分辨率到低分辨率），与 PrepareBackboneFeatures 的输出一致。
        /// </summary>
        public FrameOutput TrackStep(
            int frameIdx,
            bool isInitCondFrame,
            IList<Tensor> currentVisionFeats,
            IList<Tensor> currentVisionPosEmbeds,
            IList<(long H, long W)> featSizes,
            PointInputs? pointInputs,
            Tensor? maskInputs,
            VideoOutputDict outputDict,
            int numFrames,
            bool trackInReverse = false,
            bool runMemEncoder = true,
            Tensor? prevSamMaskLogits = null)
        {
            IList<Tensor>? highResFeatures = null;
            if (currentVisionFeats.Count > 1)
            {
                highResFeatures = new List<Tensor>();
                for (int i = 0; i < currentVisionFeats.Count - 1; i++)
                {
                    var (h, w) = featSizes[i];
                    Tensor x = currentVisionFeats[i];
                    highResFeatures.Add(x.permute(1, 2, 0).view(x.shape[1], x.shape[2], h, w));
                }
            }

            SamHeadsOutput samOutputs;
            if (maskInputs is not null && use_mask_input_as_output_without_sam)
            {
                Tensor pixFeat = currentVisionFeats[^1].permute(1, 2, 0);
                var (h0, w0) = featSizes[^1];
                pixFeat = pixFeat.view(-1, hidden_dim, h0, w0);
                samOutputs = UseMaskAsOutput(pixFeat, highResFeatures, maskInputs);
            }
            else
            {
                Tensor pixFeat = PrepareMemoryConditionedFeatures(
                    frameIdx,
                    isInitCondFrame,
                    new List<Tensor> { currentVisionFeats[^1] },
                    new List<Tensor> { currentVisionPosEmbeds[^1] },
                    new List<(long, long)> { featSizes[^1] },
                    outputDict,
                    numFrames,
                    trackInReverse);

                Tensor? effectiveMaskInputs = maskInputs;
                if (prevSamMaskLogits is not null)
                {
                    effectiveMaskInputs = prevSamMaskLogits;
                }
                bool multimaskOutput = UseMultimask(isInitCondFrame, pointInputs);
                samOutputs = ForwardSamHeads(pixFeat, pointInputs, effectiveMaskInputs, highResFeatures, multimaskOutput);
            }

            var currentOut = new FrameOutput
            {
                PredMasks = samOutputs.LowResMasks,
                PredMasksHighRes = samOutputs.HighResMasks,
                ObjPtr = samOutputs.ObjPtr,
                ObjectScoreLogits = samOutputs.ObjectScoreLogits,
            };

            if (runMemEncoder && num_maskmem > 0)
            {
                var (maskmemFeatures, maskmemPosEnc) = EncodeNewMemory(
                    currentVisionFeats, featSizes, samOutputs.HighResMasks, samOutputs.ObjectScoreLogits);
                currentOut.MaskmemFeatures = maskmemFeatures;
                currentOut.MaskmemPosEnc = maskmemPosEnc;
            }

            return currentOut;
        }
    }
}
