using SAMTorchSharp.Modeling.Sam2;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using System.Collections.Concurrent;

namespace SAMTorchSharp
{
    /// <summary>
    /// Cached feature from _get_image_feature + PrepareBackboneFeatures output.
    /// </summary>
    public class CachedFeature
    {
        public Tensor Image { get; set; } = null!;
        public IList<Tensor> BackboneFpn { get; set; } = null!;
        public IList<Tensor> VisionPosEnc { get; set; } = null!;
        public (IList<Tensor> VisionFeats, IList<Tensor> VisionPosEmbeds, IList<(long H, long W)> FeatSizes) Expanded { get; set; } = default!;
    }

    /// <summary>
    /// Per-object tracking output record, corresponding to Python's FrameOutput (compact form).
    /// </summary>
    public class TrackingFrameOutput
    {
        public Tensor? MaskmemFeatures { get; set; }
        public IList<Tensor>? MaskmemPosEnc { get; set; }
        public Tensor PredMasks { get; set; } = null!;
        public Tensor ObjPtr { get; set; } = null!;
        public Tensor ObjectScoreLogits { get; set; } = null!;
    }

    /// <summary>
    /// Per-object output dict (cond / non_cond frame outputs).
    /// </summary>
    public class ObjectOutputDict
    {
        public Dictionary<int, TrackingFrameOutput> CondFrameOutputs { get; } = new();
        public Dictionary<int, TrackingFrameOutput> NonCondFrameOutputs { get; } = new();
    }

    /// <summary>
    /// Point prompt input (stored by frame_idx key).
    /// </summary>
    public class PointInputPerFrame
    {
        public Tensor PointCoords { get; set; } = null!;
        public Tensor PointLabels { get; set; } = null!;
    }

    /// <summary>
    /// Entry record in frames_tracked_per_obj.
    /// </summary>
    public class FrameTrackedInfo
    {
        public bool Reverse { get; set; }
    }

    /// <summary>
    /// SAM2 video inference high-level wrapper.
    /// Corresponds to sam2/sam2_video_predictor.py: SAM2VideoPredictor.
    /// Manages inference_state (dictionary-based), provides init_state,
    /// add_new_points_or_box, add_new_mask, propagate_in_video,
    /// clear_all_prompts_in_frame, reset_state.
    /// </summary>
    public class SAM2VideoPredictor : IDisposable
    {
        private readonly Sam2Base _model;
        private readonly double _fillHoleArea;
        private readonly bool _nonOverlapMasks;
        private readonly bool _clearNonCondMemAroundInput;
        private readonly bool _addAllFramesToCorrectAsCond;
        private readonly Device _modelDevice;

        public int ImageSize => _model.image_size;

        public SAM2VideoPredictor(Sam2Base model,
            double fillHoleArea = 0,
            bool nonOverlapMasks = false,
            bool clearNonCondMemAroundInput = false,
            bool addAllFramesToCorrectAsCond = false)
        {
            _model = model;
            _fillHoleArea = fillHoleArea;
            _nonOverlapMasks = nonOverlapMasks;
            _clearNonCondMemAroundInput = clearNonCondMemAroundInput;
            _addAllFramesToCorrectAsCond = addAllFramesToCorrectAsCond;
            _modelDevice = _model.parameters().FirstOrDefault()?.device ?? CPU;
        }

        // =====================================================================
        // Public API
        // =====================================================================

        /// <summary>
        /// Initialize inference state.
        /// </summary>
        public Dictionary<string, object> InitState(Tensor images, long originalHeight, long originalWidth, bool offloadStateToCPU = false)
        {
            var state = new Dictionary<string, object>();
            var device = _modelDevice;

            state["images"] = images;
            state["num_frames"] = images.size(0);
            state["offload_video_to_cpu"] = false;
            state["offload_state_to_cpu"] = offloadStateToCPU;
            state["video_height"] = originalHeight;
            state["video_width"] = originalWidth;
            state["device"] = device;
            state["storage_device"] = offloadStateToCPU ? torch.device("cpu") : device;

            state["point_inputs_per_obj"] = new Dictionary<long, Dictionary<int, PointInputPerFrame>>();
            state["mask_inputs_per_obj"] = new Dictionary<long, Dictionary<int, Tensor>>();
            state["cached_features"] = new Dictionary<int, CachedFeature>();
            state["constants"] = new Dictionary<string, object>();
            state["obj_id_to_idx"] = new ConcurrentDictionary<long, long>();
            state["obj_idx_to_id"] = new ConcurrentDictionary<long, long>();
            state["obj_ids"] = new List<long>();
            state["output_dict_per_obj"] = new Dictionary<long, ObjectOutputDict>();
            state["temp_output_dict_per_obj"] = new Dictionary<long, ObjectOutputDict>();
            state["frames_tracked_per_obj"] = new Dictionary<long, Dictionary<int, FrameTrackedInfo>>();

            // Warm up frame 0
            _GetImageFeature(state, 0, 1);

            return state;
        }

        /// <summary>
        /// Add new points (or box) to a frame.
        /// </summary>
        public (int FrameIdx, List<long> ObjIds, Tensor VideoResMasks) AddNewPointsOrBox(
            Dictionary<string, object> state,
            int frameIdx,
            long objId,
            Tensor points,
            Tensor labels,
            bool clearOldPoints = true,
            Tensor? box = null)
        {
            var objIdx = _ObjIdToIdx(state, objId);
            var pointInputsPerFrame = (Dictionary<long, Dictionary<int, PointInputPerFrame>>)state["point_inputs_per_obj"];
            var maskInputsPerFrame = (Dictionary<long, Dictionary<int, Tensor>>)state["mask_inputs_per_obj"];

            var videoH = (long)state["video_height"];
            var videoW = (long)state["video_width"];
            var device = (Device)state["device"];

            Tensor pts = points.to(device);
            Tensor lbls = labels.to(device).to(ScalarType.Int32);

            if (pts.dim() == 2) pts = pts.unsqueeze(0);
            if (lbls.dim() == 1) lbls = lbls.unsqueeze(0);

            // Scale coordinates: [orig_px] / [W, H] * image_size
            Tensor ptsNormalized = pts.clone();
            var wTensor = tensor((float)videoW, device: device);
            var hTensor = tensor((float)videoH, device: device);
            var imgSizeTensor = tensor((float)_model.image_size, device: device);

            var col0 = ptsNormalized.index(new TensorIndex[] { TensorIndex.Ellipsis, 0 });
            var col1 = ptsNormalized.index(new TensorIndex[] { TensorIndex.Ellipsis, 1 });
            col0 = (col0 / wTensor) * imgSizeTensor;
            col1 = (col1 / hTensor) * imgSizeTensor;
            var cols = stack(new[] { col0, col1 }, dim: -1);

            if (box is not null)
            {
                if (!clearOldPoints)
                    throw new InvalidOperationException("cannot add box without clearing old points");
                var b = box.to(device).reshape(1, 2, 2);
                var bl = tensor(new long[] { 2, 3 }, dtype: ScalarType.Int32, device: device).reshape(1, 2);
                cols = cat(new[] { b, cols }, dim: 1);
                lbls = cat(new[] { bl, lbls }, dim: 1);
            }
            else
            {
                ptsNormalized = cols;
            }

            var ptInputs = pointInputsPerFrame[objIdx];
            if (!ptInputs.ContainsKey(frameIdx))
                ptInputs[frameIdx] = new PointInputPerFrame();

            var existing = ptInputs[frameIdx];
            if (existing is not null && !clearOldPoints)
            {
                ptsNormalized = cat(new[] { existing.PointCoords, ptsNormalized }, dim: 1);
                lbls = cat(new[] { existing.PointLabels, lbls }, dim: 1);
            }

            ptInputs[frameIdx] = new PointInputPerFrame
            {
                PointCoords = ptsNormalized.contiguous(),
                PointLabels = lbls.contiguous()
            };
            maskInputsPerFrame[objIdx].Remove(frameIdx);

            var framesTracked = (Dictionary<long, Dictionary<int, FrameTrackedInfo>>)state["frames_tracked_per_obj"];
            var objTracked = framesTracked[objIdx];
            bool isInitCondFrame = !objTracked.ContainsKey(frameIdx);
            bool reverse = isInitCondFrame ? false : objTracked[frameIdx].Reverse;

            var outputDicts = (Dictionary<long, ObjectOutputDict>)state["output_dict_per_obj"];
            var tempOutputDicts = (Dictionary<long, ObjectOutputDict>)state["temp_output_dict_per_obj"];
            var objOutputDict = outputDicts[objIdx];
            var objTempOutputDict = tempOutputDicts[objIdx];

            bool isCond = isInitCondFrame || _addAllFramesToCorrectAsCond;

            // Get prev_sam_mask_logits
            Tensor? prevSamMaskLogits = null;
            TrackingFrameOutput? prevOut = null;

            if (isCond)
            {
                objTempOutputDict.CondFrameOutputs.TryGetValue(frameIdx, out prevOut);
                if (prevOut == null) objOutputDict.CondFrameOutputs.TryGetValue(frameIdx, out prevOut);
                if (prevOut == null) objOutputDict.NonCondFrameOutputs.TryGetValue(frameIdx, out prevOut);
            }
            else
            {
                objTempOutputDict.NonCondFrameOutputs.TryGetValue(frameIdx, out prevOut);
                if (prevOut == null) objOutputDict.CondFrameOutputs.TryGetValue(frameIdx, out prevOut);
                if (prevOut == null) objOutputDict.NonCondFrameOutputs.TryGetValue(frameIdx, out prevOut);
            }

            if (prevOut is not null && prevOut.PredMasks is not null)
            {
                prevSamMaskLogits = prevOut.PredMasks.to(device).clamp(-32.0f, 32.0f);
            }

            var pointInputs = new PointInputs
            {
                PointCoords = ptInputs[frameIdx].PointCoords,
                PointLabels = ptInputs[frameIdx].PointLabels
            };

            var (currentOut, _) = _RunSingleFrameInference(
                state, objOutputDict, frameIdx, 1, isInitCondFrame, pointInputs,
                null, reverse, runMemEncoder: false, prevSamMaskLogits);

            if (isCond)
                objTempOutputDict.CondFrameOutputs[frameIdx] = currentOut;
            else
                objTempOutputDict.NonCondFrameOutputs[frameIdx] = currentOut;

            var objIds = (List<long>)state["obj_ids"];
            var consolidated = _ConsolidateTempOutputAcrossObj(state, frameIdx, isCond, consolidateAtVideoRes: true);
            var (_, videoResMasks) = _GetOrigVideoResOutput(state, consolidated["pred_masks"]);

            return (frameIdx, objIds, videoResMasks);
        }

        /// <summary>
        /// Add new mask to a frame.
        /// </summary>
        public (int FrameIdx, List<long> ObjIds, Tensor VideoResMasks) AddNewMask(
            Dictionary<string, object> state,
            int frameIdx,
            long objId,
            Tensor mask)
        {
            var objIdx = _ObjIdToIdx(state, objId);
            var maskInputsPerFrame = (Dictionary<long, Dictionary<int, Tensor>>)state["mask_inputs_per_obj"];
            var pointInputsPerFrame = (Dictionary<long, Dictionary<int, PointInputPerFrame>>)state["point_inputs_per_obj"];

            var modelDevice = _modelDevice;

            Tensor maskInput;
            if (mask.dtype != ScalarType.Float32)
            {
                maskInput = mask.to(ScalarType.Float32, device: modelDevice).unsqueeze(0).unsqueeze(0);
            }
            else
            {
                maskInput = mask.unsqueeze(0).unsqueeze(0);
            }

            long maskH = maskInput.size(2);
            long maskW = maskInput.size(3);

            if (maskH != _model.image_size || maskW != _model.image_size)
            {
                maskInput = interpolate(maskInput,
                    size: new long[] { _model.image_size, _model.image_size },
                    mode: InterpolationMode.Bilinear, align_corners: false);
                maskInput = (maskInput >= 0.5f).to(ScalarType.Float32);
            }

            maskInputsPerFrame[objIdx][frameIdx] = maskInput;
            pointInputsPerFrame[objIdx].Remove(frameIdx);

            var framesTracked = (Dictionary<long, Dictionary<int, FrameTrackedInfo>>)state["frames_tracked_per_obj"];
            var objTracked = framesTracked[objIdx];
            bool isInitCondFrame = !objTracked.ContainsKey(frameIdx);
            bool reverse = isInitCondFrame ? false : objTracked[frameIdx].Reverse;

            var outputDicts = (Dictionary<long, ObjectOutputDict>)state["output_dict_per_obj"];
            var tempOutputDicts = (Dictionary<long, ObjectOutputDict>)state["temp_output_dict_per_obj"];
            var objOutputDict = outputDicts[objIdx];
            var objTempOutputDict = tempOutputDicts[objIdx];

            bool isCond = isInitCondFrame || _addAllFramesToCorrectAsCond;

            var (currentOut, _) = _RunSingleFrameInference(
                state, objOutputDict, frameIdx, 1, isInitCondFrame,
                pointInputs: null, maskInputs: maskInput,
                reverse, runMemEncoder: false);

            if (isCond)
                objTempOutputDict.CondFrameOutputs[frameIdx] = currentOut;
            else
                objTempOutputDict.NonCondFrameOutputs[frameIdx] = currentOut;

            var objIds = (List<long>)state["obj_ids"];
            var consolidated = _ConsolidateTempOutputAcrossObj(state, frameIdx, isCond, consolidateAtVideoRes: true);
            var (_, videoResMasks) = _GetOrigVideoResOutput(state, consolidated["pred_masks"]);

            return (frameIdx, objIds, videoResMasks);
        }

        /// <summary>
        /// Remove all input points or mask in a specific frame for a given object.
        /// </summary>
        public (int FrameIdx, List<long> ObjIds, Tensor VideoResMasks)? ClearAllPromptsInFrame(
            Dictionary<string, object> state,
            int frameIdx,
            long objId)
        {
            var objIdx = _ObjIdToIdx(state, objId);

            var ptInputs = (Dictionary<long, Dictionary<int, PointInputPerFrame>>)state["point_inputs_per_obj"];
            var msInputs = (Dictionary<long, Dictionary<int, Tensor>>)state["mask_inputs_per_obj"];
            ptInputs[objIdx].Remove(frameIdx);
            msInputs[objIdx].Remove(frameIdx);

            var tempOutputDicts = (Dictionary<long, ObjectOutputDict>)state["temp_output_dict_per_obj"];
            tempOutputDicts[objIdx].CondFrameOutputs.Remove(frameIdx);
            tempOutputDicts[objIdx].NonCondFrameOutputs.Remove(frameIdx);

            var outputDicts = (Dictionary<long, ObjectOutputDict>)state["output_dict_per_obj"];
            var objOutputDict = outputDicts[objIdx];
            var outRec = objOutputDict.CondFrameOutputs.GetValueOrDefault(frameIdx);
            if (outRec != null)
            {
                objOutputDict.CondFrameOutputs.Remove(frameIdx);
                objOutputDict.NonCondFrameOutputs[frameIdx] = outRec;
                var framesTracked = (Dictionary<long, Dictionary<int, FrameTrackedInfo>>)state["frames_tracked_per_obj"];
                framesTracked[objIdx].Remove(frameIdx);
            }

            var objIds = (List<long>)state["obj_ids"];
            bool isCond = false;
            foreach (var tmpDict in tempOutputDicts.Values)
            {
                if (tmpDict.CondFrameOutputs.ContainsKey(frameIdx))
                {
                    isCond = true;
                    break;
                }
            }

            var consolidated = _ConsolidateTempOutputAcrossObj(state, frameIdx, isCond, consolidateAtVideoRes: true);
            var (_, videoResMasks) = _GetOrigVideoResOutput(state, consolidated["pred_masks"]);

            return (frameIdx, objIds, videoResMasks);
        }

        /// <summary>
        /// Propagate the input points across frames to track in the entire video.
        /// </summary>
        public IEnumerable<(int FrameIdx, List<long> ObjIds, Tensor VideoResMasks)> PropagateInVideo(
            Dictionary<string, object> state,
            int? startFrameIdx = null,
            int? maxFrameNumToTrack = null,
            bool reverse = false)
        {
            _PropagateInVideoPreflight(state);

            var objIds = (List<long>)state["obj_ids"];
            var numFrames = (long)state["num_frames"];
            var outputDicts = (Dictionary<long, ObjectOutputDict>)state["output_dict_per_obj"];
            var batch = outputDicts.Count;

            int start;
            if (startFrameIdx == null)
            {
                start = int.MaxValue;
                foreach (var od in outputDicts.Values)
                {
                    foreach (var t in od.CondFrameOutputs.Keys)
                    {
                        if (t < start) start = t;
                    }
                }
                if (start == int.MaxValue) start = 0;
            }
            else
            {
                start = startFrameIdx.Value;
            }

            int maxFrames = maxFrameNumToTrack ?? (int)numFrames;
            int end;
            IEnumerable<int> processingOrder;

            if (reverse)
            {
                end = Math.Max(start - maxFrames, 0);
                if (start > 0)
                {
                    processingOrder = Enumerable.Range(end, start - end + 1).Reverse();
                }
                else
                {
                    processingOrder = Array.Empty<int>();
                }
            }
            else
            {
                end = Math.Min(start + maxFrames, (int)numFrames - 1);
                processingOrder = Enumerable.Range(start, end - start + 1);
            }

            foreach (int frameIdx in processingOrder)
            {
                var predMasksPerObj = new List<Tensor>(batch);

                for (int objIdx = 0; objIdx < batch; objIdx++)
                {
                    var objOutputDict = outputDicts[objIdx];

                    if (objOutputDict.CondFrameOutputs.ContainsKey(frameIdx))
                    {
                        var currentOut = objOutputDict.CondFrameOutputs[frameIdx];
                        var predMasks = currentOut.PredMasks.to(_modelDevice);
                        if (_clearNonCondMemAroundInput)
                        {
                            _ClearNonCondMemAroundInput(state, frameIdx, objIdx);
                        }
                        predMasksPerObj.Add(predMasks);
                    }
                    else
                    {
                        var (compactOut, predMasksGpu) = _RunSingleFrameInference(
                            state, objOutputDict, frameIdx, 1,
                            isInitCondFrame: false, pointInputs: null, maskInputs: null,
                            reverse, runMemEncoder: true);

                        objOutputDict.NonCondFrameOutputs[frameIdx] = compactOut;

                        var framesTracked = (Dictionary<long, Dictionary<int, FrameTrackedInfo>>)state["frames_tracked_per_obj"];
                        if (!framesTracked[objIdx].ContainsKey(frameIdx))
                            framesTracked[objIdx][frameIdx] = new FrameTrackedInfo { Reverse = reverse };
                        else
                            framesTracked[objIdx][frameIdx].Reverse = reverse;

                        predMasksPerObj.Add(predMasksGpu);
                    }
                }

                Tensor allPredMasks;
                if (predMasksPerObj.Count > 1)
                {
                    allPredMasks = cat(predMasksPerObj, dim: 0);
                }
                else
                {
                    allPredMasks = predMasksPerObj[0];
                }

                var (_, videoResMasks) = _GetOrigVideoResOutput(state, allPredMasks);
                yield return (frameIdx, objIds, videoResMasks);
            }
        }

        /// <summary>
        /// Remove all input points or mask in all frames throughout the video.
        /// </summary>
        public void ResetState(Dictionary<string, object> state)
        {
            _ResetTrackingResults(state);
            var objIdToIdx = (ConcurrentDictionary<long, long>)state["obj_id_to_idx"];
            var objIdxToId = (ConcurrentDictionary<long, long>)state["obj_idx_to_id"];
            var objIds = (List<long>)state["obj_ids"];
            objIdToIdx.Clear();
            objIdxToId.Clear();
            objIds.Clear();
        }

        // =====================================================================
        // Internal helpers
        // =====================================================================

        private long _ObjIdToIdx(Dictionary<string, object> state, long objId)
        {
            var objIdToIdx = (ConcurrentDictionary<long, long>)state["obj_id_to_idx"];
            if (objIdToIdx.TryGetValue(objId, out var idx))
                return idx;

            idx = objIdToIdx.Count;
            objIdToIdx[objId] = idx;

            var objIdxToId = (ConcurrentDictionary<long, long>)state["obj_idx_to_id"];
            objIdxToId[idx] = objId;

            var objIds = (List<long>)state["obj_ids"];
            objIds.Add(objId);

            var ptInputs = (Dictionary<long, Dictionary<int, PointInputPerFrame>>)state["point_inputs_per_obj"];
            var msInputs = (Dictionary<long, Dictionary<int, Tensor>>)state["mask_inputs_per_obj"];
            var outputDicts = (Dictionary<long, ObjectOutputDict>)state["output_dict_per_obj"];
            var tempOutputDicts = (Dictionary<long, ObjectOutputDict>)state["temp_output_dict_per_obj"];
            var framesTracked = (Dictionary<long, Dictionary<int, FrameTrackedInfo>>)state["frames_tracked_per_obj"];

            ptInputs[idx] = new Dictionary<int, PointInputPerFrame>();
            msInputs[idx] = new Dictionary<int, Tensor>();
            outputDicts[idx] = new ObjectOutputDict();
            tempOutputDicts[idx] = new ObjectOutputDict();
            framesTracked[idx] = new Dictionary<int, FrameTrackedInfo>();

            return idx;
        }

        private void _GetImageFeature(Dictionary<string, object> state, int frameIdx, long batchSize)
        {
            var cachedFeatures = (Dictionary<int, CachedFeature>)state["cached_features"];
            if (cachedFeatures.TryGetValue(frameIdx, out var cached) && cached != null)
                return;

            var images = (Tensor)state["images"];
            var device = (Device)state["device"];

            var image = images[frameIdx].to(device).to(ScalarType.Float32).unsqueeze(0);
            var backboneOut = _model.ForwardImage(image);

            var expandedFpn = backboneOut.BackboneFpn.Select(f => f.expand(new long[] { batchSize, -1, -1, -1 })).ToList();
            var expandedPos = backboneOut.VisionPosEnc.Select(p => p.expand(new long[] { batchSize, -1, -1, -1 })).ToList();

            var feats = _model.PrepareBackboneFeatures(new BackboneOut
            {
                BackboneFpn = expandedFpn,
                VisionPosEnc = expandedPos
            });

            cachedFeatures[frameIdx] = new CachedFeature
            {
                Image = image,
                BackboneFpn = expandedFpn,
                VisionPosEnc = expandedPos,
                Expanded = feats
            };
        }

        private (TrackingFrameOutput CompactOut, Tensor PredMasksGpu) _RunSingleFrameInference(
            Dictionary<string, object> state,
            ObjectOutputDict outputDict,
            int frameIdx,
            long batchSize,
            bool isInitCondFrame,
            PointInputs? pointInputs,
            Tensor? maskInputs,
            bool reverse,
            bool runMemEncoder,
            Tensor? prevSamMaskLogits = null)
        {
            var cachedFeatures = (Dictionary<int, CachedFeature>)state["cached_features"];
            var storageDevice = (Device)state["storage_device"];

            if (!cachedFeatures.TryGetValue(frameIdx, out var cached) || cached == null)
            {
                _GetImageFeature(state, frameIdx, batchSize);
                cached = cachedFeatures[frameIdx];
            }

            if (cached.Image.size(0) != batchSize)
            {
                var expandedImage = cached.Image.expand(new long[] { batchSize, -1, -1, -1 });
                var expandedFpn = cached.BackboneFpn.Select(f => f.expand(new long[] { batchSize, -1, -1, -1 })).ToList();
                var expandedPos = cached.VisionPosEnc.Select(p => p.expand(new long[] { batchSize, -1, -1, -1 })).ToList();

                cached.Expanded = _model.PrepareBackboneFeatures(new BackboneOut
                {
                    BackboneFpn = expandedFpn,
                    VisionPosEnc = expandedPos
                });

                cached.Image = expandedImage;
                cached.BackboneFpn = expandedFpn;
                cached.VisionPosEnc = expandedPos;
            }

            var (visionFeats, visionPos, featSizes) = cached.Expanded;

            var videoOutputDict = new VideoOutputDict();
            foreach (var kv in outputDict.CondFrameOutputs)
            {
                videoOutputDict.CondFrameOutputs[kv.Key] = new FrameOutput
                {
                    MaskmemFeatures = kv.Value.MaskmemFeatures,
                    MaskmemPosEnc = kv.Value.MaskmemPosEnc,
                    ObjPtr = kv.Value.ObjPtr,
                    PredMasks = kv.Value.PredMasks,
                    PredMasksHighRes = kv.Value.PredMasks,
                    ObjectScoreLogits = kv.Value.ObjectScoreLogits
                };
            }
            foreach (var kv in outputDict.NonCondFrameOutputs)
            {
                videoOutputDict.NonCondFrameOutputs[kv.Key] = new FrameOutput
                {
                    MaskmemFeatures = kv.Value.MaskmemFeatures,
                    MaskmemPosEnc = kv.Value.MaskmemPosEnc,
                    ObjPtr = kv.Value.ObjPtr,
                    PredMasks = kv.Value.PredMasks,
                    PredMasksHighRes = kv.Value.PredMasks,
                    ObjectScoreLogits = kv.Value.ObjectScoreLogits
                };
            }

            var currentOut = _model.TrackStep(
                frameIdx: frameIdx,
                isInitCondFrame: isInitCondFrame,
                currentVisionFeats: visionFeats,
                currentVisionPosEmbeds: visionPos,
                featSizes: featSizes,
                pointInputs: pointInputs,
                maskInputs: maskInputs,
                outputDict: videoOutputDict,
                numFrames: (int)state["num_frames"],
                trackInReverse: reverse,
                runMemEncoder: runMemEncoder,
                prevSamMaskLogits: prevSamMaskLogits);

            Tensor maskmemFeatures = currentOut.MaskmemFeatures;
            if (maskmemFeatures is not null)
            {
                maskmemFeatures = maskmemFeatures.to(ScalarType.BFloat16).to(storageDevice, non_blocking: true);
            }

            Tensor predMasksGpu = currentOut.PredMasks.to(storageDevice, non_blocking: true);

            var constants = (Dictionary<string, object>)state["constants"];
            IList<Tensor>? maskmemPosEnc = currentOut.MaskmemPosEnc;
            if (maskmemPosEnc != null && maskmemPosEnc.Count > 0)
            {
                if (!constants.ContainsKey("maskmem_pos_enc"))
                {
                    constants["maskmem_pos_enc"] = maskmemPosEnc.Select(m => m.index(0)).ToList();
                }
            }

            var compactOut = new TrackingFrameOutput
            {
                MaskmemFeatures = maskmemFeatures,
                MaskmemPosEnc = maskmemPosEnc,
                PredMasks = predMasksGpu,
                ObjPtr = currentOut.ObjPtr,
                ObjectScoreLogits = currentOut.ObjectScoreLogits
            };

            return (compactOut, predMasksGpu);
        }

        private Dictionary<string, Tensor> _ConsolidateTempOutputAcrossObj(
            Dictionary<string, object> state,
            int frameIdx,
            bool isCond,
            bool consolidateAtVideoRes = false)
        {
            var outputDicts = (Dictionary<long, ObjectOutputDict>)state["output_dict_per_obj"];
            var tempOutputDicts = (Dictionary<long, ObjectOutputDict>)state["temp_output_dict_per_obj"];
            var storageDevice = (Device)state["storage_device"];
            var numObjects = outputDicts.Count;

            long consolidatedH, consolidatedW;
            if (consolidateAtVideoRes)
            {
                consolidatedH = (long)state["video_height"];
                consolidatedW = (long)state["video_width"];
            }
            else
            {
                consolidatedH = consolidatedW = _model.image_size / 4;
            }

            const float NoObjScore = -1024.0f;
            var predMasks = full(new long[] { numObjects, 1, consolidatedH, consolidatedW },
                NoObjScore, dtype: ScalarType.Float32, device: storageDevice);

            foreach (var kvp in outputDicts)
            {
                var objIdx = kvp.Key;
                var objTempDict = tempOutputDicts[objIdx];
                var objOutputDict = outputDicts[objIdx];

                TrackingFrameOutput? outRec = null;
                if (isCond)
                {
                    objTempDict.CondFrameOutputs.TryGetValue(frameIdx, out outRec);
                    if (outRec == null) objOutputDict.CondFrameOutputs.TryGetValue(frameIdx, out outRec);
                    if (outRec == null) objOutputDict.NonCondFrameOutputs.TryGetValue(frameIdx, out outRec);
                }
                else
                {
                    objTempDict.NonCondFrameOutputs.TryGetValue(frameIdx, out outRec);
                    if (outRec == null) objOutputDict.CondFrameOutputs.TryGetValue(frameIdx, out outRec);
                    if (outRec == null) objOutputDict.NonCondFrameOutputs.TryGetValue(frameIdx, out outRec);
                }

                if (outRec == null) continue;

                var objMask = outRec.PredMasks;
                if (objMask.shape[2] == consolidatedH && objMask.shape[3] == consolidatedW)
                {
                    predMasks.index(new TensorIndex[] { objIdx }).copy_(objMask);
                }
                else
                {
                    var resized = interpolate(objMask.to(ScalarType.Float32),
                        size: new long[] { consolidatedH, consolidatedW },
                        mode: InterpolationMode.Bilinear, align_corners: false);
                    predMasks.index(new TensorIndex[] { objIdx }).copy_(resized);
                }
            }

            return new Dictionary<string, Tensor> { { "pred_masks", predMasks } };
        }

        private (Tensor AnyResMasks, Tensor VideoResMasks) _GetOrigVideoResOutput(
            Dictionary<string, object> state, Tensor anyResMasks)
        {
            var device = (Device)state["device"];
            var videoH = (long)state["video_height"];
            var videoW = (long)state["video_width"];

            anyResMasks = anyResMasks.to(device, non_blocking: true);

            Tensor videoResMasks;
            if (anyResMasks.shape[2] == videoH && anyResMasks.shape[3] == videoW)
            {
                videoResMasks = anyResMasks;
            }
            else
            {
                videoResMasks = interpolate(anyResMasks.to(ScalarType.Float32),
                    size: new long[] { videoH, videoW },
                    mode: InterpolationMode.Bilinear, align_corners: false);
            }

            if (_nonOverlapMasks)
            {
                // Placeholder: non-overlapping constraints not yet implemented
            }

            return (anyResMasks, videoResMasks);
        }

        private void _PropagateInVideoPreflight(Dictionary<string, object> state)
        {
            var outputDicts = (Dictionary<long, ObjectOutputDict>)state["output_dict_per_obj"];
            var tempOutputDicts = (Dictionary<long, ObjectOutputDict>)state["temp_output_dict_per_obj"];
            var device = (Device)state["device"];
            var batch = outputDicts.Count;

            if (batch == 0)
                throw new InvalidOperationException("No input points or masks are provided for any object.");

            for (int objIdx = 0; objIdx < batch; objIdx++)
            {
                var objOutputDict = outputDicts[objIdx];
                var objTempDict = tempOutputDicts[objIdx];

                foreach (bool isCondFlag in new[] { false, true })
                {
                    var tempDict = isCondFlag ? objTempDict.CondFrameOutputs : objTempDict.NonCondFrameOutputs;
                    var targetDict = isCondFlag ? objOutputDict.CondFrameOutputs : objOutputDict.NonCondFrameOutputs;

                    foreach (var kvp2 in tempDict.ToArray())
                    {
                        var frameIdx = kvp2.Key;
                        var outRec = kvp2.Value;

                        if (outRec.MaskmemFeatures is null)
                        {
                            var highResMasks = interpolate(
                                outRec.PredMasks.to(device),
                                size: new long[] { _model.image_size, _model.image_size },
                                mode: InterpolationMode.Bilinear, align_corners: false);

                            var (maskmemFeatures, maskmemPosEnc) = _EncodeNewMemory(
                                state, frameIdx, 1, highResMasks, outRec.ObjectScoreLogits);

                            outRec.MaskmemFeatures = maskmemFeatures;
                            outRec.MaskmemPosEnc = maskmemPosEnc;
                        }

                        targetDict[frameIdx] = outRec;

                        if (_clearNonCondMemAroundInput)
                        {
                            _ClearNonCondMemAroundInput(state, frameIdx, objIdx);
                        }
                    }

                    if (isCondFlag)
                        objTempDict.CondFrameOutputs.Clear();
                    else
                        objTempDict.NonCondFrameOutputs.Clear();
                }

                foreach (var frameIdx in objOutputDict.CondFrameOutputs.Keys.ToArray())
                {
                    objOutputDict.NonCondFrameOutputs.Remove(frameIdx);
                }
            }
        }

        private (Tensor MaskmemFeatures, IList<Tensor> MaskmemPosEnc) _EncodeNewMemory(
            Dictionary<string, object> state, int frameIdx, long batchSize,
            Tensor highResMasks, Tensor objectScoreLogits)
        {
            var cachedFeatures = (Dictionary<int, CachedFeature>)state["cached_features"];
            var cached = cachedFeatures[frameIdx];
            var (visionFeats, visionPos, featSizes) = cached.Expanded;

            return _model.EncodeNewMemory(visionFeats, featSizes, highResMasks, objectScoreLogits);
        }

        private void _ClearNonCondMemAroundInput(Dictionary<string, object> state, int frameIdx, int objIdx)
        {
            var outputDicts = (Dictionary<long, ObjectOutputDict>)state["output_dict_per_obj"];
            var r = _model.memory_temporal_stride_for_eval;
            int begin = frameIdx - r * _model.num_maskmem;
            int end = frameIdx + r * _model.num_maskmem;

            var objOutputDict = outputDicts[objIdx];
            for (int t = begin; t <= end; t++)
            {
                objOutputDict.NonCondFrameOutputs.Remove(t);
            }
        }

        private void _ResetTrackingResults(Dictionary<string, object> state)
        {
            var ptInputs = (Dictionary<long, Dictionary<int, PointInputPerFrame>>)state["point_inputs_per_obj"];
            var msInputs = (Dictionary<long, Dictionary<int, Tensor>>)state["mask_inputs_per_obj"];
            var outputDicts = (Dictionary<long, ObjectOutputDict>)state["output_dict_per_obj"];
            var tempOutputDicts = (Dictionary<long, ObjectOutputDict>)state["temp_output_dict_per_obj"];
            var framesTracked = (Dictionary<long, Dictionary<int, FrameTrackedInfo>>)state["frames_tracked_per_obj"];

            foreach (var v in ptInputs.Values) v.Clear();
            foreach (var v in msInputs.Values) v.Clear();
            foreach (var v in outputDicts.Values)
            {
                v.CondFrameOutputs.Clear();
                v.NonCondFrameOutputs.Clear();
            }
            foreach (var v in tempOutputDicts.Values)
            {
                v.CondFrameOutputs.Clear();
                v.NonCondFrameOutputs.Clear();
            }
            foreach (var v in framesTracked.Values) v.Clear();
        }

        public void Dispose()
        {
            _model.Dispose();
        }
    }
}
