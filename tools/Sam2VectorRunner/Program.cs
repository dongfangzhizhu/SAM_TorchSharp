using SAMTorchSharp.Modeling.Sam2;
using TorchSharp;
using TorchSharp.Modules;
using TorchSharp.PyBridge;
using MaskDecoder = SAMTorchSharp.Modeling.Sam2.MaskDecoder;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Sam2VectorRunner
{
    /// <summary>
    /// 命令行工具：加载 Python 端生成的测试向量（weights.safetensors + input.safetensors），
    /// 用 .NET(TorchSharp) 实现的 SAM2 模块跑一次前向，导出 output_net.safetensors，
    /// 供 SafetensorsCompare 与 Python 端的 output_py.safetensors 做数值比对。
    ///
    /// 用法:
    ///   Sam2VectorRunner hiera --variant tiny --dir &lt;testdata/hiera_tiny&gt;
    /// </summary>
    public static class Program
    {
        public static int Main(string[] args)
        {
            if (args.Length == 0)
            {
                PrintUsage();
                return 2;
            }

            string module = args[0];
            var opts = ParseOptions(args.Skip(1).ToArray());

            if (!opts.TryGetValue("dir", out var dir) || string.IsNullOrEmpty(dir))
            {
                Console.Error.WriteLine("[错误] 缺少 --dir 参数");
                return 2;
            }

            try
            {
                switch (module)
                {
                    case "hiera":
                        return RunHiera(dir, opts.GetValueOrDefault("variant", "tiny"));
                    case "image_encoder":
                        return RunImageEncoder(dir, opts.GetValueOrDefault("variant", "tiny"), int.Parse(opts.GetValueOrDefault("d-model", "256")));
                    case "prompt_encoder":
                        return RunPromptEncoder(dir);
                    case "mask_decoder":
                        return RunMaskDecoder(dir);
                    case "sam2_image":
                        return RunSam2Image(dir, int.Parse(opts.GetValueOrDefault("image-size", "256")));
                    case "sam2_real":
                        return RunSam2Real(dir, opts.GetValueOrDefault("variant", "tiny"));
                    case "rope_attention":
                        return RunRopeAttention(dir);
                    case "memory_encoder":
                        return RunMemoryEncoder(dir);
                    default:
                        Console.Error.WriteLine($"[错误] 未知模块: {module}");
                        PrintUsage();
                        return 2;
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[异常] {ex}");
                return 3;
            }
        }

        private static int RunHiera(string dir, string variant)
        {
            using var _ = NewDisposeScope();
            using var noGrad = no_grad();

            Hiera model = BuildHiera(variant);
            model.eval();

            string weightsPath = Path.Combine(dir, "weights.safetensors");
            string inputPath = Path.Combine(dir, "input.safetensors");
            string outputPath = Path.Combine(dir, "output_net.safetensors");

            model.load_safetensors(weightsPath);

            var inputs = Safetensors.LoadStateDict(inputPath);
            Tensor x = inputs["x"];

            IList<Tensor> outputs = model.forward(x);

            var outDict = new Dictionary<string, Tensor>();
            for (int i = 0; i < outputs.Count; i++)
            {
                outDict[$"feat_{i}"] = outputs[i].contiguous();
            }

            Safetensors.SaveStateDict(outputPath, outDict);

            Console.WriteLine($"[hiera:{variant}] 输入 x: {string.Join('x', x.shape)}");
            for (int i = 0; i < outputs.Count; i++)
            {
                Console.WriteLine($"  feat_{i}: {string.Join('x', outputs[i].shape)}");
            }
            Console.WriteLine($"已保存: {outputPath}");

            return 0;
        }

        private static Hiera BuildHiera(string variant)
        {
            return variant switch
            {
                "tiny" => new Hiera(
                    embedDim: 96,
                    numHeads: 1,
                    stages: new[] { 1, 2, 7, 2 },
                    globalAttBlocks: new[] { 5, 7, 9 },
                    windowPosEmbedBkgSpatialSize: (7, 7),
                    windowSpec: new[] { 8, 4, 14, 7 }),
                "large" => new Hiera(
                    embedDim: 144,
                    numHeads: 2,
                    stages: new[] { 2, 6, 36, 4 },
                    globalAttBlocks: new[] { 23, 33, 43 },
                    windowPosEmbedBkgSpatialSize: (7, 7),
                    windowSpec: new[] { 8, 4, 16, 8 }),
                _ => throw new ArgumentException($"未知 variant: {variant}")
            };
        }

        private static int RunImageEncoder(string dir, string variant, int dModel)
        {
            using var _ = NewDisposeScope();
            using var noGrad = no_grad();

            Hiera trunk = BuildHiera(variant);
            var positionEncoding = new PositionEmbeddingSine(numPosFeats: dModel, normalize: true);
            var neck = new FpnNeck(
                positionEncoding: positionEncoding,
                dModel: dModel,
                backboneChannelList: trunk.ChannelList,
                fpnTopDownLevels: new[] { 2, 3 },
                fpnInterpModel: "nearest");
            var model = new ImageEncoder(trunk, neck, scalp: 1);
            model.eval();

            string weightsPath = Path.Combine(dir, "weights.safetensors");
            string inputPath = Path.Combine(dir, "input.safetensors");
            string outputPath = Path.Combine(dir, "output_net.safetensors");

            model.load_safetensors(weightsPath);

            var inputs = Safetensors.LoadStateDict(inputPath);
            Tensor x = inputs["x"];

            var output = model.forward(x);

            var outDict = new Dictionary<string, Tensor> { ["vision_features"] = output.VisionFeatures.contiguous() };
            for (int i = 0; i < output.BackboneFpn.Count; i++)
            {
                outDict[$"backbone_fpn_{i}"] = output.BackboneFpn[i].contiguous();
            }
            for (int i = 0; i < output.VisionPosEnc.Count; i++)
            {
                outDict[$"vision_pos_enc_{i}"] = output.VisionPosEnc[i].contiguous();
            }

            Safetensors.SaveStateDict(outputPath, outDict);

            Console.WriteLine($"[image_encoder:{variant}] vision_features: {string.Join('x', output.VisionFeatures.shape)}");
            foreach (var kv in outDict)
            {
                Console.WriteLine($"  {kv.Key}: {string.Join('x', kv.Value.shape)}");
            }
            Console.WriteLine($"已保存: {outputPath}");

            return 0;
        }

        private static int RunPromptEncoder(string rootDir)
        {
            using var _ = NewDisposeScope();
            using var noGrad = no_grad();

            var model = new SAMTorchSharp.Modeling.Sam2.PromptEncoder(
                embed_dim: 256,
                image_embedding_size: (64, 64),
                input_image_size: (1024, 1024),
                mask_in_chans: 16);
            model.eval();
            model.load_safetensors(Path.Combine(rootDir, "weights.safetensors"));

            foreach (var caseName in new[] { "points_only", "boxes_only", "points_boxes_masks" })
            {
                string caseDir = Path.Combine(rootDir, caseName);
                string inputPath = Path.Combine(caseDir, "input.safetensors");
                string outputPath = Path.Combine(caseDir, "output_net.safetensors");

                var inputs = Safetensors.LoadStateDict(inputPath);

                Tuple<Tensor, Tensor>? points = null;
                if (inputs.ContainsKey("point_coords"))
                {
                    points = Tuple.Create(inputs["point_coords"], inputs["point_labels"]);
                }
                Tensor? boxes = inputs.TryGetValue("boxes", out var b) ? b : null;
                Tensor? masks = inputs.TryGetValue("masks", out var m) ? m : null;

                var (sparse, dense) = model.forward(points, boxes, masks);

                var outDict = new Dictionary<string, Tensor>
                {
                    ["sparse_embeddings"] = sparse.contiguous(),
                    ["dense_embeddings"] = dense.contiguous(),
                };
                Safetensors.SaveStateDict(outputPath, outDict);

                Console.WriteLine($"[prompt_encoder:{caseName}] sparse={string.Join('x', sparse.shape)} dense={string.Join('x', dense.shape)}");
                Console.WriteLine($"  已保存: {outputPath}");
            }

            return 0;
        }

        private static int RunMaskDecoder(string rootDir)
        {
            using var _ = NewDisposeScope();
            using var noGrad = no_grad();

            var transformer = new TwoWayTransformer(depth: 2, embeddingDim: 256, numHeads: 8, mlpDim: 2048);
            var model = new MaskDecoder(
                transformerDim: 256,
                transformer: transformer,
                numMultimaskOutputs: 3,
                iouHeadDepth: 3,
                iouHeadHiddenDim: 256,
                useHighResFeatures: true,
                iouPredictionUseSigmoid: true,
                dynamicMultimaskViaStability: true,
                predObjScores: true,
                predObjScoresMlp: true,
                useMultimaskTokenForObjPtr: true);
            model.eval();
            model.load_safetensors(Path.Combine(rootDir, "weights.safetensors"));

            foreach (var (caseName, multimaskOutput) in new[] { ("single_mask", false), ("multi_mask", true) })
            {
                string caseDir = Path.Combine(rootDir, caseName);
                var inputs = Safetensors.LoadStateDict(Path.Combine(caseDir, "input.safetensors"));

                var (masks, iouPred, samTokensOut, objectScoreLogits) = model.forward(
                    inputs["image_embeddings"],
                    inputs["image_pe"],
                    inputs["sparse_prompt_embeddings"],
                    inputs["dense_prompt_embeddings"],
                    multimaskOutput,
                    false,
                    new[] { inputs["feat_s0"], inputs["feat_s1"] });

                var outDict = new Dictionary<string, Tensor>
                {
                    ["masks"] = masks.contiguous(),
                    ["iou_pred"] = iouPred.contiguous(),
                    ["sam_tokens_out"] = samTokensOut.contiguous(),
                    ["object_score_logits"] = objectScoreLogits.contiguous(),
                };
                string outputPath = Path.Combine(caseDir, "output_net.safetensors");
                Safetensors.SaveStateDict(outputPath, outDict);

                Console.WriteLine($"[mask_decoder:{caseName}] masks={string.Join('x', masks.shape)} iou_pred={string.Join('x', iouPred.shape)}");
                Console.WriteLine($"  已保存: {outputPath}");
            }

            return 0;
        }

        private static int RunSam2Image(string dir, int imageSize)
        {
            using var _ = NewDisposeScope();
            using var noGrad = no_grad();

            const int dModel = 256;
            Hiera trunk = BuildHiera("tiny");
            var positionEncoding = new PositionEmbeddingSine(numPosFeats: dModel, normalize: true);
            var neck = new FpnNeck(
                positionEncoding: positionEncoding,
                dModel: dModel,
                backboneChannelList: trunk.ChannelList,
                fpnTopDownLevels: new[] { 2, 3 },
                fpnInterpModel: "nearest");
            var imageEncoder = new ImageEncoder(trunk, neck, scalp: 1);

            var promptEncoder = new SAMTorchSharp.Modeling.Sam2.PromptEncoder(
                embed_dim: dModel,
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
            model.eval();

            model.load_safetensors(Path.Combine(dir, "weights.safetensors"));

            var inputs = Safetensors.LoadStateDict(Path.Combine(dir, "input.safetensors"));
            Tensor x = inputs["x"];
            Tensor pointCoords = inputs["point_coords"];
            Tensor pointLabels = inputs["point_labels"];

            var backboneOut = model.ForwardImage(x);
            var (visionFeats, featSizes) = model.PrepareBackboneFeatures(backboneOut);
            visionFeats[^1] = visionFeats[^1] + model.no_mem_embed;

            var bbFeatSizes = new (long, long)[]
            {
                (imageSize / 4, imageSize / 4),
                (imageSize / 8, imageSize / 8),
                (imageSize / 16, imageSize / 16),
            };

            var feats = new Tensor[visionFeats.Count];
            for (int i = 0; i < visionFeats.Count; i++)
            {
                int reversedIdx = visionFeats.Count - 1 - i;
                var (h, w) = bbFeatSizes[reversedIdx];
                Tensor feat = visionFeats[reversedIdx];
                feats[reversedIdx] = feat.permute(1, 2, 0).view(1, -1, h, w);
            }

            Tensor imageEmbed = feats[^1];
            var highResFeats = feats.Take(feats.Length - 1).ToList();

            var (sparseEmbeddings, denseEmbeddings) = promptEncoder.forward(
                Tuple.Create(pointCoords, pointLabels), null, null);

            var (lowResMasks, iouPredictions, _, _) = maskDecoder.forward(
                imageEmbed,
                promptEncoder.get_dense_pe(),
                sparseEmbeddings,
                denseEmbeddings,
                true,
                false,
                highResFeats);

            var outDict = new Dictionary<string, Tensor>
            {
                ["image_embed"] = imageEmbed.contiguous(),
                ["high_res_feat_0"] = highResFeats[0].contiguous(),
                ["high_res_feat_1"] = highResFeats[1].contiguous(),
                ["low_res_masks"] = lowResMasks.contiguous(),
                ["iou_predictions"] = iouPredictions.contiguous(),
            };
            string outputPath = Path.Combine(dir, "output_net.safetensors");
            Safetensors.SaveStateDict(outputPath, outDict);

            Console.WriteLine($"[sam2_image] image_embed={string.Join('x', imageEmbed.shape)}");
            foreach (var kv in outDict)
            {
                Console.WriteLine($"  {kv.Key}: {string.Join('x', kv.Value.shape)}");
            }
            Console.WriteLine($"已保存: {outputPath}");

            return 0;
        }

        private static int RunSam2Real(string dir, string variant)
        {
            using var _ = NewDisposeScope();
            using var noGrad = no_grad();

            const int imageSize = 1024;
            const int dModel = 256;

            Hiera trunk = variant switch
            {
                "tiny" => new Hiera(
                    embedDim: 96, numHeads: 1,
                    stages: new[] { 1, 2, 7, 2 },
                    globalAttBlocks: new[] { 5, 7, 9 },
                    windowPosEmbedBkgSpatialSize: (7, 7),
                    windowSpec: new[] { 8, 4, 14, 7 }),
                "small" => new Hiera(
                    embedDim: 96, numHeads: 1,
                    stages: new[] { 1, 2, 11, 2 },
                    globalAttBlocks: new[] { 7, 10, 13 },
                    windowPosEmbedBkgSpatialSize: (7, 7),
                    windowSpec: new[] { 8, 4, 14, 7 }),
                _ => throw new ArgumentException($"未知 variant: {variant}")
            };

            var positionEncoding = new PositionEmbeddingSine(numPosFeats: dModel, normalize: true);
            var neck = new FpnNeck(
                positionEncoding: positionEncoding,
                dModel: dModel,
                backboneChannelList: trunk.ChannelList,
                fpnTopDownLevels: new[] { 2, 3 },
                fpnInterpModel: "nearest");
            var imageEncoder = new ImageEncoder(trunk, neck, scalp: 1);

            var promptEncoder = new SAMTorchSharp.Modeling.Sam2.PromptEncoder(
                embed_dim: dModel,
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
                predObjScores: true,
                predObjScoresMlp: true,
                useMultimaskTokenForObjPtr: true);

            var model = new Sam2Base(
                imageEncoder, promptEncoder, maskDecoder,
                imageSize: imageSize, backboneStride: 16,
                useHighResFeaturesInSam: true, directlyAddNoMemEmbed: true);
            model.eval();

            model.load_safetensors(Path.Combine(dir, "weights.safetensors"));

            var inputs = Safetensors.LoadStateDict(Path.Combine(dir, "input.safetensors"));
            Tensor preprocessed = inputs["preprocessed_image"];
            Tensor pointCoordsOrig = inputs["point_coords_orig"]; // 1xNx2, (x,y) in orig pixel coords
            Tensor pointLabels = inputs["point_labels"];
            Tensor origHw = inputs["orig_hw"]; // [[h, w]]

            double origH = origHw[0, 0].item<float>();
            double origW = origHw[0, 1].item<float>();

            // 对应 SAM2Transforms.transform_coords(normalize=True): coords/[w,h] * resolution
            Tensor pointCoords1024 = pointCoordsOrig.clone();
            pointCoords1024[TensorIndex.Ellipsis, TensorIndex.Single(0)] =
                pointCoords1024[TensorIndex.Ellipsis, TensorIndex.Single(0)] / origW * imageSize;
            pointCoords1024[TensorIndex.Ellipsis, TensorIndex.Single(1)] =
                pointCoords1024[TensorIndex.Ellipsis, TensorIndex.Single(1)] / origH * imageSize;

            var backboneOut = model.ForwardImage(preprocessed);
            var (visionFeats, _) = model.PrepareBackboneFeatures(backboneOut);
            visionFeats[^1] = visionFeats[^1] + model.no_mem_embed;

            var bbFeatSizes = new (long, long)[] { (256, 256), (128, 128), (64, 64) };
            var feats = new Tensor[visionFeats.Count];
            for (int i = 0; i < visionFeats.Count; i++)
            {
                int reversedIdx = visionFeats.Count - 1 - i;
                var (h, w) = bbFeatSizes[reversedIdx];
                feats[reversedIdx] = visionFeats[reversedIdx].permute(1, 2, 0).view(1, -1, h, w);
            }
            Tensor imageEmbed = feats[^1];
            var highResFeats = feats.Take(feats.Length - 1).ToList();

            var (sparseEmbeddings, denseEmbeddings) = promptEncoder.forward(
                Tuple.Create(pointCoords1024, pointLabels), null, null);

            var (lowResMasksRaw, iouPredictions, _, _) = maskDecoder.forward(
                imageEmbed, promptEncoder.get_dense_pe(), sparseEmbeddings, denseEmbeddings,
                true, false, highResFeats);

            Tensor lowResMasks = lowResMasksRaw.clamp(-32.0, 32.0);

            // 对应 SAM2Transforms.postprocess_masks：双线性插值回原图分辨率（不含孔洞填充，max_hole_area=0）
            Tensor masks = functional.interpolate(
                lowResMasksRaw.to(ScalarType.Float32),
                size: new long[] { (long)origH, (long)origW },
                mode: InterpolationMode.Bilinear,
                align_corners: false);

            // Python 端 predictor.predict() 会 squeeze(0) 去掉 batch 维，这里保持输出对齐以便直接比对。
            var outDict = new Dictionary<string, Tensor>
            {
                ["image_embed"] = imageEmbed.contiguous(),
                ["high_res_feat_0"] = highResFeats[0].contiguous(),
                ["high_res_feat_1"] = highResFeats[1].contiguous(),
                ["low_res_masks"] = lowResMasks.squeeze(0).contiguous(),
                ["iou_predictions"] = iouPredictions.squeeze(0).contiguous(),
                ["masks"] = masks.squeeze(0).contiguous(),
            };
            string outputPath = Path.Combine(dir, "output_net.safetensors");
            Safetensors.SaveStateDict(outputPath, outDict);

            Console.WriteLine($"[sam2_real:{variant}] image_embed={string.Join('x', imageEmbed.shape)}");
            foreach (var kv in outDict)
            {
                Console.WriteLine($"  {kv.Key}: {string.Join('x', kv.Value.shape)}");
            }
            Console.WriteLine($"已保存: {outputPath}");

            return 0;
        }

        private static int RunRopeAttention(string rootDir)
        {
            using var _ = NewDisposeScope();
            using var noGrad = no_grad();

            // self_attn 用例
            {
                string caseDir = Path.Combine(rootDir, "self_attn");
                var model = new RoPEAttention(
                    embeddingDim: 256, numHeads: 1, downsampleRate: 1,
                    ropeTheta: 10000.0, featSizes: (64, 64));
                model.eval();
                model.load_safetensors(Path.Combine(caseDir, "weights.safetensors"));

                var inputs = Safetensors.LoadStateDict(Path.Combine(caseDir, "input.safetensors"));
                Tensor outT = model.forward(inputs["q"], inputs["k"], inputs["v"], 0);

                Safetensors.SaveStateDict(Path.Combine(caseDir, "output_net.safetensors"),
                    new Dictionary<string, Tensor> { ["output"] = outT.contiguous() });
                Console.WriteLine($"[rope:self_attn] out={string.Join('x', outT.shape)}");
            }

            // cross_attn 用例
            {
                string caseDir = Path.Combine(rootDir, "cross_attn");
                var model = new RoPEAttention(
                    embeddingDim: 256, numHeads: 1, downsampleRate: 1,
                    ropeTheta: 10000.0, featSizes: (64, 64),
                    ropeKRepeat: true, kvInDim: 64);
                model.eval();
                model.load_safetensors(Path.Combine(caseDir, "weights.safetensors"));

                var inputs = Safetensors.LoadStateDict(Path.Combine(caseDir, "input.safetensors"));
                Tensor outT = model.forward(inputs["q"], inputs["k"], inputs["v"], 4);

                Safetensors.SaveStateDict(Path.Combine(caseDir, "output_net.safetensors"),
                    new Dictionary<string, Tensor> { ["output"] = outT.contiguous() });
                Console.WriteLine($"[rope:cross_attn] out={string.Join('x', outT.shape)}");
            }

            return 0;
        }

        private static int RunMemoryEncoder(string dir)
        {
            using var _ = NewDisposeScope();
            using var noGrad = no_grad();

            var maskDownsampler = new MaskDownSampler(kernelSize: 3, stride: 2, padding: 1);
            var fuser = new Fuser(
                layer: new CXBlock(dim: 256, kernelSize: 7, padding: 3, layerScaleInitValue: 1e-6, useDwconv: true),
                numLayers: 2);
            var positionEncoding = new PositionEmbeddingSine(numPosFeats: 64, normalize: true);
            var model = new MemoryEncoder(outDim: 64, maskDownsampler: maskDownsampler, fuser: fuser, positionEncoding: positionEncoding);
            model.eval();
            model.load_safetensors(Path.Combine(dir, "weights.safetensors"));

            var inputs = Safetensors.LoadStateDict(Path.Combine(dir, "input.safetensors"));
            var output = model.forward(inputs["pix_feat"], inputs["masks"], skipMaskSigmoid: false);

            var outDict = new Dictionary<string, Tensor>
            {
                ["vision_features"] = output.VisionFeatures.contiguous(),
                ["vision_pos_enc_0"] = output.VisionPosEnc[0].contiguous(),
            };
            string outputPath = Path.Combine(dir, "output_net.safetensors");
            Safetensors.SaveStateDict(outputPath, outDict);

            Console.WriteLine($"[memory_encoder] vision_features={string.Join('x', output.VisionFeatures.shape)}");
            Console.WriteLine($"已保存: {outputPath}");

            return 0;
        }

        private static Dictionary<string, string> ParseOptions(string[] args)
        {
            var dict = new Dictionary<string, string>();
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i].StartsWith("--") && i + 1 < args.Length)
                {
                    dict[args[i][2..]] = args[i + 1];
                    i++;
                }
            }
            return dict;
        }

        private static void PrintUsage()
        {
            Console.WriteLine("Sam2VectorRunner - 运行.NET版SAM2模块并导出输出safetensors用于数值校验");
            Console.WriteLine();
            Console.WriteLine("用法:");
            Console.WriteLine("  Sam2VectorRunner hiera --variant tiny --dir <testdata/hiera_tiny>");
        }
    }
}
