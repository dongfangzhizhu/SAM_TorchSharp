using SAMTorchSharp.Modeling.Sam2;
using TorchSharp;
using TorchSharp.Modules;
using TorchSharp.PyBridge;
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
