using SAMTorchSharp.Modeling.Sam2;
using TorchSharp;
using TorchSharp.PyBridge;
using static TorchSharp.torch;

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

            Hiera model = variant switch
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
