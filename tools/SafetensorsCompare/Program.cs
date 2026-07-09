using TorchSharp;
using TorchSharp.PyBridge;
using static TorchSharp.torch;

namespace SafetensorsCompare
{
    /// <summary>
    /// 命令行工具：比较两个 safetensors 文件中各个张量的数值差异。
    /// 用于 SAM2/SAM3 从 Python 迁移到 .NET(TorchSharp) 过程中的逐模块数值校验：
    ///   - expected: Python 参考实现导出的输出（如 output_py.safetensors）
    ///   - actual:   .NET 迁移实现导出的输出（如 output_net.safetensors）
    ///
    /// 用法:
    ///   SafetensorsCompare &lt;expected.safetensors&gt; &lt;actual.safetensors&gt; [--atol 1e-4] [--rtol 1e-4]
    ///
    /// 退出码:
    ///   0  - 所有张量都在容差范围内一致
    ///   1  - 存在超出容差的差异，或 key 缺失/形状不匹配
    ///   2  - 参数错误 / 文件不存在
    /// </summary>
    public static class Program
    {
        public static int Main(string[] args)
        {
            string? expectedPath = null;
            string? actualPath = null;
            double atol = 1e-4;
            double rtol = 1e-4;
            bool verbose = false;

            var positional = new List<string>();
            for (int i = 0; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--atol":
                        atol = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
                        break;
                    case "--rtol":
                        rtol = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
                        break;
                    case "--verbose":
                    case "-v":
                        verbose = true;
                        break;
                    case "--help":
                    case "-h":
                        PrintUsage();
                        return 0;
                    default:
                        positional.Add(args[i]);
                        break;
                }
            }

            if (positional.Count < 2)
            {
                PrintUsage();
                return 2;
            }

            expectedPath = positional[0];
            actualPath = positional[1];

            if (!File.Exists(expectedPath))
            {
                Console.Error.WriteLine($"[错误] 找不到期望值文件: {expectedPath}");
                return 2;
            }
            if (!File.Exists(actualPath))
            {
                Console.Error.WriteLine($"[错误] 找不到实际值文件: {actualPath}");
                return 2;
            }

            Dictionary<string, Tensor> expected = Safetensors.LoadStateDict(expectedPath);
            Dictionary<string, Tensor> actual = Safetensors.LoadStateDict(actualPath);

            var allKeys = expected.Keys.Union(actual.Keys).OrderBy(k => k, StringComparer.Ordinal).ToList();

            bool allOk = true;
            var rows = new List<(string Key, string Status, string Detail)>();

            foreach (var key in allKeys)
            {
                if (!expected.ContainsKey(key))
                {
                    allOk = false;
                    rows.Add((key, "缺失(期望值无)", "该key仅存在于actual文件中"));
                    continue;
                }
                if (!actual.ContainsKey(key))
                {
                    allOk = false;
                    rows.Add((key, "缺失(实际值无)", "该key仅存在于expected文件中"));
                    continue;
                }

                Tensor e = expected[key];
                Tensor a = actual[key];

                if (!e.shape.SequenceEqual(a.shape))
                {
                    allOk = false;
                    rows.Add((key, "形状不匹配", $"expected={string.Join('x', e.shape)} actual={string.Join('x', a.shape)}"));
                    continue;
                }

                using var ed = e.to(ScalarType.Float64);
                using var ad = a.to(ScalarType.Float64);
                using var diff = (ed - ad).abs();

                double maxAbsDiff = diff.numel() > 0 ? diff.max().item<double>() : 0.0;
                using var denom = ed.abs().clamp_min(1e-8);
                using var relDiffT = diff / denom;
                double maxRelDiff = relDiffT.numel() > 0 ? relDiffT.max().item<double>() : 0.0;

                bool within = maxAbsDiff <= atol + rtol * ed.abs().max().item<double>();
                // 使用更严格的逐元素判断: |diff| <= atol + rtol*|expected|
                using var tol = atol + rtol * ed.abs();
                using var okMask = diff <= tol;
                bool elementWiseOk = okMask.numel() == 0 || okMask.all().item<bool>();

                if (!elementWiseOk)
                {
                    allOk = false;
                    rows.Add((key, "数值不一致", $"max_abs_diff={maxAbsDiff:E4} max_rel_diff={maxRelDiff:E4} shape={string.Join('x', e.shape)}"));
                }
                else
                {
                    rows.Add((key, "一致", $"max_abs_diff={maxAbsDiff:E4} max_rel_diff={maxRelDiff:E4}"));
                }
            }

            foreach (var (key, status, detail) in rows)
            {
                if (status == "一致" && !verbose) continue;
                Console.WriteLine($"[{status}] {key}: {detail}");
            }

            int okCount = rows.Count(r => r.Status == "一致");
            Console.WriteLine();
            Console.WriteLine($"总计 {rows.Count} 个张量, 一致 {okCount} 个, 不一致/缺失 {rows.Count - okCount} 个 (atol={atol}, rtol={rtol})");
            Console.WriteLine(allOk ? "结果: 通过" : "结果: 失败");

            return allOk ? 0 : 1;
        }

        private static void PrintUsage()
        {
            Console.WriteLine("SafetensorsCompare - 比较两个safetensors文件中张量的数值一致性");
            Console.WriteLine();
            Console.WriteLine("用法:");
            Console.WriteLine("  SafetensorsCompare <expected.safetensors> <actual.safetensors> [--atol 1e-4] [--rtol 1e-4] [--verbose]");
        }
    }
}
