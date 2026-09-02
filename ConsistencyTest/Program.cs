using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using SAMTorchSharp.Modeling.Sam2;
using SAMTorchSharp.Modeling.Sam3;
using static TorchSharp.torch;

namespace ConsistencyTest;

internal static class Program
{
    private const int RuntimeError = 1;
    private const int UsageError = 2;
    private const int ValidationError = 3;

    public static int Main(string[] args)
    {
        try
        {
            if (args.Length == 0 || IsHelp(args[0]))
            {
                PrintHelp();
                return 0;
            }

            var options = CliOptions.Parse(args.Skip(1).ToArray());
            return args[0].ToLowerInvariant() switch
            {
                "info" => RunInfo(options),
                "compare" => RunCompare(options),
                "sam2-checkpoint" => RunSam2Checkpoint(options),
                "sam2-image" => RunSam2Image(options),
                "sam2-video" => RunSam2Video(options),
                "sam3-checkpoint" => RunSam3Checkpoint(options),
                "sam3-run" => RunSam3(options),
                "self-test" => RunSelfTest(options),
                _ => Fail($"Unknown command '{args[0]}'. Run with --help for usage.", UsageError),
            };
        }
        catch (CliException ex)
        {
            return Fail(ex.Message, UsageError);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"ERROR: {ex.GetType().Name}: {ex.Message}");
            return RuntimeError;
        }
    }

    private static int RunInfo(CliOptions options)
    {
        options.EnsureNoUnused();
        Console.WriteLine("SAM_TorchSharp consistency CLI");
        Console.WriteLine("Target framework: net8.0");
        Console.WriteLine("Configured native runtime: libtorch-cpu-win-x64");
        Console.WriteLine($"Process architecture: {System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture}");
        Console.WriteLine($"Working directory: {Environment.CurrentDirectory}");
        return 0;
    }

    private static int RunCompare(CliOptions options)
    {
        var expectedPath = options.RequirePath("expected");
        var actualPath = options.RequirePath("actual");
        var absoluteTolerance = options.GetDouble("atol", 1e-5, min: 0);
        var relativeTolerance = options.GetDouble("rtol", 1e-4, min: 0);
        options.EnsureNoUnused();

        var expected = NpyFile.ReadFloat32(expectedPath);
        var actual = NpyFile.ReadFloat32(actualPath);
        var result = NumericComparison.Compare(expected, actual, absoluteTolerance, relativeTolerance);
        PrintComparison(result, expectedPath, actualPath);
        return result.Passed ? 0 : ValidationError;
    }

    private static int RunSam3(CliOptions options)
    {
        var checkpointPath = options.RequirePath("checkpoint");
        var imagePath = options.GetOptionalPath("image");
        var pointsPath = options.GetOptionalPath("points");
        var labelsPath = options.GetOptionalPath("labels");
        var boxPath = options.GetOptionalPath("box");
        var outputDirectory = Path.GetFullPath(options.Get("output") ?? Path.Combine(Environment.CurrentDirectory, "sam3-output"));
        var caption = options.Get("caption") ?? "a dog";
        var deviceName = (options.Get("device") ?? "cpu").ToLowerInvariant();
        var seed = options.GetInt("seed", 42);
        var minCoverage = options.GetDouble("min-coverage", 0, min: 0, max: 100);
        options.EnsureNoUnused();

        if (deviceName != "cpu")
            throw new CliException("Only --device cpu is supported by the configured libtorch-cpu-win-x64 runtime.");

        var checkpointFormat = GetSam3CheckpointFormat(checkpointPath);
        var geometryInputs = Sam3ImageCommand.LoadGeometryInputs(pointsPath, labelsPath, boxPath);
        var geometricPrompt = Sam3ImageCommand.CreateGeometryPrompt(geometryInputs);

        Directory.CreateDirectory(outputDirectory);
        Console.WriteLine("Building SAM3 detector model...");
        using var model = new BuildSam3New().Build();

        Console.WriteLine($"Loading checkpoint: {checkpointPath}");
        int loaded;
        int skipped;
        int missing;
        double coverage;
        if (checkpointFormat == Sam3CheckpointFormat.OfficialSafetensors)
        {
            var report = new Sam3CheckpointLoaderNew().LoadModelWithReport(model, checkpointPath, CPU);
            WriteSam3CheckpointReport(outputDirectory, report);
            (loaded, skipped, missing) = (report.LoadedKeys.Count, report.SkippedKeys.Count,
                report.MissingKeys.Count + report.ShapeMismatches.Count);
            coverage = report.Coverage;
        }
        else
        {
            var report = new Sam3CheckpointLoaderBinary().LoadModelWithReport(model, checkpointPath, CPU);
            WriteSam3CheckpointReport(outputDirectory, report);
            (loaded, skipped, missing) = (report.LoadedKeys.Count, report.SkippedKeys.Count,
                report.MissingKeys.Count + report.ShapeMismatches.Count);
            coverage = report.Coverage;
        }

        Console.WriteLine($"Checkpoint: loaded={loaded}, skipped={skipped}, missing={missing}, coverage={coverage:F2}%");
        if (coverage < minCoverage)
            return Fail($"Checkpoint coverage {coverage:F2}% is below required {minCoverage:F2}%.", ValidationError);

        manual_seed(seed);
        using var input = imagePath is null
            ? randn(new long[] { 1, 3, 1008, 1008 }, dtype: ScalarType.Float32, device: CPU)
            : Sam3ImageCommand.ToTensor(Sam3ImageCommand.LoadImage(imagePath));
        var stopwatch = Stopwatch.StartNew();
        var outputs = model.Forward(input, new[] { caption }, geometricPrompt);
        stopwatch.Stop();

        try
        {
            Sam3ImageCommand.ValidateOutputs(outputs);
            foreach (var (name, tensor) in outputs)
            {
                var outputPath = Path.Combine(outputDirectory, $"{name}.npy");
                NpyFile.WriteFloat32(outputPath, tensor);
                Console.WriteLine($"{name}: [{string.Join(", ", tensor.shape)}] -> {outputPath}");
            }

            var summary = new
            {
                checkpoint = checkpointPath,
                image = imagePath,
                input = imagePath is null ? "seeded-random" : "preprocessed-image",
                device = deviceName,
                caption,
                pointCount = geometryInputs.Points?.Shape[0] ?? 0,
                hasBox = geometryInputs.Box is not null,
                seed,
                loaded,
                skipped,
                missing,
                coverage,
                inferenceMilliseconds = stopwatch.ElapsedMilliseconds,
                outputs = outputs.ToDictionary(pair => pair.Key, pair => pair.Value.shape),
            };
            File.WriteAllText(
                Path.Combine(outputDirectory, "summary.json"),
                JsonSerializer.Serialize(summary, new JsonSerializerOptions { WriteIndented = true }));
        }
        finally
        {
            foreach (var tensor in outputs.Values)
                tensor.Dispose();
            geometricPrompt.point_embeddings?.Dispose();
            geometricPrompt.point_labels?.Dispose();
            geometricPrompt.box_embeddings?.Dispose();
        }

        Console.WriteLine($"Inference completed in {stopwatch.ElapsedMilliseconds} ms.");
        return 0;
    }

    internal static Sam3CheckpointFormat GetSam3CheckpointFormat(string checkpointPath)
    {
        try
        {
            return Sam3CheckpointLoaderNew.DetectFormat(checkpointPath);
        }
        catch (NotSupportedException exception)
        {
            throw new CliException($"--checkpoint must be an existing .safetensors, converted .bin, or .pt file with a converted sibling .bin. {exception.Message}");
        }
    }

    private static int RunSam3Checkpoint(CliOptions options)
    {
        var checkpointPath = options.RequirePath("checkpoint");
        var outputDirectory = Path.GetFullPath(options.Get("output") ?? Environment.CurrentDirectory);
        var minCoverage = options.GetDouble("min-coverage", 100, min: 0, max: 100);
        options.EnsureNoUnused();

        Console.WriteLine("Building SAM3 detector model...");
        using var model = new BuildSam3New().Build();
        Console.WriteLine($"Loading checkpoint: {checkpointPath}");
        var report = Sam3CheckpointLoaderNew.DetectFormat(checkpointPath) switch
        {
            Sam3CheckpointFormat.OfficialSafetensors =>
                new Sam3CheckpointLoaderNew().LoadModelWithReport(model, checkpointPath, CPU),
            Sam3CheckpointFormat.ConvertedBinary =>
                new Sam3CheckpointLoaderBinary().LoadModelWithReport(model, checkpointPath, CPU),
            _ => throw new UnreachableException(),
        };
        WriteSam3CheckpointReport(outputDirectory, report);
        Console.WriteLine($"Checkpoint: loaded={report.LoadedKeys.Count}, skipped={report.SkippedKeys.Count}, " +
                          $"missing={report.MissingKeys.Count}, shape_mismatch={report.ShapeMismatches.Count}, " +
                          $"coverage={report.Coverage:F2}%");
        return report.Coverage >= minCoverage ? 0 :
            Fail($"Checkpoint coverage {report.Coverage:F2}% is below required {minCoverage:F2}%.", ValidationError);
    }

    internal static string WriteSam3CheckpointReport(
        string outputDirectory,
        Sam3CheckpointLoadReport report)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(outputDirectory);
        ArgumentNullException.ThrowIfNull(report);

        var directory = Path.GetFullPath(outputDirectory);
        Directory.CreateDirectory(directory);
        var reportPath = Path.Combine(directory, "checkpoint-report.json");
        File.WriteAllText(reportPath, JsonSerializer.Serialize(report, new JsonSerializerOptions
        {
            WriteIndented = true,
            Converters = { new JsonStringEnumConverter() },
        }));
        Console.WriteLine($"Checkpoint report: {reportPath}");
        return reportPath;
    }

    private static int RunSam2Checkpoint(CliOptions options)
    {
        var checkpointPath = options.RequirePath("checkpoint");
        var variantName = options.Get("variant") ?? throw new CliException("Missing required option '--variant'.");
        var outputPath = Path.GetFullPath(options.Get("output") ?? "sam2-checkpoint-summary.json");
        var strict = options.GetBool("strict", true);
        options.EnsureNoUnused();

        var variant = variantName.ToLowerInvariant() switch
        {
            "sam2-tiny" => Sam2ModelVariant.Sam2Tiny,
            "sam2-small" => Sam2ModelVariant.Sam2Small,
            "sam2.1-tiny" => Sam2ModelVariant.Sam21Tiny,
            "sam2.1-small" => Sam2ModelVariant.Sam21Small,
            _ => throw new CliException("--variant must be sam2-tiny, sam2-small, sam2.1-tiny, or sam2.1-small."),
        };

        Console.WriteLine($"Building {variantName} inference model...");
        using var model = Sam2ModelBuilder.Build(variant);
        var report = Sam2CheckpointLoader.Load(model, checkpointPath, strict);
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        File.WriteAllText(outputPath, JsonSerializer.Serialize(report, new JsonSerializerOptions
        {
            WriteIndented = true,
            Converters = { new JsonStringEnumConverter() },
        }));
        Console.WriteLine($"loaded={report.LoadedKeys.Count}, missing={report.MissingKeys.Count}, " +
                          $"unexpected={report.UnexpectedKeys.Count}, shape_mismatch={report.ShapeMismatches.Count}, " +
                          $"coverage={report.Coverage:F2}%");
        Console.WriteLine($"Report: {outputPath}");
        return report.IsComplete ? 0 : ValidationError;
    }

    private static int RunSam2Video(CliOptions options)
    {
        var checkpointPath = options.RequirePath("checkpoint");
        var variantName = options.Get("variant") ?? throw new CliException("Missing required option '--variant'.");
        var vectorDirectory = Path.GetFullPath(options.Get("vectors") ?? throw new CliException("Missing required option '--vectors'."));
        if (!Directory.Exists(vectorDirectory)) throw new CliException($"Vector directory does not exist: {vectorDirectory}");
        var reportPath = Path.GetFullPath(options.Get("output") ?? Path.Combine(vectorDirectory, "parity_cs.json"));
        var absoluteTolerance = options.GetDouble("atol", 2e-2, min: 0);
        var relativeTolerance = options.GetDouble("rtol", 1e-3, min: 0);
        options.EnsureNoUnused();
        return Sam2VideoCommand.Run(
            Sam2ImageCommand.ParseVariant(variantName), variantName.ToLowerInvariant(), checkpointPath,
            vectorDirectory, reportPath, absoluteTolerance, relativeTolerance);
    }

    private static int RunSam2Image(CliOptions options)
    {
        var checkpointPath = options.RequirePath("checkpoint");
        var variantName = options.Get("variant") ?? throw new CliException("Missing required option '--variant'.");
        var imagePath = options.RequirePath("image");
        var pointsPath = options.GetOptionalPath("points");
        var labelsPath = options.GetOptionalPath("labels");
        var boxPath = options.GetOptionalPath("box");
        var maskInputPath = options.GetOptionalPath("mask-input");
        var outputDirectory = Path.GetFullPath(options.Get("output") ?? Path.Combine(Environment.CurrentDirectory, "sam2-image-output"));
        var deviceName = (options.Get("device") ?? "cpu").ToLowerInvariant();
        var multimask = options.GetBool("multimask", true);
        var returnLogits = options.GetBool("return-logits", false);
        options.EnsureNoUnused();

        if (deviceName != "cpu")
            throw new CliException("Only --device cpu is supported by the configured libtorch-cpu-win-x64 runtime.");

        var variant = Sam2ImageCommand.ParseVariant(variantName);
        var inputs = Sam2ImageCommand.LoadInputs(imagePath, pointsPath, labelsPath, boxPath, maskInputPath);
        return Sam2ImageCommand.Run(
            variant, variantName.ToLowerInvariant(), checkpointPath, outputDirectory,
            inputs, multimask, returnLogits);
    }

    private static int RunSelfTest(CliOptions options)
    {
        options.EnsureNoUnused();
        var directory = Path.Combine(Path.GetTempPath(), $"sam-consistency-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            var path = Path.Combine(directory, "roundtrip.npy");
            var source = new NpyArray(new[] { 2, 2 }, new[] { 0f, 1f, -2.5f, 4.25f });
            NpyFile.WriteFloat32(path, source);
            var restored = NpyFile.ReadFloat32(path);
            if (!NumericComparison.Compare(source, restored, 0, 0).Passed)
                return Fail("NPY round-trip self-test failed.", ValidationError);

            var close = new NpyArray(new[] { 2, 2 }, new[] { 0f, 1.00001f, -2.5f, 4.25f });
            if (!NumericComparison.Compare(source, close, 1e-4, 0).Passed)
                return Fail("Numeric tolerance self-test failed.", ValidationError);

            Console.WriteLine("Self-test passed: NPY v1 float32 round-trip and numeric tolerances are valid.");
            return 0;
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static void PrintComparison(ComparisonResult result, string expectedPath, string actualPath)
    {
        Console.WriteLine($"Expected: {expectedPath}");
        Console.WriteLine($"Actual:   {actualPath}");
        Console.WriteLine($"Shape: [{string.Join(", ", result.Shape)}], count={result.Count}");
        Console.WriteLine($"max_abs_error={result.MaxAbsoluteError.ToString("G9", CultureInfo.InvariantCulture)}");
        Console.WriteLine($"mean_abs_error={result.MeanAbsoluteError.ToString("G9", CultureInfo.InvariantCulture)}");
        Console.WriteLine($"max_rel_error={result.MaxRelativeError.ToString("G9", CultureInfo.InvariantCulture)}");
        Console.WriteLine($"mismatched={result.MismatchCount}, non_finite={result.NonFiniteCount}");
        Console.WriteLine(result.Passed ? "PASS" : "FAIL");
    }

    private static bool IsHelp(string value) => value is "-h" or "--help" or "help";

    private static int Fail(string message, int exitCode)
    {
        Console.Error.WriteLine($"ERROR: {message}");
        return exitCode;
    }

    private static void PrintHelp()
    {
        Console.WriteLine("""
            SAM_TorchSharp consistency CLI

            Usage:
              ConsistencyTest info
              ConsistencyTest self-test
              ConsistencyTest compare --expected <file.npy> --actual <file.npy> [--atol 1e-5] [--rtol 1e-4]
              ConsistencyTest sam2-checkpoint --variant <name> --checkpoint <model.pt|model.safetensors> [options]
              ConsistencyTest sam2-image --variant <name> --checkpoint <model.pt|model.safetensors> --image <image.npy> [prompts] [options]
              ConsistencyTest sam2-video --variant <name> --checkpoint <model.pt|model.safetensors> --vectors <directory> [options]
              ConsistencyTest sam3-checkpoint --checkpoint <model.safetensors> [options]
              ConsistencyTest sam3-run --checkpoint <model.safetensors|model.bin> [options]

            sam2-checkpoint options:
              --variant <name>          sam2-tiny, sam2-small, sam2.1-tiny, or sam2.1-small
              --output <file.json>      Loading report (default: ./sam2-checkpoint-summary.json)
              --strict <true|false>     Require zero missing/unexpected/mismatched keys (default: true)

            sam2-image prompts and options:
              --image <file.npy>        Required float32 HWC [H,W,3] RGB image in the [0,1] range
              --points <file.npy>       Optional float32 [N,2] original-pixel (x,y) coordinates
              --labels <file.npy>       Required with --points: float32 [N], values 0 or 1
              --box <file.npy>          Optional float32 [4] or [2,2] x0,y0,x1,y1 box
              --mask-input <file.npy>   Optional float32 [1,256,256] logits for refinement
              --output <directory>      masks.npy, scores.npy, low_res_logits.npy, summary.json
              --device cpu              Execution device; CPU is currently the only supported runtime
              --multimask <true|false>  Return three candidate masks instead of one (default: true)
              --return-logits <bool>    Return full-resolution logits instead of binary masks (default: false)

            sam3-checkpoint options:
              --output <directory>       Report directory (default: current directory)
              --min-coverage <percent>   Fail below loadable-tensor coverage (default: 100)

            sam3-run options:
              --image <file.npy>          Optional normalized float32 [1,3,1008,1008] image
              --points <file.npy>         Optional float32 [N,2] model-input pixel (x,y) coordinates
              --labels <file.npy>         Required with --points: float32 [N], values 0 or 1
              --box <file.npy>            Optional float32 [4] model-input pixel x0,y0,x1,y1 box
              --output <directory>       Output directory (default: ./sam3-output)
              --caption <text>           Text prompt (default: "a dog")
              --device cpu               Execution device; CPU is currently the only supported runtime
              --seed <integer>           Random input seed (default: 42)
              --min-coverage <percent>   Fail below checkpoint loading coverage (default: 0)

            Exit codes:
              0 success, 1 runtime failure, 2 invalid usage, 3 validation/parity failure
            """);
    }
}

internal sealed class CliException(string message) : Exception(message);

internal sealed class CliOptions
{
    private readonly Dictionary<string, string> _values;
    private readonly HashSet<string> _used = new(StringComparer.OrdinalIgnoreCase);

    private CliOptions(Dictionary<string, string> values) => _values = values;

    public static CliOptions Parse(string[] args)
    {
        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (var index = 0; index < args.Length; index += 2)
        {
            var key = args[index];
            if (!key.StartsWith("--", StringComparison.Ordinal) || key.Length == 2)
                throw new CliException($"Expected an option beginning with '--', but found '{key}'.");
            if (index + 1 >= args.Length || args[index + 1].StartsWith("--", StringComparison.Ordinal))
                throw new CliException($"Option '{key}' requires a value.");
            if (!values.TryAdd(key[2..], args[index + 1]))
                throw new CliException($"Option '{key}' was specified more than once.");
        }
        return new CliOptions(values);
    }

    public string? Get(string name)
    {
        _used.Add(name);
        return _values.GetValueOrDefault(name);
    }

    public string RequirePath(string name)
    {
        var value = Get(name);
        if (string.IsNullOrWhiteSpace(value))
            throw new CliException($"Missing required option '--{name}'.");
        var path = Path.GetFullPath(value);
        if (!File.Exists(path))
            throw new CliException($"File supplied to '--{name}' does not exist: {path}");
        return path;
    }

    public string? GetOptionalPath(string name)
    {
        var value = Get(name);
        if (value is null) return null;
        var path = Path.GetFullPath(value);
        if (!File.Exists(path))
            throw new CliException($"File supplied to '--{name}' does not exist: {path}");
        return path;
    }

    public int GetInt(string name, int defaultValue)
    {
        var value = Get(name);
        if (value is null) return defaultValue;
        if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed))
            throw new CliException($"Option '--{name}' must be an integer.");
        return parsed;
    }

    public double GetDouble(string name, double defaultValue, double min, double max = double.MaxValue)
    {
        var value = Get(name);
        if (value is null) return defaultValue;
        if (!double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed) ||
            double.IsNaN(parsed) || parsed < min || parsed > max)
            throw new CliException($"Option '--{name}' must be a number between {min} and {max}.");
        return parsed;
    }

    public bool GetBool(string name, bool defaultValue)
    {
        var value = Get(name);
        if (value is null) return defaultValue;
        if (!bool.TryParse(value, out var parsed))
            throw new CliException($"Option '--{name}' must be true or false.");
        return parsed;
    }

    public void EnsureNoUnused()
    {
        var unused = _values.Keys.Where(key => !_used.Contains(key)).ToArray();
        if (unused.Length > 0)
            throw new CliException($"Unknown option(s): {string.Join(", ", unused.Select(key => $"--{key}"))}.");
    }
}