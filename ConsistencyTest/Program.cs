using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
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
        var outputDirectory = Path.GetFullPath(options.Get("output") ?? Path.Combine(Environment.CurrentDirectory, "sam3-output"));
        var caption = options.Get("caption") ?? "a dog";
        var deviceName = (options.Get("device") ?? "cpu").ToLowerInvariant();
        var seed = options.GetInt("seed", 42);
        var minCoverage = options.GetDouble("min-coverage", 0, min: 0, max: 100);
        options.EnsureNoUnused();

        if (deviceName != "cpu")
            throw new CliException("Only --device cpu is supported by the configured libtorch-cpu-win-x64 runtime.");

        var extension = Path.GetExtension(checkpointPath).ToLowerInvariant();
        if (extension is not ".safetensors" and not ".bin")
            throw new CliException("--checkpoint must be an existing .safetensors or converted .bin file. Convert .pt explicitly before running the CLI.");

        Directory.CreateDirectory(outputDirectory);
        Console.WriteLine("Building SAM3 detector model...");
        using var model = new BuildSam3New().Build();

        Console.WriteLine($"Loading checkpoint: {checkpointPath}");
        int loaded;
        int skipped;
        int missing;
        if (extension == ".safetensors")
        {
            var result = new Sam3CheckpointLoaderNew().LoadModel(model, checkpointPath, CPU);
            (loaded, skipped, missing) = (result.Item1, result.Item2, result.Item3);
        }
        else
        {
            var result = new Sam3CheckpointLoaderBinary().LoadModel(model, checkpointPath, CPU);
            (loaded, skipped, missing) = (result.Item1, result.Item2, result.Item3);
        }

        var total = loaded + skipped + missing;
        var coverage = total == 0 ? 0 : loaded * 100.0 / total;
        Console.WriteLine($"Checkpoint: loaded={loaded}, skipped={skipped}, missing={missing}, coverage={coverage:F2}%");
        if (coverage < minCoverage)
            return Fail($"Checkpoint coverage {coverage:F2}% is below required {minCoverage:F2}%.", ValidationError);

        manual_seed(seed);
        using var input = randn(new long[] { 1, 3, 1008, 1008 }, dtype: ScalarType.Float32, device: CPU);
        var stopwatch = Stopwatch.StartNew();
        var outputs = model.Forward(input, new[] { caption }, geometricPrompt: null);
        stopwatch.Stop();

        try
        {
            foreach (var (name, tensor) in outputs)
            {
                var outputPath = Path.Combine(outputDirectory, $"{name}.npy");
                NpyFile.WriteFloat32(outputPath, tensor);
                Console.WriteLine($"{name}: [{string.Join(", ", tensor.shape)}] -> {outputPath}");
            }

            var summary = new
            {
                checkpoint = checkpointPath,
                device = deviceName,
                caption,
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
        }

        Console.WriteLine($"Inference completed in {stopwatch.ElapsedMilliseconds} ms.");
        return 0;
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
              ConsistencyTest sam3-run --checkpoint <model.safetensors|model.bin> [options]

            sam3-run options:
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

    public void EnsureNoUnused()
    {
        var unused = _values.Keys.Where(key => !_used.Contains(key)).ToArray();
        if (unused.Length > 0)
            throw new CliException($"Unknown option(s): {string.Join(", ", unused.Select(key => $"--{key}"))}.");
    }
}