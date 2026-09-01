using System.Text.Json;
using SAMTorchSharp;
using SAMTorchSharp.Modeling.Sam2;
using TorchSharp;
using TorchSharp.PyBridge;
using static TorchSharp.torch;

namespace ConsistencyTest;

internal static class Sam2VideoCommand
{
    public static int Run(
        Sam2ModelVariant variant,
        string variantName,
        string checkpointPath,
        string vectorDirectory,
        string reportPath,
        double absoluteTolerance,
        double relativeTolerance)
    {
        var inputPath = RequireVectorFile(vectorDirectory, "input.safetensors");
        var expectedPath = RequireVectorFile(vectorDirectory, "output_py.safetensors");
        var intermediatePath = RequireVectorFile(vectorDirectory, "intermediate_py.safetensors");
        var inputs = Safetensors.LoadStateDict(inputPath);
        var expected = Safetensors.LoadStateDict(expectedPath);
        var expectedIntermediate = Safetensors.LoadStateDict(intermediatePath);
        ValidateVectors(inputs, expected);

        using var model = Sam2ModelBuilder.Build(variant);
        using var predictor = new SAM2VideoPredictor(model);
        var checkpoint = Sam2CheckpointLoader.Load(model, checkpointPath, strict: true);
        using var frames = inputs["preprocessed_frames"].to(ScalarType.Float32);
        using var points = inputs["point_coords"].to(ScalarType.Float32);
        using var labels = inputs["point_labels"].to(ScalarType.Int32);
        using var originalSize = inputs["orig_hw"].to(ScalarType.Float32);
        var height = checked((long)originalSize[0, 0].item<float>());
        var width = checked((long)originalSize[0, 1].item<float>());
        var state = predictor.InitState(frames, height, width);

        var initial = predictor.AddNewPointsOrBox(state, 0, objId: 1, points, labels);
        var actual = predictor.PropagateInVideo(state).ToDictionary(item => item.FrameIdx, item => item.VideoResMasks);
        initial.VideoResMasks.Dispose();

        var comparisons = new Dictionary<string, ComparisonResult>();
        var intermediateComparisons = new Dictionary<string, ComparisonResult>();
        try
        {
            foreach (var pair in expected.OrderBy(pair => ParseFrameIndex(pair.Key)))
            {
                var frameIndex = ParseFrameIndex(pair.Key);
                if (!actual.TryGetValue(frameIndex, out var actualTensor))
                    throw new InvalidDataException($"The .NET predictor did not produce frame {frameIndex}.");
                comparisons[pair.Key] = NumericComparison.Compare(
                    ToNpyArray(pair.Value), ToNpyArray(actualTensor), absoluteTolerance, relativeTolerance);
            }

            var frame0 = ((Dictionary<long, ObjectOutputDict>)state["output_dict_per_obj"])[0].CondFrameOutputs[0];
            var actualIntermediate = new Dictionary<string, Tensor>
            {
                ["frame0_maskmem_features"] = frame0.MaskmemFeatures!,
                ["frame0_maskmem_pos_enc"] = frame0.MaskmemPosEnc![^1],
                ["frame0_obj_ptr"] = frame0.ObjPtr,
                ["frame0_object_score_logits"] = frame0.ObjectScoreLogits,
            };
            foreach (var pair in expectedIntermediate)
                intermediateComparisons[pair.Key] = NumericComparison.Compare(
                    ToNpyArray(pair.Value), ToNpyArray(actualIntermediate[pair.Key]), absoluteTolerance, relativeTolerance);
        }
        finally
        {
            foreach (var tensor in actual.Values) tensor.Dispose();
        }

        var passed = comparisons.Values.Concat(intermediateComparisons.Values).All(result => result.Passed);
        var report = new
        {
            variant = variantName,
            checkpoint = checkpointPath,
            checkpointCoverage = checkpoint.Coverage,
            absoluteTolerance,
            relativeTolerance,
            passed,
            intermediates = intermediateComparisons,
            frames = comparisons,
        };
        Directory.CreateDirectory(Path.GetDirectoryName(reportPath)!);
        File.WriteAllText(reportPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
        foreach (var pair in comparisons)
            Console.WriteLine($"{pair.Key}: max_abs={pair.Value.MaxAbsoluteError:E6}, mean_abs={pair.Value.MeanAbsoluteError:E6}, mismatches={pair.Value.MismatchCount}");
        foreach (var pair in intermediateComparisons)
            Console.WriteLine($"{pair.Key}: max_abs={pair.Value.MaxAbsoluteError:E6}, mean_abs={pair.Value.MeanAbsoluteError:E6}, mismatches={pair.Value.MismatchCount}");
        return passed ? 0 : 3;
    }

    internal static void ValidateVectors(
        IReadOnlyDictionary<string, Tensor> inputs,
        IReadOnlyDictionary<string, Tensor> expected)
    {
        foreach (var key in new[] { "preprocessed_frames", "point_coords", "point_labels", "orig_hw" })
            if (!inputs.ContainsKey(key)) throw new InvalidDataException($"Input vectors are missing '{key}'.");
        if (inputs["preprocessed_frames"].dim() != 4 || inputs["preprocessed_frames"].size(1) != 3)
            throw new InvalidDataException("preprocessed_frames must have shape [T,3,H,W].");
        if (!inputs["point_coords"].shape.SequenceEqual(new long[] { 1, 2, 2 }) ||
            !inputs["point_labels"].shape.SequenceEqual(new long[] { 1, 2 }))
            throw new InvalidDataException("Point vectors must have shapes [1,2,2] and [1,2].");
        if (!inputs["orig_hw"].shape.SequenceEqual(new long[] { 1, 2 }))
            throw new InvalidDataException("orig_hw must have shape [1,2].");
        if (expected.Count == 0 || expected.Keys.Any(key => !key.StartsWith("frame", StringComparison.Ordinal) || !key.EndsWith("_masks", StringComparison.Ordinal)))
            throw new InvalidDataException("Expected vectors must contain frameN_masks tensors.");
    }

    private static string RequireVectorFile(string directory, string name)
    {
        var path = Path.Combine(directory, name);
        return File.Exists(path) ? path : throw new FileNotFoundException($"Missing video parity vector '{name}'.", path);
    }

    private static int ParseFrameIndex(string key) =>
        int.Parse(key.AsSpan("frame".Length, key.Length - "frame".Length - "_masks".Length));

    private static NpyArray ToNpyArray(Tensor tensor)
    {
        using var contiguous = tensor.to(CPU).to(ScalarType.Float32).contiguous();
        return new NpyArray(contiguous.shape.Select(checked(value => (int)value)).ToArray(), contiguous.flatten().data<float>().ToArray());
    }
}