using SAMTorchSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace ConsistencyTest.Tests;

public sealed class Sam2ImagePredictorTests
{
    [Fact]
    public void RequiresAnImageBeforePrediction()
    {
        using var model = BuildSam2.BuildSam2HieraTiny(imageSize: 64);
        using var predictor = new SAM2ImagePredictor(model);
        using var point = tensor(new float[] { 8, 8 }).reshape(1, 2);
        using var label = tensor(new long[] { 1 });

        var error = Assert.Throws<InvalidOperationException>(() => predictor.Predict(point, label));

        Assert.Contains("SetImage", error.Message);
    }

    [Fact]
    public void PredictsDifferentSizedImagesFromOneBackboneBatch()
    {
        using var model = BuildSam2.BuildSam2HieraTiny(imageSize: 64);
        using var predictor = new SAM2ImagePredictor(model);
        using var firstImage = rand(32, 48, 3);
        using var secondImage = rand(40, 24, 3);
        using var firstBox = tensor(new float[] { 4, 3, 40, 28 });
        using var secondBox = tensor(new float[] { 2, 5, 20, 35 });

        predictor.SetImageBatch([firstImage, secondImage]);
        var results = predictor.PredictBatch(boxBatch: [firstBox, secondBox], multimaskOutput: false);

        Assert.Equal(2, results.Count);
        Assert.Equal([1L, 1L, 32L, 48L], results[0].Masks.shape);
        Assert.Equal([1L, 1L, 40L, 24L], results[1].Masks.shape);
        Assert.Equal([1L, 1L], results[0].IouPredictions.shape);
        Assert.Equal([1L, 1L, 16L, 16L], results[0].LowResMasks.shape);

        DisposeResults(results);
    }

    [Fact]
    public void EnforcesSingleAndBatchPredictionModes()
    {
        using var model = BuildSam2.BuildSam2HieraTiny(imageSize: 64);
        using var predictor = new SAM2ImagePredictor(model);
        using var image = rand(32, 32, 3);
        using var point = tensor(new float[] { 8, 8 }).reshape(1, 2);
        using var label = tensor(new long[] { 1 });

        predictor.SetImage(image);
        Assert.Throws<InvalidOperationException>(() => predictor.PredictBatch([point], [label]));

        predictor.SetImageBatch([image]);
        Assert.Throws<InvalidOperationException>(() => predictor.Predict(point, label));
        var error = Assert.Throws<ArgumentException>(() => predictor.PredictBatch(
            pointCoordsBatch: [point, point], pointLabelsBatch: [label, label]));
        Assert.Contains("one entry per image", error.Message);
    }

    [Fact]
    public void PredictsMultipleBoxesOnOneImageLikeOfficialNotebook()
    {
        using var model = BuildSam2.BuildSam2HieraTiny(imageSize: 64);
        using var predictor = new SAM2ImagePredictor(model);
        using var image = rand(32, 48, 3);
        using var boxes = tensor(new float[]
        {
            2, 3, 20, 28,
            22, 4, 45, 30,
        }).reshape(2, 4);

        predictor.SetImage(image);
        var result = predictor.Predict(box: boxes, multimaskOutput: false);

        Assert.Equal([2L, 1L, 32L, 48L], result.Masks.shape);
        Assert.Equal([2L, 1L], result.IouPredictions.shape);
        Assert.Equal([2L, 1L, 16L, 16L], result.LowResMasks.shape);
        result.Masks.Dispose();
        result.IouPredictions.Dispose();
        result.LowResMasks.Dispose();
    }

    private static void DisposeResults(
        IReadOnlyList<(Tensor Masks, Tensor IouPredictions, Tensor LowResMasks)> results)
    {
        foreach (var result in results)
        {
            result.Masks.Dispose();
            result.IouPredictions.Dispose();
            result.LowResMasks.Dispose();
        }
    }
}