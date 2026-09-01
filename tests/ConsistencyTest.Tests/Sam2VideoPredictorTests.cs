using SAMTorchSharp;
using static TorchSharp.torch;

namespace ConsistencyTest.Tests;

public sealed class Sam2VideoPredictorTests
{
    [Fact]
    public void PropagatesPointPromptAcrossPreprocessedFrames()
    {
        using var model = BuildSam2.BuildSam2HieraTiny(imageSize: 64);
        using var predictor = new SAM2VideoPredictor(model);
        using var frames = rand(2, 3, 64, 64);
        using var points = tensor(new float[] { 24, 16 }).reshape(1, 2);
        using var labels = tensor(new long[] { 1 });
        var state = predictor.InitState(frames, originalHeight: 32, originalWidth: 48);

        var initial = predictor.AddNewPointsOrBox(state, 0, objId: 7, points, labels);
        var propagated = predictor.PropagateInVideo(state).ToList();

        Assert.Equal(0, initial.FrameIdx);
        Assert.Equal([7L], initial.ObjIds);
        Assert.Equal([1L, 1L, 32L, 48L], initial.VideoResMasks.shape);
        Assert.Equal([0, 1], propagated.Select(result => result.FrameIdx).ToArray());
        Assert.All(propagated, result =>
        {
            Assert.Equal([7L], result.ObjIds);
            Assert.Equal([1L, 1L, 32L, 48L], result.VideoResMasks.shape);
            result.VideoResMasks.Dispose();
        });
        initial.VideoResMasks.Dispose();
    }

    [Fact]
    public void ResetStateRemovesEveryPerObjectContainer()
    {
        using var model = BuildSam2.BuildSam2HieraTiny(imageSize: 64);
        using var predictor = new SAM2VideoPredictor(model);
        using var frames = rand(1, 3, 64, 64);
        using var firstPoint = tensor(new float[] { 12, 12 }).reshape(1, 2);
        using var secondPoint = tensor(new float[] { 20, 20 }).reshape(1, 2);
        using var labels = tensor(new long[] { 1 });
        var state = predictor.InitState(frames, originalHeight: 32, originalWidth: 32);

        var first = predictor.AddNewPointsOrBox(state, 0, objId: 10, firstPoint, labels);
        var second = predictor.AddNewPointsOrBox(state, 0, objId: 20, secondPoint, labels);
        first.VideoResMasks.Dispose();
        second.VideoResMasks.Dispose();
        predictor.ResetState(state);

        Assert.Empty((Dictionary<long, Dictionary<int, PointInputPerFrame>>)state["point_inputs_per_obj"]);
        Assert.Empty((Dictionary<long, Dictionary<int, TorchSharp.torch.Tensor>>)state["mask_inputs_per_obj"]);
        Assert.Empty((Dictionary<long, ObjectOutputDict>)state["output_dict_per_obj"]);
        Assert.Empty((Dictionary<long, ObjectOutputDict>)state["temp_output_dict_per_obj"]);
        Assert.Empty((Dictionary<long, Dictionary<int, FrameTrackedInfo>>)state["frames_tracked_per_obj"]);
    }
}