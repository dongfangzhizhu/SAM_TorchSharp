using SAMTorchSharp;
using static TorchSharp.torch;

namespace ConsistencyTest.Tests;

public sealed class Sam2VideoPredictorTests
{
    [Fact]
    public void SupportsBoxOnlyAndAdditionalCorrectionPoints()
    {
        using var model = BuildSam2.BuildSam2HieraTiny(imageSize: 64);
        using var predictor = new SAM2VideoPredictor(model);
        using var frames = rand(1, 3, 64, 64);
        using var box = tensor(new float[] { 12, 8, 36, 24 });
        using var correction = tensor(new float[] { 24, 16 }).reshape(1, 2);
        using var correctionLabel = tensor(new long[] { 0 });
        var state = predictor.InitState(frames, originalHeight: 32, originalWidth: 48);

        var boxResult = predictor.AddNewPointsOrBox(state, 0, objId: 7, box: box);
        var correctionResult = predictor.AddNewPointsOrBox(
            state, 0, objId: 7, correction, correctionLabel, clearOldPoints: false);

        var prompts = ((Dictionary<long, Dictionary<int, PointInputPerFrame>>)state["point_inputs_per_obj"])[0][0];
        Assert.Equal([1L, 3L, 2L], prompts.PointCoords.shape);
        Assert.Equal([2, 3, 0], prompts.PointLabels.flatten().data<int>().ToArray());
        Assert.Equal(
            [16f, 16f, 48f, 48f, 32f, 32f],
            prompts.PointCoords.flatten().data<float>().ToArray());
        boxResult.VideoResMasks.Dispose();
        correctionResult.VideoResMasks.Dispose();
    }

    [Fact]
    public void MaskPromptReplacesPointsAndPropagatesBackward()
    {
        using var model = BuildSam2.BuildSam2HieraTiny(imageSize: 64);
        using var predictor = new SAM2VideoPredictor(model);
        using var frames = rand(3, 3, 64, 64);
        using var points = tensor(new float[] { 24, 16 }).reshape(1, 2);
        using var labels = tensor(new long[] { 1 });
        using var mask = zeros(32, 48);
        mask[TensorIndex.Slice(8, 24), TensorIndex.Slice(12, 36)] = 1;
        var state = predictor.InitState(frames, originalHeight: 32, originalWidth: 48);

        var pointResult = predictor.AddNewPointsOrBox(state, 2, objId: 9, points, labels);
        var maskResult = predictor.AddNewMask(state, 2, objId: 9, mask);
        var propagated = predictor.PropagateInVideo(state, startFrameIdx: 2, reverse: true).ToList();

        Assert.Empty(((Dictionary<long, Dictionary<int, PointInputPerFrame>>)state["point_inputs_per_obj"])[0]);
        var storedMask = ((Dictionary<long, Dictionary<int, Tensor>>)state["mask_inputs_per_obj"])[0][2];
        Assert.Equal([1L, 1L, 64L, 64L], storedMask.shape);
        Assert.Equal([2, 1, 0], propagated.Select(result => result.FrameIdx).ToArray());
        Assert.All(propagated, result =>
        {
            Assert.Equal([1L, 1L, 32L, 48L], result.VideoResMasks.shape);
            result.VideoResMasks.Dispose();
        });
        var tracked = ((Dictionary<long, Dictionary<int, FrameTrackedInfo>>)state["frames_tracked_per_obj"])[0];
        Assert.All(tracked.Values, info => Assert.True(info.Reverse));
        pointResult.VideoResMasks.Dispose();
        maskResult.VideoResMasks.Dispose();
    }

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

    [Fact]
    public void ClearsPromptsRemovesObjectsAndAllowsObjectsAddedDuringTracking()
    {
        using var model = BuildSam2.BuildSam2HieraTiny(imageSize: 64);
        using var predictor = new SAM2VideoPredictor(model);
        using var frames = rand(2, 3, 64, 64);
        using var firstPoint = tensor(new float[] { 12, 12 }).reshape(1, 2);
        using var secondPoint = tensor(new float[] { 20, 20 }).reshape(1, 2);
        using var thirdPoint = tensor(new float[] { 24, 24 }).reshape(1, 2);
        using var labels = tensor(new long[] { 1 });
        var state = predictor.InitState(frames, originalHeight: 32, originalWidth: 32);

        var first = predictor.AddNewPointsOrBox(state, 0, objId: 10, firstPoint, labels);
        var second = predictor.AddNewPointsOrBox(state, 0, objId: 20, secondPoint, labels);
        var initialPropagation = predictor.PropagateInVideo(state).ToList();
        var third = predictor.AddNewPointsOrBox(state, 1, objId: 30, thirdPoint, labels);
        var cleared = predictor.ClearAllPromptsInFrame(state, 1, objId: 30);
        var readded = predictor.AddNewPointsOrBox(state, 1, objId: 30, thirdPoint, labels);
        var removed = predictor.RemoveObject(state, objId: 20);
        var finalPropagation = predictor.PropagateInVideo(state).ToList();

        Assert.NotNull(cleared);
        Assert.Equal([10L, 30L], removed.ObjIds);
        Assert.Single(removed.UpdatedFrames);
        Assert.Equal(0, removed.UpdatedFrames[0].FrameIdx);
        Assert.Equal([2L, 1L, 32L, 32L], removed.UpdatedFrames[0].VideoResMasks.shape);
        var mappings = (System.Collections.Concurrent.ConcurrentDictionary<long, long>)state["obj_id_to_idx"];
        Assert.Equal(0, mappings[10]);
        Assert.Equal(1, mappings[30]);
        Assert.Equal([0L, 1L], ((Dictionary<long, ObjectOutputDict>)state["output_dict_per_obj"]).Keys.Order().ToArray());
        Assert.Contains(1, ((Dictionary<long, Dictionary<int, PointInputPerFrame>>)state["point_inputs_per_obj"])[1].Keys);
        Assert.All(finalPropagation, result => Assert.Equal([10L, 30L], result.ObjIds));

        first.VideoResMasks.Dispose();
        second.VideoResMasks.Dispose();
        third.VideoResMasks.Dispose();
        cleared.Value.VideoResMasks.Dispose();
        readded.VideoResMasks.Dispose();
        removed.UpdatedFrames.ForEach(result => result.VideoResMasks.Dispose());
        initialPropagation.ForEach(result => result.VideoResMasks.Dispose());
        finalPropagation.ForEach(result => result.VideoResMasks.Dispose());
    }
}