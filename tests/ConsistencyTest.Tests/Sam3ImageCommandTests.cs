namespace ConsistencyTest.Tests;

using static TorchSharp.torch;

public sealed class Sam3ImageCommandTests
{
    [Fact]
    public void AcceptsFiniteNormalizedNchwImage()
    {
        var image = new NpyArray([1, 3, 1008, 1008], new float[3 * 1008 * 1008]);

        Sam3ImageCommand.ValidateImage(image);
    }

    [Fact]
    public void RejectsWrongImageShape()
    {
        var image = new NpyArray([1008, 1008, 3], new float[3 * 1008 * 1008]);

        var error = Assert.Throws<CliException>(() => Sam3ImageCommand.ValidateImage(image));

        Assert.Contains("[1,3,1008,1008]", error.Message);
    }

    [Fact]
    public void RejectsNonFiniteImage()
    {
        var values = new float[3 * 1008 * 1008];
        values[0] = float.NaN;
        var image = new NpyArray([1, 3, 1008, 1008], values);

        var error = Assert.Throws<CliException>(() => Sam3ImageCommand.ValidateImage(image));

        Assert.Contains("finite", error.Message);
    }

    [Fact]
    public void AcceptsFiniteSam3Outputs()
    {
        using var boxes = zeros(1, 2, 4);
        using var logits = zeros(1, 2, 1);
        using var masks = zeros(1, 2, 8, 8);
        using var semantic = zeros(1, 1, 8, 8);

        Sam3ImageCommand.ValidateOutputs(new Dictionary<string, TorchSharp.torch.Tensor>
        {
            ["pred_boxes"] = boxes,
            ["pred_logits"] = logits,
            ["pred_masks"] = masks,
            ["semantic_seg"] = semantic,
        });
    }

    [Fact]
    public void RejectsNonFiniteOrMissingSam3Outputs()
    {
        using var boxes = zeros(1, 2, 4);
        using var logits = zeros(1, 2, 1);
        using var masks = zeros(1, 2, 8, 8);
        using var nonFiniteMasks = full([1, 2, 8, 8], float.NaN);
        var outputs = new Dictionary<string, TorchSharp.torch.Tensor>
        {
            ["pred_boxes"] = boxes,
            ["pred_logits"] = logits,
            ["pred_masks"] = masks,
        };

        var missing = Assert.Throws<InvalidOperationException>(() => Sam3ImageCommand.ValidateOutputs(outputs));
        Assert.Contains("semantic_seg", missing.Message);

        using var semantic = zeros(1, 1, 8, 8);
        outputs["semantic_seg"] = semantic;
        outputs["pred_masks"] = nonFiniteMasks;
        var nonFinite = Assert.Throws<InvalidOperationException>(() => Sam3ImageCommand.ValidateOutputs(outputs));
        Assert.Contains("non-finite", nonFinite.Message);
    }
}