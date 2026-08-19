namespace ConsistencyTest.Tests;

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
}