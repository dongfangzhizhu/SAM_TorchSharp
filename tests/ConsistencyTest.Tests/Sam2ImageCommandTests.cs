using SAMTorchSharp.Modeling.Sam2;

namespace ConsistencyTest.Tests;

public sealed class Sam2ImageCommandTests
{
    private static readonly NpyArray Image = new([2, 3, 3], Enumerable.Repeat(0.5f, 18).ToArray());

    [Theory]
    [InlineData("sam2-tiny", Sam2ModelVariant.Sam2Tiny)]
    [InlineData("sam2-small", Sam2ModelVariant.Sam2Small)]
    [InlineData("SAM2.1-TINY", Sam2ModelVariant.Sam21Tiny)]
    [InlineData("sam2.1-small", Sam2ModelVariant.Sam21Small)]
    public void ParsesSupportedVariant(string name, Sam2ModelVariant expected)
    {
        Assert.Equal(expected, Sam2ImageCommand.ParseVariant(name));
    }

    [Fact]
    public void AcceptsPointAndBoxPrompts()
    {
        var inputs = new Sam2ImageInputs(
            Image,
            new NpyArray([2, 2], [0, 0, 3, 2]),
            new NpyArray([2], [1, 0]),
            new NpyArray([4], [0, 0, 3, 2]));

        Sam2ImageCommand.ValidateInputs(inputs);
    }

    [Fact]
    public void AcceptsTwoByTwoBox()
    {
        var inputs = new Sam2ImageInputs(Image, null, null, new NpyArray([2, 2], [0, 0, 3, 2]));

        Sam2ImageCommand.ValidateInputs(inputs);
    }

    [Theory]
    [MemberData(nameof(InvalidInputs))]
    public void RejectsInvalidInputContract(object value, string expectedMessage)
    {
        var inputs = Assert.IsType<Sam2ImageInputs>(value);
        var error = Assert.Throws<CliException>(() => Sam2ImageCommand.ValidateInputs(inputs));

        Assert.Contains(expectedMessage, error.Message);
    }

    public static TheoryData<object, string> InvalidInputs => new()
    {
        { new Sam2ImageInputs(new NpyArray([3, 2, 3], Enumerable.Repeat(1.1f, 18).ToArray()), new NpyArray([1, 2], [0, 0]), new NpyArray([1], [1]), null), "[0,1]" },
        { new Sam2ImageInputs(Image, null, null, null), "At least one" },
        { new Sam2ImageInputs(Image, new NpyArray([1, 2], [0, 0]), null, null), "supplied together" },
        { new Sam2ImageInputs(Image, new NpyArray([2], [0, 0]), new NpyArray([1], [1]), null), "[N,2]" },
        { new Sam2ImageInputs(Image, new NpyArray([1, 2], [0, 0]), new NpyArray([2], [1, 0]), null), "matching" },
        { new Sam2ImageInputs(Image, new NpyArray([1, 2], [0, 0]), new NpyArray([1], [2]), null), "0 (negative) or 1" },
        { new Sam2ImageInputs(Image, new NpyArray([1, 2], [4, 0]), new NpyArray([1], [1]), null), "image bounds" },
        { new Sam2ImageInputs(Image, null, null, new NpyArray([1, 4], [0, 0, 3, 2])), "shape [4] or [2,2]" },
        { new Sam2ImageInputs(Image, null, null, new NpyArray([4], [2, 1, 1, 2])), "x0 <= x1" },
        { new Sam2ImageInputs(Image, null, null, new NpyArray([4], [0, 0, 3, 2]), new NpyArray([256, 256], new float[256 * 256])), "[1,256,256]" },
        { new Sam2ImageInputs(Image, null, null, new NpyArray([4], [0, 0, 3, 2]), new NpyArray([1, 256, 256], Enumerable.Repeat(float.NaN, 256 * 256).ToArray())), "must be finite" },
    };
}