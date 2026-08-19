namespace ConsistencyTest.Tests;

public sealed class NumericComparisonTests
{
    [Fact]
    public void ExactValuesPassWithZeroTolerance()
    {
        var values = new NpyArray([3], [0f, 1f, -2f]);

        var result = NumericComparison.Compare(values, values, 0, 0);

        Assert.True(result.Passed);
        Assert.Equal(0, result.MismatchCount);
        Assert.Equal(0, result.MaxAbsoluteError);
    }

    [Fact]
    public void AbsoluteToleranceAcceptsSmallDifference()
    {
        var expected = new NpyArray([1], [0f]);
        var actual = new NpyArray([1], [0.00009f]);

        var result = NumericComparison.Compare(expected, actual, 0.0001, 0);

        Assert.True(result.Passed);
    }

    [Fact]
    public void RelativeToleranceAcceptsScaledDifference()
    {
        var expected = new NpyArray([1], [100f]);
        var actual = new NpyArray([1], [100.09f]);

        var result = NumericComparison.Compare(expected, actual, 0, 0.001);

        Assert.True(result.Passed);
    }

    [Fact]
    public void DifferenceOutsideToleranceFailsAndReportsMetrics()
    {
        var expected = new NpyArray([2], [1f, 2f]);
        var actual = new NpyArray([2], [1.5f, 1f]);

        var result = NumericComparison.Compare(expected, actual, 0.1, 0);

        Assert.False(result.Passed);
        Assert.Equal(2, result.MismatchCount);
        Assert.Equal(1, result.MaxAbsoluteError);
        Assert.Equal(0.75, result.MeanAbsoluteError);
        Assert.Equal(0.5, result.MaxRelativeError);
    }

    [Fact]
    public void ShapeMismatchThrows()
    {
        var expected = new NpyArray([2], [1f, 2f]);
        var actual = new NpyArray([1, 2], [1f, 2f]);

        var error = Assert.Throws<InvalidDataException>(() =>
            NumericComparison.Compare(expected, actual, 0, 0));

        Assert.Contains("Shape mismatch", error.Message);
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void MatchingNonFiniteValuesStillFailValidation(float value)
    {
        var expected = new NpyArray([1], [value]);
        var actual = new NpyArray([1], [value]);

        var result = NumericComparison.Compare(expected, actual, 0, 0);

        Assert.False(result.Passed);
        Assert.Equal(1, result.NonFiniteCount);
        Assert.Equal(0, result.MismatchCount);
    }

    [Fact]
    public void DifferentNonFiniteValuesCountAsMismatch()
    {
        var expected = new NpyArray([1], [float.PositiveInfinity]);
        var actual = new NpyArray([1], [float.NegativeInfinity]);

        var result = NumericComparison.Compare(expected, actual, 0, 0);

        Assert.False(result.Passed);
        Assert.Equal(1, result.NonFiniteCount);
        Assert.Equal(1, result.MismatchCount);
    }
}