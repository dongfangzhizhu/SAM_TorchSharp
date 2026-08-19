namespace ConsistencyTest;

internal sealed record ComparisonResult(
    int[] Shape,
    int Count,
    double MaxAbsoluteError,
    double MeanAbsoluteError,
    double MaxRelativeError,
    int MismatchCount,
    int NonFiniteCount)
{
    public bool Passed => MismatchCount == 0 && NonFiniteCount == 0;
}

internal static class NumericComparison
{
    public static ComparisonResult Compare(NpyArray expected, NpyArray actual, double absoluteTolerance, double relativeTolerance)
    {
        if (!expected.Shape.SequenceEqual(actual.Shape))
            throw new InvalidDataException(
                $"Shape mismatch: expected [{string.Join(", ", expected.Shape)}], actual [{string.Join(", ", actual.Shape)}].");

        double maxAbsolute = 0;
        double sumAbsolute = 0;
        double maxRelative = 0;
        var mismatches = 0;
        var nonFinite = 0;

        for (var index = 0; index < expected.Values.Length; index++)
        {
            var expectedValue = expected.Values[index];
            var actualValue = actual.Values[index];
            if (!float.IsFinite(expectedValue) || !float.IsFinite(actualValue))
            {
                if (BitConverter.SingleToInt32Bits(expectedValue) != BitConverter.SingleToInt32Bits(actualValue))
                    mismatches++;
                nonFinite++;
                continue;
            }

            var absolute = Math.Abs((double)actualValue - expectedValue);
            var relative = absolute / Math.Max(Math.Abs((double)expectedValue), 1e-12);
            maxAbsolute = Math.Max(maxAbsolute, absolute);
            maxRelative = Math.Max(maxRelative, relative);
            sumAbsolute += absolute;
            if (absolute > absoluteTolerance + relativeTolerance * Math.Abs((double)expectedValue))
                mismatches++;
        }

        return new ComparisonResult(
            expected.Shape,
            expected.Values.Length,
            maxAbsolute,
            expected.Values.Length == 0 ? 0 : sumAbsolute / expected.Values.Length,
            maxRelative,
            mismatches,
            nonFinite);
    }
}