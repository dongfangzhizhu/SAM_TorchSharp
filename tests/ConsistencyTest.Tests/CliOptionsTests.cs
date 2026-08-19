namespace ConsistencyTest.Tests;

public sealed class CliOptionsTests
{
    [Theory]
    [InlineData("true", true)]
    [InlineData("false", false)]
    [InlineData("TRUE", true)]
    public void ParsesBooleanOptions(string value, bool expected)
    {
        var options = CliOptions.Parse(["--strict", value]);

        Assert.Equal(expected, options.GetBool("strict", !expected));
        options.EnsureNoUnused();
    }

    [Fact]
    public void RejectsInvalidBooleanOption()
    {
        var options = CliOptions.Parse(["--strict", "yes"]);

        var error = Assert.Throws<CliException>(() => options.GetBool("strict", true));

        Assert.Contains("must be true or false", error.Message);
    }
}