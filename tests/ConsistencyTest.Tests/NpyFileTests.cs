using System.Text;

namespace ConsistencyTest.Tests;

public sealed class NpyFileTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), $"sam-npy-tests-{Guid.NewGuid():N}");

    public NpyFileTests() => Directory.CreateDirectory(_directory);

    [Fact]
    public void RoundTripScalar()
    {
        AssertRoundTrip(new NpyArray([], [4.25f]));
    }

    [Fact]
    public void RoundTripVector()
    {
        AssertRoundTrip(new NpyArray([3], [1f, -2.5f, 0f]));
    }

    [Fact]
    public void RoundTripMultidimensionalArray()
    {
        AssertRoundTrip(new NpyArray([2, 2], [0f, 1f, -2.5f, 4.25f]));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void ReadsSupportedNpyVersions(byte majorVersion)
    {
        var path = WriteNpy(majorVersion, "<f4", false, "(2,)", [1.5f, -3f]);

        var result = NpyFile.ReadFloat32(path);

        Assert.Equal([2], result.Shape);
        Assert.Equal([1.5f, -3f], result.Values);
    }

    [Fact]
    public void RejectsUnsupportedVersion()
    {
        var path = WriteNpy(4, "<f4", false, "(1,)", [1f]);

        var error = Assert.Throws<InvalidDataException>(() => NpyFile.ReadFloat32(path));

        Assert.Contains("Unsupported NPY version 4", error.Message);
    }

    [Theory]
    [InlineData(">f4")]
    [InlineData("<f8")]
    [InlineData("<i4")]
    public void RejectsUnsupportedDtype(string dtype)
    {
        var path = WriteNpy(1, dtype, false, "(1,)", [1f]);

        var error = Assert.Throws<InvalidDataException>(() => NpyFile.ReadFloat32(path));

        Assert.Contains("Only little-endian float32", error.Message);
    }

    [Fact]
    public void RejectsFortranOrder()
    {
        var path = WriteNpy(1, "<f4", true, "(1,)", [1f]);

        var error = Assert.Throws<InvalidDataException>(() => NpyFile.ReadFloat32(path));

        Assert.Contains("Fortran-order", error.Message);
    }

    [Fact]
    public void RejectsPayloadShorterThanShape()
    {
        var path = WriteNpy(1, "<f4", false, "(2,)", [1f]);

        var error = Assert.Throws<InvalidDataException>(() => NpyFile.ReadFloat32(path));

        Assert.Contains("payload length", error.Message);
    }

    [Fact]
    public void RejectsPayloadLongerThanShape()
    {
        var path = WriteNpy(1, "<f4", false, "(1,)", [1f, 2f]);

        var error = Assert.Throws<InvalidDataException>(() => NpyFile.ReadFloat32(path));

        Assert.Contains("payload length", error.Message);
    }

    [Fact]
    public void WriterRejectsShapeValueCountMismatch()
    {
        var path = Path.Combine(_directory, "invalid.npy");

        var error = Assert.Throws<ArgumentException>(() =>
            NpyFile.WriteFloat32(path, new NpyArray([2, 2], [1f])));

        Assert.Equal("array", error.ParamName);
    }

    [Fact]
    public void RejectsInvalidMagic()
    {
        var path = Path.Combine(_directory, "invalid-magic.npy");
        File.WriteAllBytes(path, [0, 1, 2, 3, 4, 5, 1, 0]);

        var error = Assert.Throws<InvalidDataException>(() => NpyFile.ReadFloat32(path));

        Assert.Contains("not a NumPy NPY file", error.Message);
    }

    private void AssertRoundTrip(NpyArray source)
    {
        var path = Path.Combine(_directory, $"roundtrip-{Guid.NewGuid():N}.npy");

        NpyFile.WriteFloat32(path, source);
        var result = NpyFile.ReadFloat32(path);

        Assert.Equal(source.Shape, result.Shape);
        Assert.Equal(source.Values, result.Values);
    }

    private string WriteNpy(byte majorVersion, string dtype, bool fortranOrder, string shape, float[] values)
    {
        var path = Path.Combine(_directory, $"manual-{Guid.NewGuid():N}.npy");
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: false);
        writer.Write(new byte[] { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y' });
        writer.Write(majorVersion);
        writer.Write((byte)0);

        if (majorVersion is 1 or 2 or 3)
        {
            var header = Encoding.ASCII.GetBytes(
                $"{{'descr': '{dtype}', 'fortran_order': {(fortranOrder ? "True" : "False")}, 'shape': {shape}, }}\n");
            if (majorVersion == 1)
                writer.Write(checked((ushort)header.Length));
            else
                writer.Write((uint)header.Length);
            writer.Write(header);
        }

        foreach (var value in values)
            writer.Write(value);
        return path;
    }

    public void Dispose()
    {
        Directory.Delete(_directory, recursive: true);
    }
}