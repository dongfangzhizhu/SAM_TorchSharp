using System.Text;
using System.Text.RegularExpressions;
using TorchSharp;
using static TorchSharp.torch;

namespace ConsistencyTest;

internal sealed record NpyArray(int[] Shape, float[] Values);

internal static partial class NpyFile
{
    private static readonly byte[] Magic = [0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y'];

    public static NpyArray ReadFloat32(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: false);
        if (!reader.ReadBytes(Magic.Length).SequenceEqual(Magic))
            throw new InvalidDataException($"'{path}' is not a NumPy NPY file.");

        var major = reader.ReadByte();
        _ = reader.ReadByte();
        var headerLength = major switch
        {
            1 => reader.ReadUInt16(),
            2 or 3 => checked((int)reader.ReadUInt32()),
            _ => throw new InvalidDataException($"Unsupported NPY version {major} in '{path}'."),
        };
        var header = Encoding.ASCII.GetString(reader.ReadBytes(headerLength));
        if (!header.Contains("'descr': '<f4'", StringComparison.Ordinal) &&
            !header.Contains("\"descr\": \"<f4\"", StringComparison.Ordinal))
            throw new InvalidDataException($"Only little-endian float32 NPY arrays are supported: '{path}'.");
        if (header.Contains("'fortran_order': True", StringComparison.Ordinal) ||
            header.Contains("\"fortran_order\": true", StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException($"Fortran-order NPY arrays are not supported: '{path}'.");

        var shapeMatch = ShapeRegex().Match(header);
        if (!shapeMatch.Success)
            throw new InvalidDataException($"Could not parse NPY shape in '{path}'.");
        var shape = shapeMatch.Groups[1].Value
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(value => int.Parse(value, System.Globalization.CultureInfo.InvariantCulture))
            .ToArray();
        var count = shape.Length == 0 ? 1 : shape.Aggregate(1L, (product, dimension) => checked(product * dimension));
        if (count > int.MaxValue)
            throw new InvalidDataException($"NPY array is too large: '{path}'.");

        var expectedBytes = checked((int)count * sizeof(float));
        var bytes = reader.ReadBytes(expectedBytes);
        if (bytes.Length != expectedBytes || stream.Position != stream.Length)
            throw new InvalidDataException($"NPY payload length does not match shape in '{path}'.");
        var values = new float[(int)count];
        Buffer.BlockCopy(bytes, 0, values, 0, expectedBytes);
        return new NpyArray(shape, values);
    }

    public static void WriteFloat32(string path, Tensor tensor)
    {
        using var contiguous = tensor.detach().to_type(ScalarType.Float32).cpu().contiguous();
        var shape = contiguous.shape.Select(value => checked((int)value)).ToArray();
        var values = contiguous.flatten().data<float>().ToArray();
        WriteFloat32(path, new NpyArray(shape, values));
    }

    public static void WriteFloat32(string path, NpyArray array)
    {
        var expectedCount = array.Shape.Length == 0 ? 1 : array.Shape.Aggregate(1L, (product, dimension) => checked(product * dimension));
        if (expectedCount != array.Values.LongLength)
            throw new ArgumentException("Shape does not match the number of values.", nameof(array));

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: false);
        writer.Write(Magic);
        writer.Write((byte)1);
        writer.Write((byte)0);

        var shape = array.Shape.Length switch
        {
            0 => "()",
            1 => $"({array.Shape[0]},)",
            _ => $"({string.Join(", ", array.Shape)},)",
        };
        var headerText = $"{{'descr': '<f4', 'fortran_order': False, 'shape': {shape}, }}";
        var preambleLength = Magic.Length + 2 + sizeof(ushort);
        var padding = 16 - ((preambleLength + headerText.Length + 1) % 16);
        if (padding == 16) padding = 0;
        var header = Encoding.ASCII.GetBytes(headerText + new string(' ', padding) + "\n");
        writer.Write(checked((ushort)header.Length));
        writer.Write(header);
        foreach (var value in array.Values)
            writer.Write(value);
    }

    [GeneratedRegex("['\"]shape['\"]\\s*:\\s*\\(([^)]*)\\)", RegexOptions.CultureInvariant)]
    private static partial Regex ShapeRegex();
}