using SAMTorchSharp.Modeling.Sam3;
using static TorchSharp.torch;

namespace ConsistencyTest.Tests;

public sealed class Sam3SegmentationHeadTests
{
    [Fact]
    public void SinePositionEncodingInterleavesOfficialSinCosFrequencies()
    {
        using var encoding = new Sam3PositionEmbeddingSine(num_pos_feats: 8);
        using var x = zeros(1);
        using var y = zeros(1);
        var (encodedX, encodedY) = encoding.EncodeXY(x, y);
        using (encodedX)
        using (encodedY)
        using (var expected = tensor(new float[,] { { 0f, 1f, 0f, 1f } }))
        {
            Assert.Equal([1L, 4L], encodedX.shape);
            Assert.True(allclose(encodedX, expected));
            Assert.True(allclose(encodedY, expected));
        }
    }

    [Fact]
    public void SinePositionEncodingAppendsBoxHeightThenWidth()
    {
        using var encoding = new Sam3PositionEmbeddingSine(num_pos_feats: 8);
        using var x = zeros(1);
        using var y = zeros(1);
        using var width = tensor(new[] { 0.25f });
        using var height = tensor(new[] { 0.75f });
        using var encoded = encoding.EncodeBoxes(x, y, width, height);

        Assert.Equal([1L, 10L], encoded.shape);
        Assert.Equal(0.75f, encoded[0, 8].item<float>(), precision: 6);
        Assert.Equal(0.25f, encoded[0, 9].item<float>(), precision: 6);
    }

    [Fact]
    public void VisionAttentionDelegatesScalingToSdpa()
    {
        var source = File.ReadAllText(FindRepositoryFile("SAMTorchSharp", "Modeling", "Sam3", "VitBackbone.cs"));

        Assert.Contains("scaled_dot_product_attention(q, k, v)", source);
        Assert.DoesNotContain("scaled_dot_product_attention(q_scaled", source);
    }

    [Fact]
    public void GeometryAttentionDelegatesScalingToSdpa()
    {
        var source = File.ReadAllText(FindRepositoryFile("SAMTorchSharp", "Modeling", "Sam3", "GeometryEncoderNew.cs"));

        Assert.Equal(2, source.Split("scaled_dot_product_attention(q, k, v)").Length - 1);
        Assert.DoesNotContain("scaled_dot_product_attention(q_scaled", source);
    }

    [Fact]
    public void GeometryLayerUsesSequenceFirstLayoutForBatchedCrossAttention()
    {
        manual_seed(37);
        using var layer = new Sam3GeometryEncoderLayer(d_model: 8, num_heads: 2);
        using var query = randn(3, 2, 8);
        using var memory = randn(5, 2, 8);
        using var memoryPos = randn(5, 2, 8);

        var (output, _) = layer.forward(query, memory, memoryPos);
        using (output)
        {
            Assert.Equal([3L, 2L, 8L], output.shape);
            Assert.True(isfinite(output).all().item<bool>());
        }
    }

    [Fact]
    public void GeometryLayerRejectsMismatchedSequenceFirstBatchSizes()
    {
        using var layer = new Sam3GeometryEncoderLayer(d_model: 8, num_heads: 2);
        using var query = zeros(3, 2, 8);
        using var memory = zeros(5, 1, 8);

        var exception = Assert.Throws<ArgumentException>(() => layer.forward(query, memory));
        Assert.Contains("batch sizes must match", exception.Message);
    }

    [Fact]
    public void GeometryEncoderRunsPostProjectionAndTransformerStack()
    {
        manual_seed(41);
        using var encoder = new Sam3GeometryEncoderNew(d_model: 8, num_geo_layers: 1);
        using var feature = randn(2, 8, 2, 3);
        var (tokens, mask) = encoder.forward(new Sam3Prompt(), [feature], [[2L, 3L]]);
        using (tokens)
        using (mask)
        {
            Assert.Equal([1L, 2L, 8L], tokens.shape);
            Assert.Equal([2L, 1L], mask.shape);
            Assert.True(isfinite(tokens).all().item<bool>());
            Assert.False(mask.any().item<bool>());
        }
    }

    [Fact]
    public void GeometryEncoderUsesSequenceFirstPointAndBoxPrompts()
    {
        manual_seed(43);
        using var encoder = new Sam3GeometryEncoderNew(d_model: 8, num_geo_layers: 0);
        using var points = rand(2, 2, 2);
        using var boxes = rand(1, 2, 4);
        using var pointMask = zeros(2, 2, dtype: ScalarType.Bool);
        using var boxMask = zeros(2, 1, dtype: ScalarType.Bool);
        var prompt = new Sam3Prompt(
            box_embeddings: boxes,
            box_mask: boxMask,
            point_embeddings: points,
            point_mask: pointMask);
        using var feature = randn(2, 8, 2, 2);

        var (tokens, mask) = encoder.forward(prompt, [feature], [[2L, 2L]]);
        using (tokens)
        using (mask)
        {
            Assert.Equal([3L, 2L, 8L], tokens.shape);
            Assert.Equal([2L, 3L], mask.shape);
            Assert.True(isfinite(tokens).all().item<bool>());
        }
    }

    [Fact]
    public void GeometryEncoderAddsPointAndBoxPositionalFeatures()
    {
        manual_seed(47);
        using var encoder = new Sam3GeometryEncoderNew(d_model: 8, num_geo_layers: 0);
        using var points = tensor(new float[,,] { { { 0.25f, 0.5f } } });
        using var boxes = tensor(new float[,,] { { { 0.5f, 0.5f, 0.25f, 0.75f } } });
        using var feature = randn(1, 8, 2, 2);
        var prompt = new Sam3Prompt(point_embeddings: points, box_embeddings: boxes);

        var (tokens, mask) = encoder.forward(prompt, [feature], [[2L, 2L]]);
        using (tokens)
        using (mask)
        {
            Assert.Equal([2L, 1L, 8L], tokens.shape);
            Assert.True(isfinite(tokens).all().item<bool>());
        }
    }

    [Fact]
    public void GeometryEncoderPreservesPointAndBoxPaddingMasks()
    {
        using var encoder = new Sam3GeometryEncoderNew(d_model: 8, num_geo_layers: 0);
        using var points = zeros(2, 1, 2);
        using var boxes = zeros(1, 1, 4);
        using var pointMask = tensor(new bool[,] { { false, true } });
        using var boxMask = tensor(new bool[,] { { true } });
        using var feature = zeros(1, 8, 2, 2);
        var prompt = new Sam3Prompt(
            point_embeddings: points, point_mask: pointMask,
            box_embeddings: boxes, box_mask: boxMask);

        var (tokens, mask) = encoder.forward(prompt, [feature], [[2L, 2L]]);
        using (tokens)
        using (mask)
        {
            Assert.Equal([3L, 1L, 8L], tokens.shape);
            Assert.Equal([1L, 3L], mask.shape);
            Assert.True(mask[0, 0].item<bool>() == false);
            Assert.True(mask[0, 1].item<bool>());
            Assert.True(mask[0, 2].item<bool>());
        }
    }

    [Fact]
    public void DetrEncoderFusesSequenceFirstImageAndPromptWithDifferentLengths()
    {
        manual_seed(53);
        using var encoder = new Sam3TransformerEncoder(
            d_model: 8, nhead: 2, num_layers: 1, dim_feedforward: 16, num_feature_levels: 1);
        using var image = randn(2, 8, 2, 2);
        using var imagePos = randn(2, 8, 2, 2);
        using var prompt = randn(3, 2, 8);
        using var promptMask = zeros(2, 3, dtype: ScalarType.Bool);

        var output = encoder.forward([image], null, [imagePos], prompt, promptMask);
        using var memory = (Tensor)output["memory"];
        Assert.Equal([4L, 2L, 8L], memory.shape);
        Assert.True(isfinite(memory).all().item<bool>());
    }

    [Fact]
    public void DetrEncoderIgnoresPaddedPromptValues()
    {
        manual_seed(59);
        using var encoder = new Sam3TransformerEncoder(
            d_model: 8, nhead: 2, num_layers: 1, dim_feedforward: 16, num_feature_levels: 1);
        using var image = randn(1, 8, 1, 2);
        using var imagePos = zeros(1, 8, 1, 2);
        using var promptA = cat([randn(1, 1, 8), zeros(2, 1, 8)], dim: 0);
        using var promptB = cat([promptA.narrow(0, 0, 1), full([2, 1, 8], 1000f)], dim: 0);
        using var promptMask = tensor(new bool[,] { { false, true, true } });

        using var memoryA = (Tensor)encoder.forward([image], null, [imagePos], promptA, promptMask)["memory"];
        using var memoryB = (Tensor)encoder.forward([image], null, [imagePos], promptB, promptMask)["memory"];
        Assert.True(allclose(memoryA, memoryB, rtol: 1e-5, atol: 1e-6));
    }

    [Fact]
    public void GeometryEncoderRejectsBatchFirstPromptShape()
    {
        using var encoder = new Sam3GeometryEncoderNew(d_model: 8, num_geo_layers: 0);
        using var points = zeros(2, 1, 2);
        using var feature = zeros(2, 8, 2, 2);

        var exception = Assert.Throws<ArgumentException>(() => encoder.forward(
            new Sam3Prompt(point_embeddings: points), [feature], [[2L, 2L]]));
        Assert.Contains("sequence, batch", exception.Message);
    }

    [Fact]
    public void EmptyGeometryPromptUsesPersistentClsEmbedding()
    {
        manual_seed(31);
        using var encoder = new Sam3GeometryEncoderNew(d_model: 8, num_geo_layers: 0);
        var (firstTokens, firstMask) = encoder.EncodeEmptyPrompt(2, CPU);
        var (secondTokens, secondMask) = encoder.EncodeEmptyPrompt(2, CPU);
        using (firstTokens)
        using (firstMask)
        using (secondTokens)
        using (secondMask)
        {
            Assert.Equal([1L, 2L, 8L], firstTokens.shape);
            Assert.True(equal(firstTokens, secondTokens).all().item<bool>());
            Assert.False(firstMask.any().item<bool>());
            Assert.True(equal(firstMask, secondMask).all().item<bool>());
        }
    }

    [Fact]
    public void ProducesPerQueryMaskLogitsAtHighestFpnResolution()
    {
        manual_seed(17);
        using var head = new Sam3MaskDecoder(d_model: 8);
        using var objectQueries = randn(1, 3, 8);
        using var encoderHiddenStates = randn(1, 1, 8);
        using var prompt = randn(2, 1, 8);
        using var promptMask = zeros(1, 2, dtype: ScalarType.Bool);
        using var level0 = randn(1, 8, 8, 8);
        using var level1 = randn(1, 8, 4, 4);
        using var level2 = randn(1, 8, 2, 2);
        using var level3 = randn(1, 8, 1, 1);

        var (masks, semantic) = head.forward(
            objectQueries,
            [level0, level1, level2, level3],
            encoderHiddenStates,
            prompt,
            promptMask);
        using (masks)
        using (semantic)
        {
            Assert.Equal([1L, 3L, 8L, 8L], masks.shape);
            Assert.Equal([1L, 1L, 8L, 8L], semantic.shape);
            Assert.True(isfinite(masks).all().item<bool>());
        }
    }

    [Fact]
    public void PromptAttentionIgnoresPaddingTokensAndRejectsAllPadding()
    {
        manual_seed(23);
        using var attention = new Sam3PromptCrossAttn(d_model: 8, nhead: 2);
        using var query = randn(3, 1, 8);
        using var validToken = randn(1, 1, 8);
        using var paddingA = zeros(2, 1, 8);
        using var paddingB = full([2, 1, 8], 1000f);
        using var keyValueA = cat([validToken, paddingA], dim: 0);
        using var keyValueB = cat([validToken, paddingB], dim: 0);
        using var paddingMask = tensor(new bool[,] { { false, true, true } });
        using var outputA = attention.forward(query, keyValueA, paddingMask);
        using var outputB = attention.forward(query, keyValueB, paddingMask);

        Assert.True(allclose(outputA, outputB, rtol: 1e-5, atol: 1e-6));

        using var allPadding = ones(1, 3, dtype: ScalarType.Bool);
        var exception = Assert.Throws<ArgumentException>(() => attention.forward(query, keyValueA, allPadding));
        Assert.Contains("at least one token", exception.Message);
    }

    [Fact]
    public void DotProductScoringIgnoresPaddingTokens()
    {
        manual_seed(29);
        using var scoring = new Sam3DotProductScoring(d_model: 8, d_proj: 8);
        using var queries = randn(1, 1, 2, 8);
        using var validToken = randn(1, 1, 8);
        using var paddingA = zeros(2, 1, 8);
        using var paddingB = full([2, 1, 8], -1000f);
        using var promptA = cat([validToken, paddingA], dim: 0);
        using var promptB = cat([validToken, paddingB], dim: 0);
        using var paddingMask = tensor(new bool[,] { { false, true, true } });
        using var scoresA = scoring.forward(queries, promptA, paddingMask);
        using var scoresB = scoring.forward(queries, promptB, paddingMask);

        Assert.True(allclose(scoresA, scoresB, rtol: 1e-5, atol: 1e-6));
    }

    [Fact]
    public void RequiresFourFpnLevels()
    {
        using var head = new Sam3MaskDecoder(d_model: 8);
        using var objectQueries = zeros(1, 1, 8);
        using var encoderHiddenStates = zeros(1, 1, 8);
        using var prompt = zeros(1, 1, 8);
        using var feature = zeros(1, 8, 1, 1);

        var exception = Assert.Throws<ArgumentException>(() => head.forward(
            objectQueries,
            [feature],
            encoderHiddenStates,
            prompt,
            null));

        Assert.Contains("four FPN feature levels", exception.Message);
    }

    private static string FindRepositoryFile(params string[] parts)
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            var candidate = Path.Combine([directory.FullName, .. parts]);
            if (File.Exists(candidate)) return candidate;
            directory = directory.Parent;
        }
        throw new FileNotFoundException($"Could not locate repository file '{Path.Combine(parts)}'.");
    }
}