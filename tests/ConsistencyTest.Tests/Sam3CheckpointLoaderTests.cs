using SAMTorchSharp.Modeling.Sam3;
using System.Text.Json;

namespace ConsistencyTest.Tests;

public sealed class Sam3CheckpointLoaderTests
{
    [Theory]
    [InlineData("model.safetensors", Sam3CheckpointFormat.OfficialSafetensors)]
    [InlineData("MODEL.SAFETENSORS", Sam3CheckpointFormat.OfficialSafetensors)]
    [InlineData("model.bin", Sam3CheckpointFormat.ConvertedBinary)]
    [InlineData("model.pt", Sam3CheckpointFormat.ConvertedBinary)]
    public void DetectsSupportedCheckpointFormats(string path, Sam3CheckpointFormat expected)
    {
        Assert.Equal(expected, Sam3CheckpointLoaderNew.DetectFormat(path));
    }

    [Theory]
    [InlineData("model.ckpt")]
    [InlineData("model")]
    public void RejectsUnknownCheckpointFormats(string path)
    {
        Assert.Throws<NotSupportedException>(() => Sam3CheckpointLoaderNew.DetectFormat(path));
    }

    [Theory]
    [InlineData("detector_model.vision_encoder.backbone.embeddings.patch_embeddings.projection.weight", "vision_backbone.patch_embed.proj.weight")]
    [InlineData("detector_model.vision_encoder.backbone.layers.3.layer_norm1.weight", "vision_backbone.block_3.norm1.weight")]
    [InlineData("detector_model.vision_encoder.neck.fpn_layers.0.scale_layers.0.weight", "fpn_neck.fpn_layer_0.deconv1.weight")]
    [InlineData("detector_model.vision_encoder.neck.fpn_layers.0.scale_layers.2.bias", "fpn_neck.fpn_layer_0.deconv2.bias")]
    [InlineData("detector_model.vision_encoder.neck.fpn_layers.3.proj2.bias", "fpn_neck.fpn_layer_3.proj2.bias")]
    [InlineData("detector_model.text_encoder.text_model.encoder.layers.7.self_attn.q_proj.weight", "text_encoder.transformer.encoder_layer_7.self_attn_q_proj.weight")]
    [InlineData("detector_model.detr_encoder.layers.1.self_attn.out_proj.bias", "transformer_encoder.layer_1.self_attn_out_proj.bias")]
    [InlineData("detector_model.detr_decoder.layers.2.text_cross_attn.q_proj.weight", "transformer_decoder.layer_2.ca_text_q_proj.weight")]
    [InlineData("detector_model.detr_decoder.box_head.layer3.bias", "transformer_decoder.box_head.2.bias")]
    [InlineData("detector_model.detr_decoder.presence_token.weight", "transformer_decoder.presence_token")]
    [InlineData("detector_model.detr_decoder.presence_token_head.layers.0.weight", "transformer_decoder.presence_head.0.weight")]
    [InlineData("detector_model.detr_decoder.presence_token_head.layers.2.bias", "transformer_decoder.presence_head.2.bias")]
    [InlineData("detector_model.detr_decoder.presence_token_out_norm.weight", "transformer_decoder.presence_layer_norm.weight")]
    [InlineData("detector_model.geometry_encoder.cls_embed.weight", "geometry_encoder.cls_embed.weight")]
    [InlineData("detector_model.geometry_encoder.output_layer_norm.weight", "geometry_encoder.encode_norm.weight")]
    [InlineData("detector_model.geometry_encoder.prompt_layer_norm.bias", "geometry_encoder.final_norm.bias")]
    [InlineData("detector_model.geometry_encoder.vision_layer_norm.weight", "geometry_encoder.vision_layer_norm.weight")]
    [InlineData("detector_model.mask_decoder.mask_embedder.layers.1.weight", "mask_decoder.mask_embedder.layer_1.weight")]
    [InlineData("detector_model.mask_decoder.semantic_projection.weight", "mask_decoder.semantic_projection.weight")]
    [InlineData("detector_model.mask_decoder.instance_projection.bias", "mask_decoder.instance_projection.bias")]
    [InlineData("detector_model.dot_product_scoring.text_mlp.layer1.weight", "dot_product_scoring.text_mlp.0.weight")]
    [InlineData("detector_model.dot_product_scoring.text_mlp.layer2.bias", "dot_product_scoring.text_mlp.1.bias")]
    public void MapsRepresentativeOfficialKeys(string checkpointKey, string modelKey)
    {
        Assert.Equal(modelKey, Sam3CheckpointLoaderNew.MapOfficialKey(checkpointKey));
    }

    [Theory]
    [InlineData("tracker_model.memory_encoder.weight")]
    [InlineData("detector_model.unknown.weight")]
    public void ReturnsNullForUnsupportedOfficialKeys(string checkpointKey)
    {
        Assert.Null(Sam3CheckpointLoaderNew.MapOfficialKey(checkpointKey));
    }

    [Fact]
    public void ReportCoverageExcludesExplicitlyUnsupportedKeys()
    {
        var report = new Sam3CheckpointLoadReport(
            "model.safetensors",
            Sam3CheckpointFormat.OfficialSafetensors,
            5,
            ["loaded.one", "loaded.two"],
            ["missing"],
            ["tracker"],
            ["shape"]);

        Assert.Equal(4, report.LoadableTensorCount);
        Assert.Equal(50, report.Coverage);
        Assert.False(report.IsComplete);
    }

    [Fact]
    public void WritesStructuredCheckpointReport()
    {
        var outputDirectory = Path.Combine(Path.GetTempPath(), $"sam3-report-{Guid.NewGuid():N}");
        var report = new Sam3CheckpointLoadReport(
            "model.safetensors",
            Sam3CheckpointFormat.OfficialSafetensors,
            5,
            ["loaded.one", "loaded.two"],
            ["missing.one"],
            ["tracker.one"],
            ["shape.one: expected [2], checkpoint [3]"]);

        try
        {
            var reportPath = Program.WriteSam3CheckpointReport(outputDirectory, report);

            Assert.Equal(Path.Combine(outputDirectory, "checkpoint-report.json"), reportPath);
            using var document = JsonDocument.Parse(File.ReadAllText(reportPath));
            var root = document.RootElement;
            Assert.Equal("OfficialSafetensors", root.GetProperty("Format").GetString());
            Assert.Equal(5, root.GetProperty("CheckpointTensorCount").GetInt32());
            Assert.Equal(2, root.GetProperty("LoadedKeys").GetArrayLength());
            Assert.Equal("missing.one", root.GetProperty("MissingKeys")[0].GetString());
            Assert.Equal("tracker.one", root.GetProperty("SkippedKeys")[0].GetString());
            Assert.Equal("shape.one: expected [2], checkpoint [3]",
                root.GetProperty("ShapeMismatches")[0].GetString());
            Assert.Equal(4, root.GetProperty("LoadableTensorCount").GetInt32());
            Assert.Equal(50, root.GetProperty("Coverage").GetDouble());
            Assert.False(root.GetProperty("IsComplete").GetBoolean());
        }
        finally
        {
            if (Directory.Exists(outputDirectory))
                Directory.Delete(outputDirectory, recursive: true);
        }
    }

    [Fact]
    public void ConvertedBinaryLoaderRejectsMissingSiblingForPtPath()
    {
        var checkpointPath = Path.Combine(Path.GetTempPath(), $"sam3-{Guid.NewGuid():N}.pt");
        using var model = new BuildSam3New().WithTextEncoder(false).Build();
        var exception = Assert.Throws<FileNotFoundException>(() =>
            new Sam3CheckpointLoaderBinary().LoadModelWithReport(model, checkpointPath));
        Assert.Equal(Path.ChangeExtension(checkpointPath, ".bin"), exception.FileName);
    }
}
