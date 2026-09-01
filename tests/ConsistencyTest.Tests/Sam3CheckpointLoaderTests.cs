using SAMTorchSharp.Modeling.Sam3;

namespace ConsistencyTest.Tests;

public sealed class Sam3CheckpointLoaderTests
{
    [Theory]
    [InlineData("model.safetensors", Sam3CheckpointFormat.OfficialSafetensors)]
    [InlineData("MODEL.SAFETENSORS", Sam3CheckpointFormat.OfficialSafetensors)]
    [InlineData("model.bin", Sam3CheckpointFormat.ConvertedBinary)]
    public void DetectsSupportedCheckpointFormats(string path, Sam3CheckpointFormat expected)
    {
        Assert.Equal(expected, Sam3CheckpointLoaderNew.DetectFormat(path));
    }

    [Theory]
    [InlineData("model.pt")]
    [InlineData("model.ckpt")]
    [InlineData("model")]
    public void RejectsUnknownCheckpointFormats(string path)
    {
        Assert.Throws<NotSupportedException>(() => Sam3CheckpointLoaderNew.DetectFormat(path));
    }

    [Theory]
    [InlineData("detector_model.vision_encoder.backbone.embeddings.patch_embeddings.projection.weight", "vision_backbone.patch_embed.proj.weight")]
    [InlineData("detector_model.vision_encoder.backbone.layers.3.layer_norm1.weight", "vision_backbone.block_3.norm1.weight")]
    [InlineData("detector_model.vision_encoder.neck.fpn_layers.2.scale_layers.0.weight", "fpn_neck.fpn_layer_2.scale_layers_0.weight")]
    [InlineData("detector_model.text_encoder.text_model.encoder.layers.7.self_attn.q_proj.weight", "text_encoder.transformer.encoder_layer_7.self_attn_q_proj.weight")]
    [InlineData("detector_model.detr_encoder.layers.1.self_attn.out_proj.bias", "transformer_encoder.layer_1.self_attn_out_proj.bias")]
    [InlineData("detector_model.detr_decoder.layers.2.text_cross_attn.q_proj.weight", "transformer_decoder.layer_2.ca_text_q_proj.weight")]
    [InlineData("detector_model.detr_decoder.box_head.layer3.bias", "transformer_decoder.box_head.2.bias")]
    [InlineData("detector_model.detr_decoder.presence_token.weight", "transformer_decoder.presence_token")]
    [InlineData("detector_model.mask_decoder.mask_embedder.layers.1.weight", "mask_decoder.mask_embedder.layer_1.weight")]
    [InlineData("detector_model.dot_product_scoring.text_mlp.layer1.weight", "dot_product_scoring.text_mlp_layer1.weight")]
    public void MapsRepresentativeOfficialKeys(string checkpointKey, string modelKey)
    {
        Assert.Equal(modelKey, Sam3CheckpointLoaderNew.MapOfficialKey(checkpointKey));
    }

    [Theory]
    [InlineData("tracker_model.memory_encoder.weight")]
    [InlineData("detector_model.geometry_encoder.cls_embed.weight")]
    [InlineData("detector_model.mask_decoder.semantic_projection.weight")]
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
}