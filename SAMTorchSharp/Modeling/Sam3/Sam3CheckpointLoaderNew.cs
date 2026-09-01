using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SAMTorchSharp.Modeling.Sam3;

public enum Sam3CheckpointFormat
{
    OfficialSafetensors,
    ConvertedBinary,
}

public sealed record Sam3CheckpointLoadReport(
    string Path,
    Sam3CheckpointFormat Format,
    int CheckpointTensorCount,
    IReadOnlyList<string> LoadedKeys,
    IReadOnlyList<string> MissingKeys,
    IReadOnlyList<string> SkippedKeys,
    IReadOnlyList<string> ShapeMismatches)
{
    public int LoadableTensorCount => LoadedKeys.Count + MissingKeys.Count + ShapeMismatches.Count;
    public double Coverage => LoadableTensorCount == 0 ? 0 : LoadedKeys.Count * 100.0 / LoadableTensorCount;
    public bool IsComplete => MissingKeys.Count == 0 && ShapeMismatches.Count == 0;
}

/// <summary>
/// Checkpoint loader for SAM3 that matches the checkpoint's actual architecture.
/// Uses TorchSharp.PyBridge.Safetensors for loading .safetensors files.
///
/// Checkpoint structure (from config.json analysis):
///   - detector_model.vision_encoder.backbone.* (ViT)
///   - detector_model.vision_encoder.neck.* (FPN)
///   - detector_model.text_encoder.text_model.* (BERT)
///   - detector_model.detr_encoder.layers.* (DETR Encoder)
///   - detector_model.detr_decoder.layers.* (DETR Decoder)
///   - detector_model.geometry_encoder.* (Geometry Encoder)
///   - detector_model.mask_decoder.* (Mask Decoder - tokens only)
///   - detector_model.dot_product_scoring.* (Scoring)
///   - tracker_model.* (Video tracking - not loaded)
///   - tracker_neck.* (Video neck - not loaded)
/// </summary>
public class Sam3CheckpointLoaderNew
{
    private enum AssignmentResult
    {
        Loaded,
        Missing,
        ShapeMismatch,
    }

    public static Sam3CheckpointFormat DetectFormat(string checkpointPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(checkpointPath);
        return Path.GetExtension(checkpointPath).ToLowerInvariant() switch
        {
            ".safetensors" => Sam3CheckpointFormat.OfficialSafetensors,
            ".bin" => Sam3CheckpointFormat.ConvertedBinary,
            var extension => throw new NotSupportedException(
                $"Unsupported SAM3 checkpoint extension '{extension}'. Use .safetensors or converted .bin."),
        };
    }

    /// <summary>
    /// Load a SAM3 model from a .safetensors checkpoint file.
    /// Returns (loaded_count, skipped_count, missing_count).
    /// </summary>
    public Tuple<int, int, int> LoadModel(Sam3BaseNew model, string checkpointPath, Device device = null)
    {
        var report = LoadModelWithReport(model, checkpointPath, device);
        return Tuple.Create(report.LoadedKeys.Count, report.SkippedKeys.Count,
            report.MissingKeys.Count + report.ShapeMismatches.Count);
    }

    public Sam3CheckpointLoadReport LoadModelWithReport(Sam3BaseNew model, string checkpointPath, Device device = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        var format = DetectFormat(checkpointPath);
        if (format != Sam3CheckpointFormat.OfficialSafetensors)
            throw new NotSupportedException("Sam3CheckpointLoaderNew loads official .safetensors checkpoints only.");

        var path = Path.GetFullPath(checkpointPath);
        if (!File.Exists(path))
            throw new FileNotFoundException("SAM3 checkpoint was not found.", path);

        device = device ?? CPU;
        var dev = device;

        var checkpoint = TorchSharp.PyBridge.Safetensors.LoadStateDict(path);
        var loaded = new List<string>();
        var missing = new List<string>();
        var unexpected = new List<string>();
        var shapeMismatches = new List<string>();

        foreach (var kvp in checkpoint)
        {
            var ckptKey = kvp.Key;
            var tensor = kvp.Value.to(dev);

            // Special handling: FPN scale_layers weights are transposed in checkpoint
            // Checkpoint stores [in_channels, out_channels, kH, kW], PyTorch expects [out_channels, in_channels, kH, kW]
            if (ckptKey.Contains("fpn_layers") && ckptKey.Contains("scale_layers") && ckptKey.EndsWith(".weight"))
            {
                tensor = tensor.transpose(0, 1);
            }

            var modelKey = MapOfficialKey(ckptKey);

            if (modelKey == null)
            {
                unexpected.Add(ckptKey);
                continue;
            }

            var assignment = TrySetValueByPath(model, modelKey, tensor, out var expectedShape);
            if (assignment == AssignmentResult.Loaded)
            {
                loaded.Add(modelKey);
            }
            else if (assignment == AssignmentResult.ShapeMismatch)
            {
                shapeMismatches.Add($"{ckptKey} -> {modelKey}: checkpoint=[{string.Join(",", tensor.shape)}], model=[{string.Join(",", expectedShape!)}]");
            }
            else
                missing.Add($"{ckptKey} -> {modelKey}");
        }

        loaded.Sort(StringComparer.Ordinal);
        missing.Sort(StringComparer.Ordinal);
        unexpected.Sort(StringComparer.Ordinal);
        shapeMismatches.Sort(StringComparer.Ordinal);
        var report = new Sam3CheckpointLoadReport(
            path, format, checkpoint.Count, loaded, missing, unexpected, shapeMismatches);
        Console.WriteLine($"[CheckpointLoader] Loaded: {loaded.Count}, Skipped: {unexpected.Count}, Missing: {missing.Count}, Shape mismatch: {shapeMismatches.Count}");
        Console.WriteLine($"[CheckpointLoader] Total checkpoint tensors: {checkpoint.Count}");
        return report;
    }

    public static string? MapOfficialKey(string ckptKey)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(ckptKey);
        if (ckptKey.StartsWith("detector_model."))
            ckptKey = ckptKey.Substring("detector_model.".Length);

        // Skip tracker_model and tracker_neck (video components not in this checkpoint)
        if (ckptKey.StartsWith("tracker_model.") || ckptKey.StartsWith("tracker_neck."))
            return null;

        // Vision backbone
        if (ckptKey.StartsWith("vision_encoder.backbone."))
        {
            var rest = ckptKey.Substring("vision_encoder.backbone.".Length);
            if (rest == "embeddings.patch_embeddings.projection.weight")
                return "vision_backbone.patch_embed.proj.weight";
            if (rest == "embeddings.position_embeddings" || rest == "embeddings.position_embeddings.weight")
                return "vision_backbone.pos_embed_buffer";
            if (rest == "layer_norm.weight")
                return "vision_backbone.norm.weight";
            if (rest == "layer_norm.bias")
                return "vision_backbone.norm.bias";
            if (rest.StartsWith("layers."))
                return MapViTLayerKey(rest);
        }

        // FPN Neck
        if (ckptKey.StartsWith("vision_encoder.neck.fpn_layers."))
        {
            var rest = ckptKey.Substring("vision_encoder.neck.fpn_layers.".Length);
            var parts = rest.Split('.');
            var level = parts[0];
            var param = string.Join(".", parts.Skip(1));

            // Normalize scale_layers.X to scale_layers_X format
            if (param.StartsWith("scale_layers."))
            {
                var sub = param.Substring("scale_layers.".Length);
                param = "scale_layers_" + sub;
            }

            return $"fpn_neck.fpn_layer_{level}.{param}";
        }

        // Text encoder
        if (ckptKey.StartsWith("text_encoder.text_model."))
        {
            var rest = ckptKey.Substring("text_encoder.text_model.".Length);
            if (rest == "embeddings.token_embedding.weight")
                return "text_encoder.transformer.token_embedding.weight";
            if (rest == "embeddings.position_embedding.weight")
                return "text_encoder.transformer.pos_embed_buffer";
            if (rest == "final_layer_norm.weight")
                return "text_encoder.transformer.final_layer_norm.weight";
            if (rest == "final_layer_norm.bias")
                return "text_encoder.transformer.final_layer_norm.bias";
            if (rest.StartsWith("encoder.layers."))
                return MapTextEncoderLayerKey(rest);
            if (rest == "text_projection.weight")
                return "text_encoder.text_projection.weight";
        }

        // DETR Encoder
        if (ckptKey.StartsWith("detr_encoder.layers."))
            return MapTransformerEncoderLayerKey(ckptKey);

        // DETR Decoder
        if (ckptKey.StartsWith("detr_decoder.layers."))
            return MapTransformerDecoderLayerKey(ckptKey);

        // DETR decoder box_head (3-layer MLP: 256->256->256->4)
        if (ckptKey.StartsWith("detr_decoder.box_head."))
            return MapBoxHeadKey(ckptKey);

        // DETR decoder ref_point_head (2-layer MLP: 512->256->256)
        if (ckptKey.StartsWith("detr_decoder.ref_point_head."))
            return MapRefPointHeadKey(ckptKey);

        // DETR decoder presence_head (3-layer MLP: 256->256->256->1)
        if (ckptKey.StartsWith("detr_decoder.presence_head."))
            return MapPresenceHeadKey(ckptKey);

        // DETR decoder box_rpb_embed_x/y (RoPE position embedding)
        if (ckptKey.Contains("detr_decoder.box_rpb_embed_x.") || ckptKey.Contains("detr_decoder.box_rpb_embed_y."))
            return MapBoxRPBKey(ckptKey, ckptKey.Contains("box_rpb_embed_x") ? "box_rpb_embed_x" : "box_rpb_embed_y");

        // DETR decoder presence_token (shape in checkpoint is [1, d_model], C# expects [1, 1, d_model])
        if (ckptKey == "detr_decoder.presence_token.weight")
            return "transformer_decoder.presence_token";

        // DETR decoder query_embed/norm keys
        if (ckptKey == "detr_decoder.query_embed.weight")
            return "transformer_decoder.query_embed";
        if (ckptKey == "detr_decoder.reference_points.weight")
            return "transformer_decoder.reference_points";
        if (ckptKey == "detr_decoder.output_layer_norm.weight")
            return "transformer_decoder.output_layer_norm.weight";
        if (ckptKey == "detr_decoder.output_layer_norm.bias")
            return "transformer_decoder.output_layer_norm.bias";
        if (ckptKey == "detr_decoder.presence_layer_norm.weight")
            return "transformer_decoder.presence_layer_norm.weight";
        if (ckptKey == "detr_decoder.presence_layer_norm.bias")
            return "transformer_decoder.presence_layer_norm.bias";

        // Geometry encoder
        if (ckptKey.StartsWith("geometry_encoder."))
        {
            var rest = ckptKey.Substring("geometry_encoder.".Length);
            // Skip geometry encoder modules not implemented in simplified C# version
            if (rest.StartsWith("cls_embed.") ||
                rest.StartsWith("boxes_direct_project.") ||
                rest.StartsWith("boxes_pool_project.") ||
                rest.StartsWith("boxes_pos_enc_project.") ||
                rest.StartsWith("output_layer_norm.") ||
                rest.StartsWith("prompt_layer_norm.") ||
                rest.StartsWith("vision_layer_norm."))
                return null;
            return MapGeometryEncoderKey(ckptKey);
        }

        // Skip mask decoder projection layers (stored as 4D Conv2d weights or wrong-shape biases in checkpoint, not Linear)
        if (ckptKey.Contains("mask_decoder.semantic_projection") || ckptKey.Contains("mask_decoder.instance_projection"))
            return null;

        // Mask decoder
        if (ckptKey.StartsWith("mask_decoder."))
            return MapMaskDecoderKey(ckptKey);

        // Dot product scoring
        if (ckptKey.StartsWith("dot_product_scoring."))
            return MapDotProductScoringKey(ckptKey);

        // Top-level text_projection (correct shape [256, 1024])
        if (ckptKey == "text_projection.weight")
            return "text_encoder.text_projection.weight";
        if (ckptKey == "text_projection.bias")
            return "text_encoder.text_projection.bias";

        // detector_model.text_projection (alternative location)
        if (ckptKey == "detector_model.text_projection.weight")
            return "text_encoder.text_projection.weight";
        if (ckptKey == "detector_model.text_projection.bias")
            return "text_encoder.text_projection.bias";

        return null;
    }

    private static string MapViTLayerKey(string rest)
    {
        var parts = rest.Split('.');
        var layerIdx = parts[1];
        var suffix = string.Join(".", parts.Skip(2));

        // Handle layer_norm1/layer_norm2 (ViT block norms)
        if (suffix == "layer_norm1.weight")
            return $"vision_backbone.block_{layerIdx}.norm1.weight";
        if (suffix == "layer_norm1.bias")
            return $"vision_backbone.block_{layerIdx}.norm1.bias";
        if (suffix == "layer_norm2.weight")
            return $"vision_backbone.block_{layerIdx}.norm2.weight";
        if (suffix == "layer_norm2.bias")
            return $"vision_backbone.block_{layerIdx}.norm2.bias";

        if (suffix.StartsWith("attention.")) suffix = suffix.Substring("attention.".Length);
        else if (suffix.StartsWith("mlp.")) suffix = suffix.Substring("mlp.".Length);
        return $"vision_backbone.block_{layerIdx}.{suffix}";
    }

    private static string MapTextEncoderLayerKey(string rest)
    {
        var parts = rest.Split('.');
        var layerIdx = parts[2];
        var suffix = string.Join(".", parts.Skip(3));

        if (suffix.StartsWith("self_attn."))
        {
            var sub = suffix.Substring("self_attn.".Length);
            // Checkpoint uses 'out_proj', C# model uses 'o_proj'
            if (sub == "out_proj.weight")
                return $"text_encoder.transformer.encoder_layer_{layerIdx}.self_attn_o_proj.weight";
            if (sub == "out_proj.bias")
                return $"text_encoder.transformer.encoder_layer_{layerIdx}.self_attn_o_proj.bias";
            return $"text_encoder.transformer.encoder_layer_{layerIdx}.self_attn_{sub}";
        }
        if (suffix == "layer_norm1.weight")
            return $"text_encoder.transformer.encoder_layer_{layerIdx}.layer_norm1.weight";
        if (suffix == "layer_norm1.bias")
            return $"text_encoder.transformer.encoder_layer_{layerIdx}.layer_norm1.bias";
        if (suffix == "layer_norm2.weight")
            return $"text_encoder.transformer.encoder_layer_{layerIdx}.layer_norm2.weight";
        if (suffix == "layer_norm2.bias")
            return $"text_encoder.transformer.encoder_layer_{layerIdx}.layer_norm2.bias";
        if (suffix == "mlp.fc1.weight")
            return $"text_encoder.transformer.encoder_layer_{layerIdx}.fc1.weight";
        if (suffix == "mlp.fc1.bias")
            return $"text_encoder.transformer.encoder_layer_{layerIdx}.fc1.bias";
        if (suffix == "mlp.fc2.weight")
            return $"text_encoder.transformer.encoder_layer_{layerIdx}.fc2.weight";
        if (suffix == "mlp.fc2.bias")
            return $"text_encoder.transformer.encoder_layer_{layerIdx}.fc2.bias";

        return null;
    }

    private static string MapTransformerEncoderLayerKey(string ckptKey)
    {
        var rest = ckptKey.Substring("detr_encoder.layers.".Length);
        var parts = rest.Split('.');
        var layerIdx = parts[0];
        var suffix = string.Join(".", parts.Skip(1));

        if (suffix.StartsWith("self_attn."))
        {
            var sub = suffix.Substring("self_attn.".Length);
            // C# fields: self_attn_q_proj, self_attn_k_proj, etc.
            return $"transformer_encoder.layer_{layerIdx}.self_attn_{sub}";
        }
        if (suffix.StartsWith("cross_attn."))
        {
            var sub = suffix.Substring("cross_attn.".Length);
            // C# fields: cross_attn_q_proj, cross_attn_k_proj, etc.
            return $"transformer_encoder.layer_{layerIdx}.cross_attn_{sub}";
        }
        if (suffix == "layer_norm1.weight")
            return $"transformer_encoder.layer_{layerIdx}.norm1.weight";
        if (suffix == "layer_norm1.bias")
            return $"transformer_encoder.layer_{layerIdx}.norm1.bias";
        if (suffix == "layer_norm2.weight")
            return $"transformer_encoder.layer_{layerIdx}.norm2.weight";
        if (suffix == "layer_norm2.bias")
            return $"transformer_encoder.layer_{layerIdx}.norm2.bias";
        if (suffix == "layer_norm3.weight")
            return $"transformer_encoder.layer_{layerIdx}.norm3.weight";
        if (suffix == "layer_norm3.bias")
            return $"transformer_encoder.layer_{layerIdx}.norm3.bias";
        if (suffix == "mlp.fc1.weight")
            return $"transformer_encoder.layer_{layerIdx}.linear1.weight";
        if (suffix == "mlp.fc1.bias")
            return $"transformer_encoder.layer_{layerIdx}.linear1.bias";
        if (suffix == "mlp.fc2.weight")
            return $"transformer_encoder.layer_{layerIdx}.linear2.weight";
        if (suffix == "mlp.fc2.bias")
            return $"transformer_encoder.layer_{layerIdx}.linear2.bias";

        return null;
    }

    /// <summary>
    /// Map DETR decoder layer keys.
    /// Checkpoint format: detr_decoder.layers.{N}.{self_attn|vision_cross_attn|text_cross_attn}.{q_proj|k_proj|v_proj|o_proj}
    /// Layer norms: detr_decoder.layers.{N}.{self_attn_layer_norm|vision_cross_attn_layer_norm|mlp_layer_norm|text_cross_attn_layer_norm}
    /// C# format: transformer_decoder.layer_{N}.{self_attn|cross_attn|ca_text}.{q_proj|k_proj|v_proj|o_proj}
    /// </summary>
    private static string MapTransformerDecoderLayerKey(string ckptKey)
    {
        var rest = ckptKey.Substring("detr_decoder.layers.".Length);
        var parts = rest.Split('.');
        var layerIdx = parts[0];
        var suffix = string.Join(".", parts.Skip(1));

        // self_attn: q_proj, k_proj, v_proj, o_proj (+ bias)
        if (suffix.StartsWith("self_attn."))
        {
            var sub = suffix.Substring("self_attn.".Length);
            return $"transformer_decoder.layer_{layerIdx}.self_attn_{sub}";
        }

        // cross_attn (visual): q_proj, k_proj, v_proj, o_proj (+ bias)
        if (suffix.StartsWith("vision_cross_attn."))
        {
            var sub = suffix.Substring("vision_cross_attn.".Length);
            return $"transformer_decoder.layer_{layerIdx}.cross_attn_{sub}";
        }

        // cross_attn_text (text): q_proj, k_proj, v_proj, o_proj (+ bias)
        if (suffix.StartsWith("text_cross_attn."))
        {
            var sub = suffix.Substring("text_cross_attn.".Length);
            return $"transformer_decoder.layer_{layerIdx}.ca_text_{sub}";
        }

        // layer_norm1 (after visual cross-attn)
        if (suffix == "vision_cross_attn_layer_norm.weight")
            return $"transformer_decoder.layer_{layerIdx}.layer_norm1.weight";
        if (suffix == "vision_cross_attn_layer_norm.bias")
            return $"transformer_decoder.layer_{layerIdx}.layer_norm1.bias";

        // layer_norm2 (after self-attn)
        if (suffix == "self_attn_layer_norm.weight")
            return $"transformer_decoder.layer_{layerIdx}.layer_norm2.weight";
        if (suffix == "self_attn_layer_norm.bias")
            return $"transformer_decoder.layer_{layerIdx}.layer_norm2.bias";

        // layer_norm3 (after MLP)
        if (suffix == "mlp_layer_norm.weight")
            return $"transformer_decoder.layer_{layerIdx}.layer_norm3.weight";
        if (suffix == "mlp_layer_norm.bias")
            return $"transformer_decoder.layer_{layerIdx}.layer_norm3.bias";

        // catext_norm (after text cross-attn)
        if (suffix == "text_cross_attn_layer_norm.weight")
            return $"transformer_decoder.layer_{layerIdx}.catext_norm.weight";
        if (suffix == "text_cross_attn_layer_norm.bias")
            return $"transformer_decoder.layer_{layerIdx}.catext_norm.bias";

        // mlp.fc1, mlp.fc2
        if (suffix == "mlp.fc1.weight")
            return $"transformer_decoder.layer_{layerIdx}.mlp_fc1.weight";
        if (suffix == "mlp.fc1.bias")
            return $"transformer_decoder.layer_{layerIdx}.mlp_fc1.bias";
        if (suffix == "mlp.fc2.weight")
            return $"transformer_decoder.layer_{layerIdx}.mlp_fc2.weight";
        if (suffix == "mlp.fc2.bias")
            return $"transformer_decoder.layer_{layerIdx}.mlp_fc2.bias";

        return null;
    }

    private static string MapBoxHeadKey(string ckptKey)
    {
        var rest = ckptKey.Substring("detr_decoder.box_head.".Length);
        var parts = rest.Split('.');
        var layerNum = parts[0]; // layer1, layer2, layer3
        var param = parts.Length > 1 ? parts[1] : "";

        int mlpLayer;
        if (layerNum == "layer1") mlpLayer = 0;
        else if (layerNum == "layer2") mlpLayer = 1;
        else if (layerNum == "layer3") mlpLayer = 2;
        else return null;

        // ModuleList children are indexed: box_head.0.weight, box_head.0.bias, etc.
        if (param == "weight")
            return $"transformer_decoder.box_head.{mlpLayer}.weight";
        if (param == "bias")
            return $"transformer_decoder.box_head.{mlpLayer}.bias";
        return null;
    }

    private static string MapPresenceHeadKey(string ckptKey)
    {
        var rest = ckptKey.Substring("detr_decoder.presence_head.".Length);
        var parts = rest.Split('.');
        var layerNum = parts[0];
        var param = parts.Length > 1 ? parts[1] : "";

        int mlpLayer;
        if (layerNum == "layer1") mlpLayer = 0;
        else if (layerNum == "layer2") mlpLayer = 1;
        else if (layerNum == "layer3") mlpLayer = 2;
        else return null;

        if (param == "weight")
            return $"transformer_decoder.presence_head.{mlpLayer}.weight";
        if (param == "bias")
            return $"transformer_decoder.presence_head.{mlpLayer}.bias";
        return null;
    }

    private static string MapRefPointHeadKey(string ckptKey)
    {
        var rest = ckptKey.Substring("detr_decoder.ref_point_head.".Length);
        var parts = rest.Split('.');
        var layerNum = parts[0];
        var param = parts.Length > 1 ? parts[1] : "";

        int mlpLayer;
        if (layerNum == "layer1") mlpLayer = 0;
        else if (layerNum == "layer2") mlpLayer = 1;
        else return null;

        if (param == "weight")
            return $"transformer_decoder.ref_point_head.{mlpLayer}.weight";
        if (param == "bias")
            return $"transformer_decoder.ref_point_head.{mlpLayer}.bias";
        return null;
    }

    private static string MapBoxRPBKey(string ckptKey, string name)
    {
        var rest = ckptKey.Substring($"detr_decoder.{name}.".Length);
        var parts = rest.Split('.');
        var layerNum = parts[0];
        var param = parts.Length > 1 ? parts[1] : "";

        int mlpLayer;
        if (layerNum == "layer1") mlpLayer = 0;
        else if (layerNum == "layer2") mlpLayer = 1;
        else return null;

        if (param == "weight")
            return $"transformer_decoder.{name}.{mlpLayer}.weight";
        if (param == "bias")
            return $"transformer_decoder.{name}.{mlpLayer}.bias";
        return null;
    }

    private static string MapGeometryEncoderKey(string ckptKey)
    {
        var rest = ckptKey.Substring("geometry_encoder.".Length);

        if (rest.StartsWith("layers."))
        {
            var parts = rest.Split('.');
            var layerIdx = parts[1];
            var suffix = string.Join(".", parts.Skip(2));

            if (suffix.StartsWith("self_attn."))
            {
                var sub = suffix.Substring("self_attn.".Length);
                return $"geometry_encoder.geo_layer_{layerIdx}.self_attn_{sub}";
            }
            if (suffix.StartsWith("cross_attn."))
            {
                var sub = suffix.Substring("cross_attn.".Length);
                return $"geometry_encoder.geo_layer_{layerIdx}.cross_attn_{sub}";
            }
            if (suffix == "layer_norm1.weight")
                return $"geometry_encoder.geo_layer_{layerIdx}.layer_norm1.weight";
            if (suffix == "layer_norm1.bias")
                return $"geometry_encoder.geo_layer_{layerIdx}.layer_norm1.bias";
            if (suffix == "layer_norm2.weight")
                return $"geometry_encoder.geo_layer_{layerIdx}.layer_norm2.weight";
            if (suffix == "layer_norm2.bias")
                return $"geometry_encoder.geo_layer_{layerIdx}.layer_norm2.bias";
            if (suffix == "layer_norm3.weight")
                return $"geometry_encoder.geo_layer_{layerIdx}.layer_norm3.weight";
            if (suffix == "layer_norm3.bias")
                return $"geometry_encoder.geo_layer_{layerIdx}.layer_norm3.bias";
            if (suffix == "mlp.fc1.weight")
                return $"geometry_encoder.geo_layer_{layerIdx}.mlp_fc1.weight";
            if (suffix == "mlp.fc1.bias")
                return $"geometry_encoder.geo_layer_{layerIdx}.mlp_fc1.bias";
            if (suffix == "mlp.fc2.weight")
                return $"geometry_encoder.geo_layer_{layerIdx}.mlp_fc2.weight";
            if (suffix == "mlp.fc2.bias")
                return $"geometry_encoder.geo_layer_{layerIdx}.mlp_fc2.bias";
        }

        // Top-level geometry encoder keys
        if (rest == "cls_embed.weight")
            return "geometry_encoder.cls_embed.weight";
        if (rest == "label_embed.weight")
            return "geometry_encoder.label_embed.weight";
        if (rest == "final_proj.weight")
            return "geometry_encoder.final_proj.weight";
        if (rest == "final_proj.bias")
            return "geometry_encoder.final_proj.bias";
        if (rest == "boxes_direct_project.weight")
            return "geometry_encoder.boxes_direct_project.weight";
        if (rest == "boxes_direct_project.bias")
            return "geometry_encoder.boxes_direct_project.bias";
        if (rest == "boxes_pool_project.weight")
            return "geometry_encoder.boxes_pool_project.weight";
        if (rest == "boxes_pool_project.bias")
            return "geometry_encoder.boxes_pool_project.bias";
        if (rest == "boxes_pos_enc_project.weight")
            return "geometry_encoder.boxes_pos_enc_project.weight";
        if (rest == "boxes_pos_enc_project.bias")
            return "geometry_encoder.boxes_pos_enc_project.bias";
        if (rest == "output_layer_norm.weight")
            return "geometry_encoder.output_layer_norm.weight";
        if (rest == "output_layer_norm.bias")
            return "geometry_encoder.output_layer_norm.bias";
        if (rest == "prompt_layer_norm.weight")
            return "geometry_encoder.prompt_layer_norm.weight";
        if (rest == "prompt_layer_norm.bias")
            return "geometry_encoder.prompt_layer_norm.bias";
        if (rest == "vision_layer_norm.weight")
            return "geometry_encoder.vision_layer_norm.weight";
        if (rest == "vision_layer_norm.bias")
            return "geometry_encoder.vision_layer_norm.bias";

        return $"geometry_encoder.{rest}";
    }

    private static string MapMaskDecoderKey(string ckptKey)
    {
        var rest = ckptKey.Substring("mask_decoder.".Length);

        if (rest.StartsWith("pixel_decoder.conv_layers."))
        {
            var parts = rest.Split('.');
            var layerIdx = parts[2];
            var suffix = string.Join(".", parts.Skip(3));
            return $"mask_decoder.pixel_decoder.conv_{layerIdx}.{suffix}";
        }
        if (rest.StartsWith("pixel_decoder.norms."))
        {
            var parts = rest.Split('.');
            var layerIdx = parts[2];
            var suffix = string.Join(".", parts.Skip(3));
            return $"mask_decoder.pixel_decoder.norm_{layerIdx}.{suffix}";
        }
        if (rest.StartsWith("mask_embedder.layers."))
        {
            var parts = rest.Split('.');
            var layerIdx = parts[2];
            var suffix = string.Join(".", parts.Skip(3));
            return $"mask_decoder.mask_embedder.layer_{layerIdx}.{suffix}";
        }
        if (rest.StartsWith("prompt_cross_attn."))
        {
            var sub = rest.Substring("prompt_cross_attn.".Length);
            return $"mask_decoder.prompt_cross_attn.{sub}";
        }
        if (rest.StartsWith("prompt_cross_attn_norm."))
        {
            var sub = rest.Substring("prompt_cross_attn_norm.".Length);
            return $"mask_decoder.prompt_cross_attn_norm.{sub}";
        }
        if (rest == "semantic_projection.weight")
            return "mask_decoder.semantic_projection.weight";
        if (rest == "semantic_projection.bias")
            return "mask_decoder.semantic_projection.bias";
        if (rest == "instance_projection.weight")
            return "mask_decoder.instance_projection.weight";
        if (rest == "instance_projection.bias")
            return "mask_decoder.instance_projection.bias";

        return $"mask_decoder.{rest}";
    }

    private static string MapDotProductScoringKey(string ckptKey)
    {
        var rest = ckptKey.Substring("dot_product_scoring.".Length);
        if (rest.StartsWith("text_mlp.layer"))
        {
            var sub = rest.Replace("text_mlp.layer1", "text_mlp_layer1").Replace("text_mlp.layer2", "text_mlp_layer2");
            return $"dot_product_scoring.{sub}";
        }
        if (rest == "text_mlp_out_norm.weight")
            return "dot_product_scoring.text_mlp_out_norm.weight";
        if (rest == "text_mlp_out_norm.bias")
            return "dot_product_scoring.text_mlp_out_norm.bias";
        if (rest == "text_proj.weight")
            return "dot_product_scoring.text_proj.weight";
        if (rest == "text_proj.bias")
            return "dot_product_scoring.text_proj.bias";
        if (rest == "query_proj.weight")
            return "dot_product_scoring.query_proj.weight";
        if (rest == "query_proj.bias")
            return "dot_product_scoring.query_proj.bias";
        return $"dot_product_scoring.{rest}";
    }

    /// <summary>
    /// Set a tensor value in the model at the given dotted path.
    /// Walks the module tree by dotted name, finds the leaf parameter, and copies data.
    /// Handles ModuleList indexing (e.g., "box_head.layers[0].weight").
    /// </summary>
    private AssignmentResult TrySetValueByPath(Module module, string path, Tensor value, out long[]? expectedShape)
    {
        expectedShape = null;
        var parts = path.Split('.');
        Module current = module;

        for (int i = 0; i < parts.Length - 1; i++)
        {
            string part = parts[i];
            int bracketStart = part.IndexOf('[');

            string moduleName;
            int index = -1;

            if (bracketStart >= 0)
            {
                moduleName = part.Substring(0, bracketStart);
                var idxStr = part.Substring(bracketStart + 1, part.IndexOf(']') - bracketStart - 1);
                index = int.Parse(idxStr);
            }
            else
            {
                moduleName = part;
            }

            // First try as a named child
            Module child = null;
            var children = current.named_children().ToList();

            // Find all children with this name
            var matchingChildren = children.Where(kvp => kvp.Item1 == moduleName).ToList();

            if (matchingChildren.Count == 1)
            {
                child = matchingChildren[0].Item2;
            }
            else if (matchingChildren.Count > 1 && index >= 0)
            {
                // Multiple children with same name - pick by index
                if (index < matchingChildren.Count)
                    child = matchingChildren[index].Item2;
            }
            else if (matchingChildren.Count == 0)
            {
                // Try as a field/property on the current module
                // This handles ModuleList fields like box_head, ref_point_head, etc.
                var fields = current.GetType().GetFields(System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic);
                foreach (var field in fields)
                {
                    if (field.Name == moduleName)
                    {
                        var fieldValue = field.GetValue(current);
                        if (fieldValue is IList<Module> moduleList && index >= 0 && index < moduleList.Count)
                        {
                            child = moduleList[index];
                            break;
                        }
                        else if (fieldValue is Module singleModule)
                        {
                            child = singleModule;
                            break;
                        }
                    }
                }

                // Also try properties
                if (child == null)
                {
                    var props = current.GetType().GetProperties(System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic);
                    foreach (var prop in props)
                    {
                        if (prop.Name == moduleName)
                        {
                            var propValue = prop.GetValue(current);
                            if (propValue is IList<Module> propModuleList && index >= 0 && index < propModuleList.Count)
                            {
                                child = propModuleList[index];
                                break;
                            }
                            else if (propValue is Module propSingleModule)
                            {
                                child = propSingleModule;
                                break;
                            }
                        }
                    }
                }
            }

            if (child == null)
                return AssignmentResult.Missing;
            current = child;
        }

        var finalKey = parts[parts.Length - 1];

        // Handle the case where finalKey refers to a Tensor field on the current module
        // (e.g., presence_token is a Tensor field, not a parameter/buffer)
        Tensor? fieldTensor = null;
        var fieldType = current.GetType();
        var fieldMember = fieldType.GetField(finalKey, System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic);
        if (fieldMember is not null && fieldMember.FieldType == typeof(Tensor))
        {
            fieldTensor = (Tensor)fieldMember.GetValue(current);
        }
        if (fieldTensor is not null)
        {
            expectedShape = fieldTensor.shape;
            // Handle shape mismatch: checkpoint may store [1, d_model] but C# expects [1, 1, d_model]
            if (fieldTensor.shape.Length != value.shape.Length)
            {
                // Try expanding value dimensions to match fieldTensor
                if (value.shape.Length == fieldTensor.shape.Length - 1)
                {
                    // Insert a dimension of size 1 at position 1
                    using var expanded = value.unsqueeze(1);
                    if (expanded.shape.Length == fieldTensor.shape.Length)
                    {
                        bool shapesMatch = true;
                        for (int i = 0; i < expanded.shape.Length; i++)
                        {
                            if (expanded.shape[i] != fieldTensor.shape[i])
                            {
                                shapesMatch = false;
                                break;
                            }
                        }
                        if (shapesMatch)
                        {
                            using var srcFld = expanded.contiguous();
                            using var _fld = torch.no_grad();
                            fieldTensor.copy_(srcFld);
                            return AssignmentResult.Loaded;
                        }
                    }
                }
            }
            else if (fieldTensor.shape.Length == value.shape.Length)
            {
                bool shapesMatch = true;
                for (int i = 0; i < fieldTensor.shape.Length; i++)
                {
                    if (fieldTensor.shape[i] != value.shape[i])
                    {
                        shapesMatch = false;
                        break;
                    }
                }
                if (shapesMatch)
                {
                    using var srcFld = value.contiguous();
                    using var _fld = torch.no_grad();
                    fieldTensor.copy_(srcFld);
                    return AssignmentResult.Loaded;
                }
            }
            return AssignmentResult.ShapeMismatch;
        }

        // First try to find as a buffer
        Tensor? buffer = null;
        try
        {
            buffer = current.get_buffer(finalKey);
        }
        catch { }
        if (buffer is not null)
        {
            expectedShape = buffer.shape;
            if (buffer.shape.Length == value.shape.Length)
            {
                bool shapesMatch = true;
                for (int i = 0; i < buffer.shape.Length; i++)
                {
                    if (buffer.shape[i] != value.shape[i])
                    {
                        shapesMatch = false;
                        break;
                    }
                }
                if (shapesMatch)
                {
                    using var srcBuf = value.contiguous();
                    using var _buf = torch.no_grad();
                    buffer.copy_(srcBuf);
                    return AssignmentResult.Loaded;
                }
            }
            return AssignmentResult.ShapeMismatch;
        }

        // Fall back to looking for a parameter
        Parameter param = null;
        foreach (var kvp in current.named_parameters(false))
        {
            if (kvp.Item1 == finalKey)
            {
                param = kvp.Item2;
                break;
            }
        }
        if ((object)param == null)
            return AssignmentResult.Missing;

        expectedShape = param.shape;
        if (param.shape.Length != value.shape.Length)
            return AssignmentResult.ShapeMismatch;
        for (int i = 0; i < param.shape.Length; i++)
        {
            if (param.shape[i] != value.shape[i])
                return AssignmentResult.ShapeMismatch;
        }

        using var src = value.contiguous();
        using var _ = torch.no_grad();
        param.copy_(src);
        return AssignmentResult.Loaded;
    }
}
