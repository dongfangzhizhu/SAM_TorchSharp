using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SAMTorchSharp.Modeling.Sam3;

public class Sam3CheckpointLoaderBinary
{
    public enum CheckpointFormat { Safetensors, PtHuggingFace }

    public Tuple<int, int, int, CheckpointFormat> LoadModel(
        Sam3BaseNew model, string checkpointPath, Device device = null)
    {
        device = device ?? CPU;
        var dev = device;
        var format = checkpointPath.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)
            ? CheckpointFormat.Safetensors : CheckpointFormat.PtHuggingFace;

        Dictionary<string, Tensor> checkpoint;
        if (format == CheckpointFormat.Safetensors)
        {
            Console.WriteLine($"[CheckpointLoader] Loading safetensors: {checkpointPath}");
            checkpoint = TorchSharp.PyBridge.Safetensors.LoadStateDict(checkpointPath);
        }
        else
        {
            var binPath = checkpointPath.Replace(".pt", ".bin");
            if (!File.Exists(binPath))
            {
                Console.WriteLine($"[CheckpointLoader] ERROR: Binary file not found: {binPath}");
                return Tuple.Create(0, 0, 0, CheckpointFormat.PtHuggingFace);
            }
            checkpoint = ReadBinaryCheckpoint(binPath);
            Console.WriteLine($"[CheckpointLoader] Loaded {checkpoint.Count} tensors from binary");
        }

        int loaded = 0, skipped = 0, missing = 0;

        foreach (var kvp in checkpoint)
        {
            var ckptKey = kvp.Key;
            var tensor = kvp.Value.to(dev);

            if (ckptKey.Contains("fpn_layers") && ckptKey.Contains("scale_layers") && ckptKey.EndsWith(".weight"))
                tensor = tensor.transpose(0, 1);

            var modelKey = MapCheckpointKeyToModelKey(ckptKey, format);
            if (modelKey == null) { skipped++; continue; }

            // Handle fused qkv -> separate projections
            if (modelKey.EndsWith(".self_attn_qkv.weight") || modelKey.EndsWith(".self_attn_qkv.bias"))
            {
                var baseKey = modelKey.Substring(0, modelKey.LastIndexOf('.'));
                var suffix = modelKey.Substring(modelKey.LastIndexOf('.') + 1);
                var d = (int)tensor.size(-1) / 3;
                using (var q = tensor.narrow(-1, 0, (long)d))
                using (var k = tensor.narrow(-1, (long)d, (long)d))
                using (var v = tensor.narrow(-1, (long)d * 2, (long)d))
                using (var _ng = torch.no_grad())
                {
                    TrySetValueByPath(model, $"{baseKey}.q_proj.{suffix}", q.contiguous());
                    TrySetValueByPath(model, $"{baseKey}.k_proj.{suffix}", k.contiguous());
                    TrySetValueByPath(model, $"{baseKey}.v_proj.{suffix}", v.contiguous());
                }
                loaded += 3;
                continue;
            }

            if (modelKey.EndsWith(".self_attn_in_proj.weight") || modelKey.EndsWith(".self_attn_in_proj.bias"))
            {
                var baseKey = modelKey.Substring(0, modelKey.LastIndexOf('.'));
                var suffix = modelKey.Substring(modelKey.LastIndexOf('.') + 1);
                var d = (int)tensor.size(-1) / 3;
                using (var q = tensor.narrow(-1, 0, (long)d))
                using (var k = tensor.narrow(-1, (long)d, (long)d))
                using (var v = tensor.narrow(-1, (long)d * 2, (long)d))
                using (var _ng = torch.no_grad())
                {
                    TrySetValueByPath(model, $"{baseKey}.q_proj.{suffix}", q.contiguous());
                    TrySetValueByPath(model, $"{baseKey}.k_proj.{suffix}", k.contiguous());
                    TrySetValueByPath(model, $"{baseKey}.v_proj.{suffix}", v.contiguous());
                }
                loaded += 3;
                continue;
            }

            if (modelKey.Contains(".self_attn.") && modelKey.Contains("in_proj"))
            {
                var baseKey = modelKey.Substring(0, modelKey.LastIndexOf('.'));
                var suffix = modelKey.Substring(modelKey.LastIndexOf('.') + 1);
                var d = (int)tensor.size(-1) / 3;
                using (var q = tensor.narrow(-1, 0, (long)d))
                using (var k = tensor.narrow(-1, (long)d, (long)d))
                using (var v = tensor.narrow(-1, (long)d * 2, (long)d))
                using (var _ng = torch.no_grad())
                {
                    TrySetValueByPath(model, $"{baseKey}.q_proj.{suffix}", q.contiguous());
                    TrySetValueByPath(model, $"{baseKey}.k_proj.{suffix}", k.contiguous());
                    TrySetValueByPath(model, $"{baseKey}.v_proj.{suffix}", v.contiguous());
                }
                loaded += 3;
                continue;
            }

            if (TrySetValueByPath(model, modelKey, tensor))
                loaded++;
            else
                missing++;
        }

        Console.WriteLine($"[CheckpointLoader] Loaded: {loaded}, Skipped: {skipped}, Missing: {missing}");
        Console.WriteLine($"[CheckpointLoader] Total checkpoint tensors: {checkpoint.Count}");
        return Tuple.Create(loaded, skipped, missing, format);
    }

    private Dictionary<string, Tensor> ReadBinaryCheckpoint(string path)
    {
        var result = new Dictionary<string, Tensor>();
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
        using var br = new BinaryReader(fs);
        var count = br.ReadUInt32();
        for (int i = 0; i < count; i++)
        {
            var keyLen = br.ReadUInt32();
            var keyBytes = br.ReadBytes((int)keyLen);
            var key = System.Text.Encoding.UTF8.GetString(keyBytes);
            var shapeLen = br.ReadUInt32();
            var shape = new long[shapeLen];
            for (int j = 0; j < shapeLen; j++) shape[j] = br.ReadInt64();
            var dtypeByte = br.ReadByte();
            long totalElements = 1;
            foreach (var s in shape) totalElements *= s;
            var dataBytes = br.ReadBytes((int)(totalElements * 4));
            var floatArr = new float[dataBytes.Length / 4];
            System.Buffer.BlockCopy(dataBytes, 0, floatArr, 0, dataBytes.Length);
            result[key] = torch.tensor(floatArr, shape, dtype: ScalarType.Float32);
        }
        return result;
    }

    private string MapCheckpointKeyToModelKey(string ckptKey, CheckpointFormat format)
    {
        if (format == CheckpointFormat.Safetensors) return MapSafetensorsKey(ckptKey);
        else return MapPtHfKey(ckptKey);
    }

    private string MapSafetensorsKey(string ckptKey)
    {
        if (ckptKey.StartsWith("detector_model.")) ckptKey = ckptKey.Substring("detector_model.".Length);
        if (ckptKey.StartsWith("tracker_model.") || ckptKey.StartsWith("tracker_neck.")) return null;

        if (ckptKey.StartsWith("vision_encoder.backbone."))
        {
            var rest = ckptKey.Substring("vision_encoder.backbone.".Length);
            if (rest == "embeddings.patch_embeddings.projection.weight") return "vision_backbone.patch_embed.proj.weight";
            if (rest == "embeddings.position_embeddings" || rest == "embeddings.position_embeddings.weight") return "vision_backbone.pos_embed_buffer";
            if (rest == "layer_norm.weight") return "vision_backbone.norm.weight";
            if (rest == "layer_norm.bias") return "vision_backbone.norm.bias";
            if (rest.StartsWith("layers.")) return MapViTLayerKey(rest);
        }
        if (ckptKey.StartsWith("vision_encoder.neck.fpn_layers."))
        {
            var rest = ckptKey.Substring("vision_encoder.neck.fpn_layers.".Length);
            var parts = rest.Split('.');
            var level = parts[0];
            var param = string.Join(".", parts.Skip(1));
            if (param.StartsWith("scale_layers.")) param = "scale_layers_" + param.Substring("scale_layers.".Length);
            return $"fpn_neck.fpn_layer_{level}.{param}";
        }
        if (ckptKey.StartsWith("text_encoder.text_model."))
        {
            var rest = ckptKey.Substring("text_encoder.text_model.".Length);
            if (rest == "embeddings.token_embedding.weight") return "text_encoder.transformer.token_embedding.weight";
            if (rest == "embeddings.position_embedding.weight") return "text_encoder.transformer.pos_embed_buffer";
            if (rest == "final_layer_norm.weight") return "text_encoder.transformer.final_layer_norm.weight";
            if (rest == "final_layer_norm.bias") return "text_encoder.transformer.final_layer_norm.bias";
            if (rest.StartsWith("encoder.layers.")) return MapTextEncoderLayerKey(rest);
            if (rest == "text_projection.weight") return "text_encoder.text_projection.weight";
        }
        if (ckptKey == "text_projection.weight") return "text_encoder.text_projection.weight";
        if (ckptKey == "text_projection.bias") return "text_encoder.text_projection.bias";
        if (ckptKey.StartsWith("detr_encoder.layers.")) return MapTransformerEncoderLayerKey(ckptKey);
        if (ckptKey.StartsWith("detr_decoder.layers.")) return MapTransformerDecoderLayerKey(ckptKey);
        if (ckptKey.StartsWith("detr_decoder.box_head.")) return MapBoxHeadKey(ckptKey);
        if (ckptKey.StartsWith("detr_decoder.ref_point_head.")) return MapRefPointHeadKey(ckptKey);
        if (ckptKey.StartsWith("detr_decoder.presence_head.")) return MapPresenceHeadKey(ckptKey);
        if (ckptKey.Contains("detr_decoder.box_rpb_embed_x.") || ckptKey.Contains("detr_decoder.box_rpb_embed_y."))
        {
            var isX = ckptKey.Contains("box_rpb_embed_x");
            return MapBoxRPBKey(ckptKey, isX ? "box_rpb_embed_x" : "box_rpb_embed_y");
        }
        if (ckptKey == "detr_decoder.presence_token.weight") return "transformer_decoder.presence_token";
        if (ckptKey == "detr_decoder.query_embed.weight") return "transformer_decoder.query_embed";
        if (ckptKey == "detr_decoder.reference_points.weight") return "transformer_decoder.reference_points";
        if (ckptKey == "detr_decoder.output_layer_norm.weight") return "transformer_decoder.output_layer_norm.weight";
        if (ckptKey == "detr_decoder.output_layer_norm.bias") return "transformer_decoder.output_layer_norm.bias";
        if (ckptKey == "detr_decoder.presence_layer_norm.weight") return "transformer_decoder.presence_layer_norm.weight";
        if (ckptKey == "detr_decoder.presence_layer_norm.bias") return "transformer_decoder.presence_layer_norm.bias";
        if (ckptKey.StartsWith("geometry_encoder."))
        {
            var rest = ckptKey.Substring("geometry_encoder.".Length);
            if (rest.StartsWith("cls_embed.") || rest.StartsWith("boxes_direct_project.") ||
                rest.StartsWith("boxes_pool_project.") || rest.StartsWith("boxes_pos_enc_project.") ||
                rest.StartsWith("output_layer_norm.") || rest.StartsWith("prompt_layer_norm.") ||
                rest.StartsWith("vision_layer_norm.")) return null;
            return MapGeometryEncoderKey(ckptKey);
        }
        if (ckptKey.Contains("mask_decoder.semantic_projection") || ckptKey.Contains("mask_decoder.instance_projection")) return null;
        if (ckptKey.StartsWith("mask_decoder.")) return MapMaskDecoderKey(ckptKey);
        if (ckptKey.StartsWith("dot_product_scoring.")) return MapDotProductScoringKey(ckptKey);
        return null;
    }

    private string MapPtHfKey(string ckptKey)
    {
        if (!ckptKey.StartsWith("detector.")) return null;
        var rest = ckptKey.Substring("detector.".Length);
        if (rest.StartsWith("tracker.") || rest.StartsWith("sam2_predictor.")) return null;

        if (rest.StartsWith("backbone.vision_backbone.convs."))
        {
            var convRest = rest.Substring("backbone.vision_backbone.convs.".Length);
            var parts = convRest.Split('.');
            var level = parts[0];
            var param = string.Join(".", parts.Skip(1));
            return $"fpn_neck.fpn_layer_{level}.{param}";
        }

        if (rest.StartsWith("backbone.vision_backbone.trunk.blocks."))
        {
            var blockRest = rest.Substring("backbone.vision_backbone.trunk.blocks.".Length);
            var parts = blockRest.Split('.');
            var blockIdx = parts[0];
            var suffix = string.Join(".", parts.Skip(1));
            if (suffix == "norm1.weight") return $"vision_backbone.block_{blockIdx}.norm1.weight";
            if (suffix == "norm1.bias") return $"vision_backbone.block_{blockIdx}.norm1.bias";
            if (suffix == "norm2.weight") return $"vision_backbone.block_{blockIdx}.norm2.weight";
            if (suffix == "norm2.bias") return $"vision_backbone.block_{blockIdx}.norm2.bias";
            if (suffix.StartsWith("attn."))
            {
                var attnSuffix = suffix.Substring("attn.".Length);
                if (attnSuffix == "qkv.weight") return $"vision_backbone.block_{blockIdx}.self_attn_qkv.weight";
                if (attnSuffix == "qkv.bias") return $"vision_backbone.block_{blockIdx}.self_attn_qkv.bias";
                if (attnSuffix == "proj.weight") return $"vision_backbone.block_{blockIdx}.self_attn_o_proj.weight";
                if (attnSuffix == "proj.bias") return $"vision_backbone.block_{blockIdx}.self_attn_o_proj.bias";
            }
            if (suffix.StartsWith("mlp.fc1.weight")) return $"vision_backbone.block_{blockIdx}.fc1.weight";
            if (suffix.StartsWith("mlp.fc1.bias")) return $"vision_backbone.block_{blockIdx}.fc1.bias";
            if (suffix.StartsWith("mlp.fc2.weight")) return $"vision_backbone.block_{blockIdx}.fc2.weight";
            if (suffix.StartsWith("mlp.fc2.bias")) return $"vision_backbone.block_{blockIdx}.fc2.bias";
            return null;
        }

        if (rest == "backbone.vision_backbone.trunk.patch_embed.proj.weight") return "vision_backbone.patch_embed.proj.weight";
        if (rest == "backbone.vision_backbone.trunk.pos_embed") return "vision_backbone.pos_embed_buffer";
        if (rest == "backbone.vision_backbone.trunk.ln_pre.weight") return "vision_backbone.norm.weight";
        if (rest == "backbone.vision_backbone.trunk.ln_pre.bias") return "vision_backbone.norm.bias";

        if (rest.StartsWith("backbone.language_backbone.encoder."))
        {
            var langRest = rest.Substring("backbone.language_backbone.encoder.".Length);
            return MapHfTextKey(langRest);
        }

        if (rest.StartsWith("backbone.language_backbone.text_projection"))
        {
            var suffix = rest.Substring("backbone.language_backbone.text_projection".Length);
            return "text_encoder.text_projection" + suffix;
        }

        // === TRANSFORMER ENCODER ===
        if (rest.StartsWith("transformer.encoder."))
        {
            var encRest = rest.Substring("transformer.encoder.".Length);
            return MapTransformerEncoderLayerKey($"detr_encoder.layers.{encRest}");
        }

        // === TRANSFORMER DECODER ===
        if (rest.StartsWith("transformer.decoder."))
        {
            var decRest = rest.Substring("transformer.decoder.".Length);
            if (decRest.StartsWith("layers."))
            {
                var layerRest2 = decRest.Substring("layers.".Length);
                var parts = layerRest2.Split('.');
                var idx = parts[0];
                var suffix = string.Join(".", parts.Skip(1));
                return MapTransformerDecoderLayerKey($"detr_decoder.layers.{idx}.{suffix}");
            }
            if (decRest == "query_embed.weight") return "transformer_decoder.query_embed";
            if (decRest == "reference_points.weight") return "transformer_decoder.reference_points";
            if (decRest == "output_layer_norm.weight") return "transformer_decoder.output_layer_norm.weight";
            if (decRest == "output_layer_norm.bias") return "transformer_decoder.output_layer_norm.bias";
            if (decRest == "presence_layer_norm.weight") return "transformer_decoder.presence_layer_norm.weight";
            if (decRest == "presence_layer_norm.bias") return "transformer_decoder.presence_layer_norm.bias";
            if (decRest == "presence_token.weight") return "transformer_decoder.presence_token";

            // HF: presence_token_head.* -> presence_head.*
            if (decRest.StartsWith("presence_token_head.layers."))
            {
                var parts = decRest.Split('.');
                var layerIdx = parts[2];
                var suffix = parts.Length > 3 ? string.Join(".", parts.Skip(3)) : "";
                return $"transformer_decoder.presence_head.{layerIdx}.{suffix}";
            }
            if (decRest == "presence_token_out_norm.weight") return "transformer_decoder.presence_layer_norm.weight";
            if (decRest == "presence_token_out_norm.bias") return "transformer_decoder.presence_layer_norm.bias";

            // HF: norm.* at decoder level -> output_layer_norm
            if (decRest == "norm.weight") return "transformer_decoder.output_layer_norm.weight";
            if (decRest == "norm.bias") return "transformer_decoder.output_layer_norm.bias";

            if (decRest.StartsWith("box_head.")) return MapBoxHeadKey($"detr_decoder.{decRest}");
            if (decRest.StartsWith("ref_point_head.")) return MapRefPointHeadKey($"detr_decoder.{decRest}");

            // HF: bbox_embed.* -> box_head.*
            if (decRest.StartsWith("bbox_embed.layers."))
            {
                var parts = decRest.Split('.');
                var layerNum = parts[2];
                var suffix = parts.Length > 3 ? parts[3] : "";
                int mlpLayer = layerNum switch { "0" => 0, "1" => 1, "2" => 2, _ => -1 };
                if (mlpLayer < 0) return null;
                if (suffix == "weight") return $"transformer_decoder.box_head.{mlpLayer}.weight";
                if (suffix == "bias") return $"transformer_decoder.box_head.{mlpLayer}.bias";
            }
            if (decRest.StartsWith("presence_head.")) return MapPresenceHeadKey($"detr_decoder.{decRest}");

            // HF: boxRPB_embed_x/y (camelCase) -> box_rpb_embed_x/y (snake_case)
            if (decRest.Contains("boxRPB_embed_x.") || decRest.Contains("boxRPB_embed_y."))
            {
                var isX = decRest.Contains("boxRPB_embed_x");
                return MapBoxRPBKey($"detr_decoder.{decRest}", isX ? "box_rpb_embed_x" : "box_rpb_embed_y");
            }
            if (decRest.Contains("box_rpb_embed_x.") || decRest.Contains("box_rpb_embed_y."))
            {
                var isX = decRest.Contains("box_rpb_embed_x");
                return MapBoxRPBKey($"detr_decoder.{decRest}", isX ? "box_rpb_embed_x" : "box_rpb_embed_y");
            }
            return null;
        }

        if (rest.StartsWith("geometry_encoder."))
        {
            var geoRest = rest.Substring("geometry_encoder.".Length);
            if (geoRest.StartsWith("cls_embed.") || geoRest.StartsWith("boxes_direct_project.") ||
                geoRest.StartsWith("boxes_pool_project.") || geoRest.StartsWith("boxes_pos_enc_project.") ||
                geoRest.StartsWith("output_layer_norm.") || geoRest.StartsWith("prompt_layer_norm.") ||
                geoRest.StartsWith("vision_layer_norm.")) return null;
            return MapGeometryEncoderKey($"geometry_encoder.{geoRest}");
        }

        if (rest.StartsWith("mask_decoder."))
        {
            if (rest.Contains("semantic_projection") || rest.Contains("instance_projection")) return null;
            return MapMaskDecoderKey($"mask_decoder.{rest}");
        }

        if (rest.StartsWith("dot_prod_scoring."))
            return MapDotProductScoringKey($"dot_product_scoring.{rest}");

        return null;
    }

    private string MapHfTextKey(string langRest)
    {
        if (langRest == "ln_final.weight") return "text_encoder.transformer.final_layer_norm.weight";
        if (langRest == "ln_final.bias") return "text_encoder.transformer.final_layer_norm.bias";
        if (langRest == "positional_embedding") return "text_encoder.transformer.pos_embed_buffer";
        if (langRest == "token_embedding.weight") return "text_encoder.transformer.token_embedding.weight";
        if (langRest.StartsWith("layers."))
        {
            var layerRest = langRest.Substring("layers.".Length);
            var parts = layerRest.Split('.');
            var layerIdx = parts[0];
            var suffix = string.Join(".", parts.Skip(1));
            if (suffix == "ln_1.weight") return $"text_encoder.transformer.encoder_layer_{layerIdx}.layer_norm1.weight";
            if (suffix == "ln_1.bias") return $"text_encoder.transformer.encoder_layer_{layerIdx}.layer_norm1.bias";
            if (suffix == "ln_2.weight") return $"text_encoder.transformer.encoder_layer_{layerIdx}.layer_norm2.weight";
            if (suffix == "ln_2.bias") return $"text_encoder.transformer.encoder_layer_{layerIdx}.layer_norm2.bias";
            if (suffix.StartsWith("self_attn."))
            {
                var attnSuffix = suffix.Substring("self_attn.".Length);
                if (attnSuffix == "in_proj_weight") return $"text_encoder.transformer.encoder_layer_{layerIdx}.self_attn_in_proj.weight";
                if (attnSuffix == "in_proj_bias") return $"text_encoder.transformer.encoder_layer_{layerIdx}.self_attn_in_proj.bias";
                if (attnSuffix == "out_proj.weight") return $"text_encoder.transformer.encoder_layer_{layerIdx}.self_attn_o_proj.weight";
                if (attnSuffix == "out_proj.bias") return $"text_encoder.transformer.encoder_layer_{layerIdx}.self_attn_o_proj.bias";
            }
            if (suffix == "mlp.c_fc.weight") return $"text_encoder.transformer.encoder_layer_{layerIdx}.fc1.weight";
            if (suffix == "mlp.c_fc.bias") return $"text_encoder.transformer.encoder_layer_{layerIdx}.fc1.bias";
            if (suffix == "mlp.c_proj.weight") return $"text_encoder.transformer.encoder_layer_{layerIdx}.fc2.weight";
            if (suffix == "mlp.c_proj.bias") return $"text_encoder.transformer.encoder_layer_{layerIdx}.fc2.bias";
        }
        return null;
    }

    private string MapViTLayerKey(string rest)
    {
        var parts = rest.Split('.');
        var layerIdx = parts[1];
        var suffix = string.Join(".", parts.Skip(2));
        if (suffix == "layer_norm1.weight") return $"vision_backbone.block_{layerIdx}.norm1.weight";
        if (suffix == "layer_norm1.bias") return $"vision_backbone.block_{layerIdx}.norm1.bias";
        if (suffix == "layer_norm2.weight") return $"vision_backbone.block_{layerIdx}.norm2.weight";
        if (suffix == "layer_norm2.bias") return $"vision_backbone.block_{layerIdx}.norm2.bias";
        if (suffix.StartsWith("attention.")) suffix = suffix.Substring("attention.".Length);
        else if (suffix.StartsWith("mlp.")) suffix = suffix.Substring("mlp.".Length);
        return $"vision_backbone.block_{layerIdx}.{suffix}";
    }

    private string MapTextEncoderLayerKey(string rest)
    {
        var parts = rest.Split('.');
        var layerIdx = parts[2];
        var suffix = string.Join(".", parts.Skip(3));
        if (suffix.StartsWith("self_attn."))
        {
            var sub = suffix.Substring("self_attn.".Length);
            if (sub == "out_proj.weight") return $"text_encoder.transformer.encoder_layer_{layerIdx}.self_attn_o_proj.weight";
            if (sub == "out_proj.bias") return $"text_encoder.transformer.encoder_layer_{layerIdx}.self_attn_o_proj.bias";
            return $"text_encoder.transformer.encoder_layer_{layerIdx}.self_attn_{sub}";
        }
        if (suffix == "layer_norm1.weight") return $"text_encoder.transformer.encoder_layer_{layerIdx}.layer_norm1.weight";
        if (suffix == "layer_norm1.bias") return $"text_encoder.transformer.encoder_layer_{layerIdx}.layer_norm1.bias";
        if (suffix == "layer_norm2.weight") return $"text_encoder.transformer.encoder_layer_{layerIdx}.layer_norm2.weight";
        if (suffix == "layer_norm2.bias") return $"text_encoder.transformer.encoder_layer_{layerIdx}.layer_norm2.bias";
        if (suffix == "mlp.fc1.weight") return $"text_encoder.transformer.encoder_layer_{layerIdx}.fc1.weight";
        if (suffix == "mlp.fc1.bias") return $"text_encoder.transformer.encoder_layer_{layerIdx}.fc1.bias";
        if (suffix == "mlp.fc2.weight") return $"text_encoder.transformer.encoder_layer_{layerIdx}.fc2.weight";
        if (suffix == "mlp.fc2.bias") return $"text_encoder.transformer.encoder_layer_{layerIdx}.fc2.bias";
        return null;
    }

    private string MapTransformerEncoderLayerKey(string ckptKey)
    {
        var rest = ckptKey.Substring("detr_encoder.layers.".Length);
        var parts = rest.Split('.');
        var layerIdx = parts[0];
        var suffix = string.Join(".", parts.Skip(1));
        if (suffix.StartsWith("self_attn."))
        {
            var sub = suffix.Substring("self_attn.".Length);
            return $"transformer_encoder.layer_{layerIdx}.self_attn_{sub}";
        }
        if (suffix.StartsWith("cross_attn."))
        {
            var sub = suffix.Substring("cross_attn.".Length);
            return $"transformer_encoder.layer_{layerIdx}.cross_attn_{sub}";
        }
        if (suffix == "layer_norm1.weight") return $"transformer_encoder.layer_{layerIdx}.norm1.weight";
        if (suffix == "layer_norm1.bias") return $"transformer_encoder.layer_{layerIdx}.norm1.bias";
        if (suffix == "layer_norm2.weight") return $"transformer_encoder.layer_{layerIdx}.norm2.weight";
        if (suffix == "layer_norm2.bias") return $"transformer_encoder.layer_{layerIdx}.norm2.bias";
        if (suffix == "layer_norm3.weight") return $"transformer_encoder.layer_{layerIdx}.norm3.weight";
        if (suffix == "layer_norm3.bias") return $"transformer_encoder.layer_{layerIdx}.norm3.bias";
        if (suffix == "mlp.fc1.weight") return $"transformer_encoder.layer_{layerIdx}.linear1.weight";
        if (suffix == "mlp.fc1.bias") return $"transformer_encoder.layer_{layerIdx}.linear1.bias";
        if (suffix == "mlp.fc2.weight") return $"transformer_encoder.layer_{layerIdx}.linear2.weight";
        if (suffix == "mlp.fc2.bias") return $"transformer_encoder.layer_{layerIdx}.linear2.bias";
        return null;
    }

    private string MapTransformerDecoderLayerKey(string ckptKey)
    {
        var rest = ckptKey.Substring("detr_decoder.layers.".Length);
        var parts = rest.Split('.');
        var layerIdx = parts[0];
        var suffix = string.Join(".", parts.Skip(1));
        if (suffix.StartsWith("self_attn."))
        {
            var sub = suffix.Substring("self_attn.".Length);
            return $"transformer_decoder.layer_{layerIdx}.self_attn_{sub}";
        }
        if (suffix.StartsWith("vision_cross_attn."))
        {
            var sub = suffix.Substring("vision_cross_attn.".Length);
            return $"transformer_decoder.layer_{layerIdx}.cross_attn_{sub}";
        }
        if (suffix.StartsWith("text_cross_attn."))
        {
            var sub = suffix.Substring("text_cross_attn.".Length);
            return $"transformer_decoder.layer_{layerIdx}.ca_text_{sub}";
        }
        if (suffix == "vision_cross_attn_layer_norm.weight") return $"transformer_decoder.layer_{layerIdx}.layer_norm1.weight";
        if (suffix == "vision_cross_attn_layer_norm.bias") return $"transformer_decoder.layer_{layerIdx}.layer_norm1.bias";
        if (suffix == "self_attn_layer_norm.weight") return $"transformer_decoder.layer_{layerIdx}.layer_norm2.weight";
        if (suffix == "self_attn_layer_norm.bias") return $"transformer_decoder.layer_{layerIdx}.layer_norm2.bias";
        if (suffix == "mlp_layer_norm.weight") return $"transformer_decoder.layer_{layerIdx}.layer_norm3.weight";
        if (suffix == "mlp_layer_norm.bias") return $"transformer_decoder.layer_{layerIdx}.layer_norm3.bias";
        if (suffix == "text_cross_attn_layer_norm.weight") return $"transformer_decoder.layer_{layerIdx}.catext_norm.weight";
        if (suffix == "text_cross_attn_layer_norm.bias") return $"transformer_decoder.layer_{layerIdx}.catext_norm.bias";
        if (suffix == "mlp.fc1.weight") return $"transformer_decoder.layer_{layerIdx}.mlp_fc1.weight";
        if (suffix == "mlp.fc1.bias") return $"transformer_decoder.layer_{layerIdx}.mlp_fc1.bias";
        if (suffix == "mlp.fc2.weight") return $"transformer_decoder.layer_{layerIdx}.mlp_fc2.weight";
        if (suffix == "mlp.fc2.bias") return $"transformer_decoder.layer_{layerIdx}.mlp_fc2.bias";
        return null;
    }

    private string MapBoxHeadKey(string ckptKey)
    {
        var rest = ckptKey.Substring("detr_decoder.box_head.".Length);
        var parts = rest.Split('.');
        var layerNum = parts[0];
        var param = parts.Length > 1 ? parts[1] : "";
        int mlpLayer = layerNum switch { "layer1" => 0, "layer2" => 1, "layer3" => 2, _ => -1 };
        if (mlpLayer < 0) return null;
        if (param == "weight") return $"transformer_decoder.box_head.{mlpLayer}.weight";
        if (param == "bias") return $"transformer_decoder.box_head.{mlpLayer}.bias";
        return null;
    }

    private string MapPresenceHeadKey(string ckptKey)
    {
        var rest = ckptKey.Substring("detr_decoder.presence_head.".Length);
        var parts = rest.Split('.');
        var layerNum = parts[0];
        var param = parts.Length > 1 ? parts[1] : "";
        int mlpLayer = layerNum switch { "layer1" => 0, "layer2" => 1, "layer3" => 2, _ => -1 };
        if (mlpLayer < 0) return null;
        if (param == "weight") return $"transformer_decoder.presence_head.{mlpLayer}.weight";
        if (param == "bias") return $"transformer_decoder.presence_head.{mlpLayer}.bias";
        return null;
    }

    private string MapRefPointHeadKey(string ckptKey)
    {
        var rest = ckptKey.Substring("detr_decoder.ref_point_head.".Length);
        var parts = rest.Split('.');
        var layerNum = parts[0];
        var param = parts.Length > 1 ? parts[1] : "";
        int mlpLayer = layerNum switch { "layer1" => 0, "layer2" => 1, _ => -1 };
        if (mlpLayer < 0) return null;
        if (param == "weight") return $"transformer_decoder.ref_point_head.{mlpLayer}.weight";
        if (param == "bias") return $"transformer_decoder.ref_point_head.{mlpLayer}.bias";
        return null;
    }

    private string MapBoxRPBKey(string ckptKey, string name)
    {
        var rest = ckptKey.Substring($"detr_decoder.{name}.".Length);
        var parts = rest.Split('.');
        var layerNum = parts[0];
        var param = parts.Length > 1 ? parts[1] : "";
        int mlpLayer = layerNum switch { "layer1" => 0, "layer2" => 1, _ => -1 };
        if (mlpLayer < 0) return null;
        if (param == "weight") return $"transformer_decoder.{name}.{mlpLayer}.weight";
        if (param == "bias") return $"transformer_decoder.{name}.{mlpLayer}.bias";
        return null;
    }

    private string MapGeometryEncoderKey(string ckptKey)
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
            if (suffix == "layer_norm1.weight") return $"geometry_encoder.geo_layer_{layerIdx}.layer_norm1.weight";
            if (suffix == "layer_norm1.bias") return $"geometry_encoder.geo_layer_{layerIdx}.layer_norm1.bias";
            if (suffix == "layer_norm2.weight") return $"geometry_encoder.geo_layer_{layerIdx}.layer_norm2.weight";
            if (suffix == "layer_norm2.bias") return $"geometry_encoder.geo_layer_{layerIdx}.layer_norm2.bias";
            if (suffix == "layer_norm3.weight") return $"geometry_encoder.geo_layer_{layerIdx}.layer_norm3.weight";
            if (suffix == "layer_norm3.bias") return $"geometry_encoder.geo_layer_{layerIdx}.layer_norm3.bias";
            if (suffix == "mlp.fc1.weight") return $"geometry_encoder.geo_layer_{layerIdx}.mlp_fc1.weight";
            if (suffix == "mlp.fc1.bias") return $"geometry_encoder.geo_layer_{layerIdx}.mlp_fc1.bias";
            if (suffix == "mlp.fc2.weight") return $"geometry_encoder.geo_layer_{layerIdx}.mlp_fc2.weight";
            if (suffix == "mlp.fc2.bias") return $"geometry_encoder.geo_layer_{layerIdx}.mlp_fc2.bias";
        }
        if (rest == "label_embed.weight") return "geometry_encoder.label_embed.weight";
        if (rest == "final_proj.weight") return "geometry_encoder.final_proj.weight";
        if (rest == "final_proj.bias") return "geometry_encoder.final_proj.bias";
        return $"geometry_encoder.{rest}";
    }

    private string MapMaskDecoderKey(string ckptKey)
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
            return $"mask_decoder.prompt_cross_attn.{rest.Substring("prompt_cross_attn.".Length)}";
        if (rest.StartsWith("prompt_cross_attn_norm."))
            return $"mask_decoder.prompt_cross_attn_norm.{rest.Substring("prompt_cross_attn_norm.".Length)}";
        if (rest == "semantic_projection.weight") return "mask_decoder.semantic_projection.weight";
        if (rest == "semantic_projection.bias") return "mask_decoder.semantic_projection.bias";
        if (rest == "instance_projection.weight") return "mask_decoder.instance_projection.weight";
        if (rest == "instance_projection.bias") return "mask_decoder.instance_projection.bias";
        return $"mask_decoder.{rest}";
    }

    private string MapDotProductScoringKey(string ckptKey)
    {
        var rest = ckptKey.Substring("dot_product_scoring.".Length);
        if (rest.StartsWith("text_mlp.layer"))
        {
            var sub = rest.Replace("text_mlp.layer1", "text_mlp_layer1").Replace("text_mlp.layer2", "text_mlp_layer2");
            return $"dot_product_scoring.{sub}";
        }
        if (rest == "text_mlp_out_norm.weight") return "dot_product_scoring.text_mlp_out_norm.weight";
        if (rest == "text_mlp_out_norm.bias") return "dot_product_scoring.text_mlp_out_norm.bias";
        if (rest == "text_proj.weight") return "dot_product_scoring.text_proj.weight";
        if (rest == "text_proj.bias") return "dot_product_scoring.text_proj.bias";
        if (rest == "query_proj.weight") return "dot_product_scoring.query_proj.weight";
        if (rest == "query_proj.bias") return "dot_product_scoring.query_proj.bias";
        return $"dot_product_scoring.{rest}";
    }

    private bool TrySetValueByPath(Module module, string path, Tensor value)
    {
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
            else { moduleName = part; }

            Module child = null;
            var children = current.named_children().ToList();
            var matchingChildren = children.Where(kvp => kvp.Item1 == moduleName).ToList();
            if (matchingChildren.Count == 1) { child = matchingChildren[0].Item2; }
            else if (matchingChildren.Count > 1 && index >= 0 && index < matchingChildren.Count) { child = matchingChildren[index].Item2; }
            else if (matchingChildren.Count == 0)
            {
                var fields = current.GetType().GetFields(System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic);
                foreach (var field in fields)
                {
                    if (field.Name == moduleName)
                    {
                        var fieldValue = field.GetValue(current);
                        if (fieldValue is IList<Module> ml && index >= 0 && index < ml.Count) { child = ml[index]; break; }
                        else if (fieldValue is Module sm) { child = sm; break; }
                    }
                }
                if (child == null)
                {
                    var props = current.GetType().GetProperties(System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic);
                    foreach (var prop in props)
                    {
                        if (prop.Name == moduleName)
                        {
                            var pv = prop.GetValue(current);
                            if (pv is IList<Module> pm && index >= 0 && index < pm.Count) { child = pm[index]; break; }
                            else if (pv is Module pm2) { child = pm2; break; }
                        }
                    }
                }
            }
            if (child == null) return false;
            current = child;
        }

        var finalKey = parts[parts.Length - 1];

        // Try field
        Tensor? fieldTensor = null;
        var fieldType = current.GetType();
        var fieldMember = fieldType.GetField(finalKey, System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic);
        if (fieldMember is not null && fieldMember.FieldType == typeof(Tensor))
        {
            fieldTensor = (Tensor)fieldMember.GetValue(current);
        }
        if (fieldTensor is not null && fieldTensor.shape.Length == value.shape.Length)
        {
            bool match = true;
            for (int i = 0; i < fieldTensor.shape.Length; i++)
            {
                if (fieldTensor.shape[i] != value.shape[i]) { match = false; break; }
            }
            if (match)
            {
                using (var src = value.contiguous()) using (var _ = torch.no_grad()) { fieldTensor.copy_(src); }
                return true;
            }
            return false;
        }

        // Try buffer
        Tensor? buffer = null;
        try { buffer = current.get_buffer(finalKey); } catch { }
        if (buffer is not null && buffer.shape.Length == value.shape.Length)
        {
            bool match = true;
            for (int i = 0; i < buffer.shape.Length; i++)
            {
                if (buffer.shape[i] != value.shape[i]) { match = false; break; }
            }
            if (match)
            {
                using (var src = value.contiguous()) using (var _ = torch.no_grad()) { buffer.copy_(src); }
                return true;
            }
            return false;
        }

        // Try parameter
        Parameter? param = null;
        foreach (var kvp in current.named_parameters(false))
        {
            if (kvp.Item1 == finalKey) { param = kvp.Item2; break; }
        }
        if (param is null) return false;
        if (param.shape.Length != value.shape.Length) return false;
        for (int i = 0; i < param.shape.Length; i++)
        {
            if (param.shape[i] != value.shape[i]) return false;
        }
        using (var src = value.contiguous()) using (var _ = torch.no_grad()) { param.copy_(src); }
        return true;
    }
}
