// Copyright (c) Sapiens AI. All rights reserved.

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System;
using System.Collections.Generic;

namespace SAMTorchSharp.Modeling.Sam3;

/// <summary>
/// Simple FPN neck as used in ViTDet for SAM3.
/// Ported from sam3/model/necks.py (Sam3DualViTDetNeck)
/// </summary>
public class Sam3DualViTDetNeck : Module
{
    private readonly Sam3ViTDetBackbone trunk;
    private readonly Sam3PositionEmbeddingSine position_encoding;
    private readonly List<Sequential> convs;
    private readonly float[] scale_factors;
    private readonly int d_model;

    public Sam3DualViTDetNeck(
        Sam3ViTDetBackbone trunk,
        Sam3PositionEmbeddingSine position_encoding,
        int d_model,
        float[]? scale_factors = null)
        : base(nameof(Sam3DualViTDetNeck))
    {
        this.trunk = trunk;
        this.position_encoding = position_encoding;
        this.d_model = d_model;
        this.scale_factors = scale_factors ?? new float[] { 4.0f, 2.0f, 1.0f, 0.5f };
        convs = new List<Sequential>();

        var dim = 1024; // Default trunk channel

        for (int i = 0; i < scale_factors.Length; i++)
        {
            var scale = scale_factors[i];
            var current = Sequential();
            int out_dim;

            if (scale == 4.0f)
            {
                current.Add(ConvTranspose2d(dim, dim / 2, 2, 2));
                current.Add(GELU());
                current.Add(ConvTranspose2d(dim / 2, dim / 4, 2, 2));
                out_dim = dim / 4;
            }
            else if (scale == 2.0f)
            {
                current.Add(ConvTranspose2d(dim, dim / 2, 2, 2));
                out_dim = dim / 2;
            }
            else if (scale == 1.0f)
            {
                out_dim = dim;
            }
            else if (scale == 0.5f)
            {
                current.Add(MaxPool2d(2, 2));
                out_dim = dim;
            }
            else
            {
                throw new NotSupportedException($"scale_factor={scale} is not supported");
            }

            current.Add(Conv2d(out_dim, d_model, 1));
            current.Add(Conv2d(d_model, d_model, 3, padding: 1));
            convs.Add(current);
        }
    }

    public Tuple<List<Tensor>, List<Tensor>> forward(Tensor x)
    {
        var backbone_out = trunk.forward(x);
        // backbone_out is Tensor (last feature)
        var last_feat = (Tensor)backbone_out;

        var sam3_out = new List<Tensor>();
        var sam3_pos = new List<Tensor>();

        for (int i = 0; i < convs.Count; i++)
        {
            var feat = convs[i].forward(last_feat);
            var pos = position_encoding.forward(feat);
            sam3_out.Add(feat);
            sam3_pos.Add(pos);
        }

        return Tuple.Create(sam3_out, sam3_pos);
    }

    public int GetNumLevels() => convs.Count;
}
