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
/// </summary>
public class Sam3DualViTDetNeck : Module
{
    private readonly Sam3ViTDetBackbone trunk;
    private readonly Sam3PositionEmbeddingSine position_encoding;
    private readonly List<Module> convs;
    private readonly float[] scale_factors;
    private readonly int d_model;
    private readonly bool add_sam2_neck;

    public Sam3DualViTDetNeck(
        Sam3ViTDetBackbone trunk,
        Sam3PositionEmbeddingSine position_encoding,
        int d_model,
        float[]? scale_factors = null,
        bool add_sam2_neck = false)
        : base(nameof(Sam3DualViTDetNeck))
    {
        this.trunk = trunk;
        this.position_encoding = position_encoding;
        this.d_model = d_model;
        this.scale_factors = scale_factors ?? new float[] { 4.0f, 2.0f, 1.0f, 0.5f };
        this.add_sam2_neck = add_sam2_neck;
        convs = new List<Module>();

        var dim = trunk.GetChannelList()[trunk.GetChannelList().Length - 1];

        for (int i = 0; i < scale_factors.Length; i++)
        {
            var scale = scale_factors[i];
            var mod = build_single_conv(scale, dim);
            convs.Add(mod);
        }
    }

    private Module build_single_conv(float scale, int dim)
    {
        Module<Tensor, Tensor> mod = Identity();

        if (scale == 4.0f)
        {
            var ct2_0 = ConvTranspose2d(dim, dim / 2, kernelSize: 2, stride: 2);
            var gelu = GELU();
            var ct2_1 = ConvTranspose2d(dim / 2, dim / 4, kernelSize: 2, stride: 2);
            mod = Sequential(ct2_0, gelu, ct2_1);
        }
        else if (scale == 2.0f)
        {
            var ct2 = ConvTranspose2d(dim, dim / 2, kernelSize: 2, stride: 2);
            mod = Sequential(ct2);
        }
        else if (scale == 0.5f)
        {
            var mp = MaxPool2d(kernelSize: 2, stride: 2);
            mod = Sequential(mp);
        }

        int out_dim;
        if (scale == 4.0f) out_dim = dim / 4;
        else if (scale == 2.0f) out_dim = dim / 2;
        else out_dim = (int)dim;

        var conv1x1 = Conv2d(out_dim, d_model, kernelSize: 1);
        var conv3x3 = Conv2d(d_model, d_model, kernelSize: 3, padding: 1);

        mod = Sequential(mod, conv1x1, conv3x3);

        return mod;
    }

    public Tuple<List<Tensor>, List<Tensor>> forward(Tensor x)
    {
        var backboneOut = trunk.forward(x);

        var sam3_out = new List<Tensor>();
        var sam3_pos = new List<Tensor>();

        for (int i = 0; i < convs.Count; i++)
        {
            var feat = ((Module<Tensor, Tensor>)convs[i]).forward(backboneOut);
            var pos = position_encoding.forward(feat);
            sam3_out.Add(feat);
            sam3_pos.Add(pos);
        }

        return Tuple.Create(sam3_out, sam3_pos);
    }

    public int GetNumLevels() => convs.Count;
}
