// Copyright (c) Sapiens AI. All rights reserved.

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System;
using System.Collections.Generic;

namespace SAMTorchSharp.Modeling.Sam3;

public class Sam3ViTFpnNeck : Module
{
    private readonly List<FpnLevelNew> fpn_layers;
    private readonly int d_model;
    private readonly int num_levels;

    public Sam3ViTFpnNeck(int inChannels = 1024, int d_model = 256, int num_levels = 4)
        : base(nameof(Sam3ViTFpnNeck))
    {
        this.d_model = d_model;
        this.num_levels = num_levels;
        fpn_layers = new List<FpnLevelNew>();
        for (int i = 0; i < num_levels; i++)
        {
            fpn_layers.Add(new FpnLevelNew(inChannels, d_model, i));
            register_module("fpn_layer_" + i.ToString(), fpn_layers[i]);
        }
        RegisterComponents();
    }

    public List<Tensor> forward(Tensor input_features)
    {
        var results = new List<Tensor>();
        for (int i = 0; i < fpn_layers.Count; i++)
        {
            var feat = fpn_layers[i].forward(input_features);
            results.Add(feat);
        }
        return results;
    }

    public int NumLevels => num_levels;
}

public class FpnLevelNew : Module
{
    private readonly ConvTranspose2d deconv1;
    private readonly ConvTranspose2d? deconv2;
    private readonly MaxPool2d? mp;
    private readonly Conv2d proj1;
    private readonly Conv2d proj2;
    private readonly int lvl;

    public FpnLevelNew(int in_channels, int out_channels, int level)
        : base(nameof(FpnLevelNew))
    {
        this.lvl = level;

        if (level == 0)
        {
            // Scale 4.0: double transposed conv (upsample 4x): 72->144->288
            // 1024 -> 512 -> 256
            deconv1 = ConvTranspose2d(in_channels, in_channels / 2, kernelSize: 2, stride: 2);
            deconv2 = ConvTranspose2d(in_channels / 2, out_channels, kernelSize: 2, stride: 2);
            // After deconv: [B, 256, 288, 288]
            proj1 = Conv2d(out_channels, out_channels, kernelSize: 1);
            proj2 = Conv2d(out_channels, out_channels, kernelSize: 3, padding: 1);
        }
        else if (level == 1)
        {
            // Scale 2.0: single transposed conv (upsample 2x): 72->144
            // 1024 -> 512
            deconv1 = ConvTranspose2d(in_channels, in_channels / 2, kernelSize: 2, stride: 2);
            deconv2 = null;
            // After deconv: [B, 512, 144, 144]
            proj1 = Conv2d(in_channels / 2, out_channels, kernelSize: 1);
            proj2 = Conv2d(out_channels, out_channels, kernelSize: 3, padding: 1);
        }
        else if (level == 2)
        {
            // Scale 1.0: identity: 72->72
            // 1024 -> 1024
            deconv1 = null!;
            deconv2 = null;
            mp = null!;
            proj1 = Conv2d(in_channels, out_channels, kernelSize: 1);
            proj2 = Conv2d(out_channels, out_channels, kernelSize: 3, padding: 1);
        }
        else
        {
            // Scale 0.5: maxpool (downsample 2x): 72->36
            // 1024 -> 1024
            mp = MaxPool2d(2, 2);
            deconv1 = null!;
            deconv2 = null;
            proj1 = Conv2d(in_channels, out_channels, kernelSize: 1);
            proj2 = Conv2d(out_channels, out_channels, kernelSize: 3, padding: 1);
        }

        RegisterComponents();
    }

    public Tensor forward(Tensor x)
    {
        Tensor scaled;
        if (lvl == 2)
        {
            scaled = x;
        }
        else if (lvl == 3)
        {
            scaled = mp.forward(x);
        }
        else if (deconv2 is not null)
        {
            scaled = deconv1.forward(x);
            scaled = deconv2.forward(scaled);
        }
        else
        {
            scaled = deconv1.forward(x);
        }
        return proj2.forward(functional.gelu(proj1.forward(scaled)));
    }
}
