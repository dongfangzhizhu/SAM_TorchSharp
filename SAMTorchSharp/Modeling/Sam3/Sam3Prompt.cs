// Copyright (c) Sapiens AI. All rights reserved.

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using System;
using System.Collections.Generic;

namespace SAMTorchSharp.Modeling.Sam3;

/// <summary>
/// C# equivalent of Python geometry_encoders.py Prompt class.
/// Utility class to manipulate geometric prompts (boxes, points, masks).
/// Sequences are in pytorch convention: sequence first, batch second.
/// </summary>
public class Sam3Prompt
{
    /// <summary>
    /// Box embeddings: [N_boxes, B, C_box]
    /// </summary>
    public Tensor? box_embeddings { get; set; }

    /// <summary>
    /// Box labels: [N_boxes, B] (long tensor). null means all positive.
    /// </summary>
    public Tensor? box_labels { get; set; }

    /// <summary>
    /// Box attention mask: [B, N_boxes]. False = valid, True = padded.
    /// </summary>
    public Tensor? box_mask { get; set; }

    /// <summary>
    /// Point embeddings: [N_points, B, C_point]
    /// </summary>
    public Tensor? point_embeddings { get; set; }

    /// <summary>
    /// Point labels: [N_points, B] (long tensor). 0=positive, 1=negative.
    /// </summary>
    public Tensor? point_labels { get; set; }

    /// <summary>
    /// Point attention mask: [B, N_points].
    /// </summary>
    public Tensor? point_mask { get; set; }

    /// <summary>
    /// Mask embeddings: [N_masks, B, 1, H_mask, W_mask]
    /// </summary>
    public Tensor? mask_embeddings { get; set; }

    /// <summary>
    /// Mask attention mask: [B, N_masks].
    /// </summary>
    public Tensor? mask_mask { get; set; }

    /// <summary>
    /// Mask labels: [N_masks, B].
    /// </summary>
    public Tensor? mask_labels { get; set; }

    public Sam3Prompt(
        Tensor? box_embeddings = null,
        Tensor? box_mask = null,
        Tensor? point_embeddings = null,
        Tensor? point_mask = null,
        Tensor? box_labels = null,
        Tensor? point_labels = null,
        Tensor? mask_embeddings = null,
        Tensor? mask_mask = null,
        Tensor? mask_labels = null)
    {
        this.box_embeddings = box_embeddings;
        this.box_mask = box_mask;
        this.point_embeddings = point_embeddings;
        this.point_mask = point_mask;
        this.box_labels = box_labels;
        this.point_labels = point_labels;
        this.mask_embeddings = mask_embeddings;
        this.mask_mask = mask_mask;
        this.mask_labels = mask_labels;
    }

    /// <summary>
    /// Convenience: number of boxes (from box_embeddings sequence dim).
    /// </summary>
    public long num_boxes => box_embeddings?.size(0) ?? 0;

    /// <summary>
    /// Convenience: number of points (from point_embeddings sequence dim).
    /// </summary>
    public long num_points => point_embeddings?.size(0) ?? 0;

    /// <summary>
    /// Convenience: point coordinates extracted from point_embeddings tensor shape.
    /// Returns the raw point tensor [N_points, B, C_point] if available.
    /// </summary>
    public Tensor? points => point_embeddings;

    /// <summary>
    /// Convenience: box coordinates extracted from box_embeddings tensor shape.
    /// Returns the raw box tensor [N_boxes, B, C_box] if available.
    /// </summary>
    public Tensor? boxes => box_embeddings;

    /// <summary>
    /// Convenience: mask tensor if available.
    /// </summary>
    public Tensor? masks => mask_embeddings;

    /// <summary>
    /// Check if this is an empty/null prompt.
    /// </summary>
    public bool IsEmpty()
    {
        return box_embeddings is null &&
               point_embeddings is null &&
               mask_embeddings is null;
    }

    /// <summary>
    /// Clone the prompt (shallow copy of tensors).
    /// </summary>
    public Sam3Prompt Clone()
    {
        return new Sam3Prompt(
            box_embeddings?.clone(),
            box_mask?.clone(),
            point_embeddings?.clone(),
            point_mask?.clone(),
            box_labels?.clone(),
            point_labels?.clone(),
            mask_embeddings?.clone(),
            mask_mask?.clone(),
            mask_labels?.clone());
    }
}
