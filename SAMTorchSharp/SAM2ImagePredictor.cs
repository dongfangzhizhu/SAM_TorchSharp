using SAMTorchSharp.Modeling.Sam2;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using System.Collections.Generic;

namespace SAMTorchSharp
{
    /// <summary>
    /// Transforms input images to/from model-expected format.
    /// Corresponds to sam2/utils/transforms.py: SAM2Transforms.
    /// </summary>
    public class SAM2Transforms
    {
        private readonly long _resolution;
        private readonly double _maskThreshold;
        private readonly double _maxHoleArea;
        private readonly double _maxSprinkleArea;

        public SAM2Transforms(long resolution, double maskThreshold = 0.0,
            double maxHoleArea = 0.0, double maxSprinkleArea = 0.0)
        {
            _resolution = resolution;
            _maskThreshold = maskThreshold;
            _maxHoleArea = maxHoleArea;
            _maxSprinkleArea = maxSprinkleArea;
        }

        /// <summary>
        /// Transform an np.ndarray image (HWC, uint8, RGB) to a Tensor [3, resolution, resolution].
        /// </summary>
        public Tensor __call(Tensor image)
        {
            if (image.dtype == ScalarType.Byte)
                image = image.to(ScalarType.Float32) / 255.0f;

            if (image.dim() == 3 && image.size(0) != 3)
            {
                // HWC to CHW
                image = image.permute(new long[] { 2, 0, 1 });
            }

            image = (image - 0.5) / 0.5;

            var h = image.size(1);
            var w = image.size(2);
            if (h != _resolution || w != _resolution)
            {
                image = interpolate(image.unsqueeze(0),
                    size: new long[] { _resolution, _resolution },
                    mode: InterpolationMode.Bilinear, align_corners: false);
                image = image.squeeze(0);
            }

            return image;
        }
    }

    /// <summary>
    /// SAM2 Image Predictor - for single-image mask generation with prompts.
    /// Corresponds to sam2/sam2_image_predictor.py: SAM2ImagePredictor.
    /// Used by SAM2AutomaticMaskGenerator internally.
    /// </summary>
    public class SAM2ImagePredictor : IDisposable
    {
        private readonly Sam2Base _model;
        private readonly SAM2Transforms _transforms;
        private bool _isImageSet = false;
        private Tensor? _imageEmbedding;
        private IList<Tensor>? _highResFeatures;
        private long[]? _origHw;
        private Device _device;

        public float MaskThreshold { get; set; }
        public double MaxHoleArea { get; set; }
        public double MaxSprinkleArea { get; set; }

        public SAM2ImagePredictor(Sam2Base model,
            double maskThreshold = 0.0,
            double maxHoleArea = 0.0,
            double maxSprinkleArea = 0.0)
        {
            _model = model;
            _device = _model.parameters().FirstOrDefault()?.device ?? CPU;
            _transforms = new SAM2Transforms(
                _model.image_size,
                maskThreshold,
                maxHoleArea,
                maxSprinkleArea);
            MaskThreshold = (float)maskThreshold;
            MaxHoleArea = maxHoleArea;
            MaxSprinkleArea = maxSprinkleArea;
        }

        public Device Device => _device;
        public SAM2Transforms Transforms => _transforms;

        /// <summary>
        /// Set an image for prediction. Computes image embedding.
        /// </summary>
        public void SetImage(Tensor image)
        {
            ResetPredictor();

            long h, w;
            if (image.dim() == 3 && image.size(0) == 3)
            {
                // CHW format
                h = image.size(1);
                w = image.size(2);
            }
            else if (image.dim() == 3)
            {
                // HWC format
                h = image.size(0);
                w = image.size(1);
            }
            else
            {
                throw new ArgumentException("Image must be 3D tensor [H,W,3] or [3,H,W]");
            }
            _origHw = new long[] { h, w };

            // Ensure HWC format for transform
            Tensor inputImage;
            if (image.size(0) == 3)
            {
                inputImage = image.permute(new long[] { 1, 2, 0 });
            }
            else
            {
                inputImage = image;
            }

            inputImage = _transforms.__call(inputImage);
            inputImage = inputImage.unsqueeze(0).to(_device);

            var backboneOut = _model.ForwardImage(inputImage);
            var (visionFeats, _, _) = _model.PrepareBackboneFeatures(backboneOut);

            if (_model.directly_add_no_mem_embed)
            {
                visionFeats[visionFeats.Count - 1] = visionFeats[visionFeats.Count - 1] + _model.no_mem_embed;
            }

            var bbFeatSizes = new[] { (256L, 256L), (128L, 128L), (64L, 64L) };
            var feats = new List<Tensor>();
            for (int i = visionFeats.Count - 1; i >= 0; i--)
            {
                var feat = visionFeats[i];
                var (fh, fw) = bbFeatSizes[Math.Min(i, bbFeatSizes.Length - 1)];
                feat = feat.permute(new long[] { 1, 2, 0 }).view(new long[] { 1, -1, fh, fw });
                feats.Add(feat);
            }

            _imageEmbedding = feats[feats.Count - 1];
            _highResFeatures = feats.Take(feats.Count - 1).ToList();
            _isImageSet = true;
        }

        /// <summary>
        /// Predict masks from point/box/mask prompts.
        /// </summary>
        public (Tensor Masks, Tensor IouPredictions, Tensor LowResMasks) Predict(
            Tensor? pointCoords = null,
            Tensor? pointLabels = null,
            Tensor? box = null,
            Tensor? maskInput = null,
            bool multimaskOutput = true,
            bool returnLogits = false)
        {
            if (!_isImageSet)
                throw new InvalidOperationException("An image must be set with SetImage before prediction.");

            Tensor? unnormCoords = null;
            Tensor? labels = null;
            Tensor? unnormBox = null;

            if (pointCoords is not null)
            {
                if (pointLabels is null)
                    throw new ArgumentException("point_labels must be supplied if point_coords is supplied.");

                unnormCoords = pointCoords.to(_device);
                labels = pointLabels.to(_device).to(ScalarType.Int32);

                if (unnormCoords.dim() == 2) unnormCoords = unnormCoords.unsqueeze(0);
                if (labels.dim() == 1) labels = labels.unsqueeze(0);
            }

            if (box is not null)
            {
                unnormBox = box.to(_device).reshape(1, 2, 2);
                var boxLabels = tensor(new long[] { 2, 3 }, dtype: ScalarType.Int32, device: _device).reshape(1, 2);
                if (unnormCoords is not null)
                {
                    unnormCoords = cat(new[] { unnormBox, unnormCoords }, dim: 1);
                    labels = cat(new[] { boxLabels, labels! }, dim: 1);
                }
                else
                {
                    unnormCoords = unnormBox;
                    labels = boxLabels;
                }
            }

            if (maskInput is not null)
            {
                if (maskInput.dim() == 3) maskInput = maskInput.unsqueeze(0);
                maskInput = maskInput.to(_device);
            }

            Tuple<Tensor, Tensor>? concatPoints = null;
            if (unnormCoords is not null)
            {
                concatPoints = Tuple.Create(unnormCoords, labels);
            }

            var (sparseEmbeddings, denseEmbeddings) = _model.sam_prompt_encoder.forward(
                concatPoints, null, maskInput);

            bool batchedMode = concatPoints is not null && concatPoints.Item1.size(0) > 1;

            var highResFeatures = _highResFeatures?.Select(f => f.unsqueeze(0)).ToList() ?? new List<Tensor>();

            var (lowResMultimasks, ious, _, _) = _model.sam_mask_decoder.forward(
                _imageEmbedding!.unsqueeze(0),
                _model.sam_prompt_encoder.get_dense_pe(),
                sparseEmbeddings,
                denseEmbeddings,
                multimaskOutput,
                batchedMode,
                highResFeatures);

            Tensor masks;
            if (returnLogits)
            {
                masks = lowResMultimasks;
            }
            else
            {
                masks = (lowResMultimasks > MaskThreshold).to(ScalarType.Float32);
            }

            Tensor lowResMasksOut = clamp(lowResMultimasks, -32.0, 32.0);

            return (masks, ious, lowResMasksOut);
        }

        public void ResetPredictor()
        {
            _isImageSet = false;
            _imageEmbedding = null;
            _highResFeatures = null;
            _origHw = null;
        }

        public void Dispose()
        {
            _model.Dispose();
        }
    }
}
