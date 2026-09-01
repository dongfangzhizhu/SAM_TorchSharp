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
        /// Transform an RGB image in HWC (preferred) or CHW layout to a float32
        /// Tensor with shape [3, resolution, resolution]. Byte inputs are scaled to [0,1].
        /// </summary>
        public Tensor __call(Tensor image)
        {
            if (image.dim() != 3)
                throw new ArgumentException("image must be a 3D RGB tensor in HWC or CHW layout.", nameof(image));

            if (image.dtype == ScalarType.Byte)
                image = image.to(ScalarType.Float32) / 255.0f;
            else if (image.dtype != ScalarType.Float32)
                image = image.to(ScalarType.Float32);

            // HWC is the documented public contract and therefore wins for the
            // ambiguous [3, W, 3] case. CHW remains supported for compatibility.
            if (image.size(2) == 3)
            {
                image = image.permute(new long[] { 2, 0, 1 });
            }
            else if (image.size(0) != 3)
            {
                throw new ArgumentException("image must have exactly 3 RGB channels in HWC or CHW layout.", nameof(image));
            }

            var h = image.size(1);
            var w = image.size(2);
            if (h != _resolution || w != _resolution)
                image = AntialiasedBilinearResize(image, _resolution, _resolution);

            using var mean = tensor(new[] { 0.485f, 0.456f, 0.406f }, dtype: ScalarType.Float32)
                .reshape(3, 1, 1).to(image.device);
            using var std = tensor(new[] { 0.229f, 0.224f, 0.225f }, dtype: ScalarType.Float32)
                .reshape(3, 1, 1).to(image.device);
            image = (image - mean) / std;

            return image;
        }

        private static Tensor AntialiasedBilinearResize(Tensor image, long outputHeight, long outputWidth)
        {
            var horizontal = ResizeDimension(image, image.size(2), outputWidth, dimension: 2);
            var vertical = ResizeDimension(horizontal, image.size(1), outputHeight, dimension: 1);
            horizontal.Dispose();
            return vertical;
        }

        private static Tensor ResizeDimension(Tensor input, long inputSize, long outputSize, long dimension)
        {
            var scale = (double)inputSize / outputSize;
            var support = Math.Max(scale, 1.0);
            var maxKernelSize = checked((int)Math.Ceiling(support * 2) + 1);
            var indices = new long[checked((int)outputSize * maxKernelSize)];
            var weights = new float[indices.Length];

            for (var outputIndex = 0; outputIndex < outputSize; outputIndex++)
            {
                var center = (outputIndex + 0.5) * scale - 0.5;
                var first = (long)Math.Floor(center - support) + 1;
                var last = (long)Math.Ceiling(center + support);
                double weightSum = 0;
                var offset = checked((int)outputIndex * maxKernelSize);
                var kernelIndex = 0;
                for (var sourceIndex = first; sourceIndex < last; sourceIndex++)
                {
                    if (sourceIndex < 0 || sourceIndex >= inputSize)
                        continue;
                    var weight = Math.Max(0, 1 - Math.Abs(sourceIndex - center) / support);
                    indices[offset + kernelIndex] = sourceIndex;
                    weights[offset + kernelIndex] = (float)weight;
                    weightSum += weight;
                    kernelIndex++;
                }
                for (var weightIndex = 0; weightIndex < maxKernelSize; weightIndex++)
                    weights[offset + weightIndex] = (float)(weights[offset + weightIndex] / weightSum);
            }

            using var indexTensor = tensor(indices, dtype: ScalarType.Int64, device: input.device);
            using var weightTensor = tensor(weights, dtype: ScalarType.Float32, device: input.device);
            using var gathered = input.index_select(dimension, indexTensor);
            if (dimension == 2)
            {
                using var shaped = gathered.reshape(input.size(0), input.size(1), outputSize, maxKernelSize);
                using var shapedWeights = weightTensor.reshape(1, 1, outputSize, maxKernelSize);
                return (shaped * shapedWeights).sum(3);
            }
            else
            {
                using var shaped = gathered.reshape(input.size(0), outputSize, maxKernelSize, input.size(2));
                using var shapedWeights = weightTensor.reshape(1, outputSize, maxKernelSize, 1);
                return (shaped * shapedWeights).sum(2);
            }
        }

        public Tensor TransformCoordinates(Tensor coordinates, long originalHeight, long originalWidth)
        {
            if (coordinates.size(-1) != 2)
                throw new ArgumentException("Coordinates must have X,Y in the last dimension.", nameof(coordinates));
            var transformed = coordinates.clone().to_type(ScalarType.Float32);
            transformed[TensorIndex.Ellipsis, TensorIndex.Single(0)] =
                transformed[TensorIndex.Ellipsis, TensorIndex.Single(0)] / originalWidth * _resolution;
            transformed[TensorIndex.Ellipsis, TensorIndex.Single(1)] =
                transformed[TensorIndex.Ellipsis, TensorIndex.Single(1)] / originalHeight * _resolution;
            return transformed;
        }

        public Tensor PostprocessMasks(Tensor masks, long originalHeight, long originalWidth) =>
            interpolate(masks.to_type(ScalarType.Float32),
                size: new[] { originalHeight, originalWidth },
                mode: InterpolationMode.Bilinear,
                align_corners: false);
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
        private bool _isBatch = false;
        private Tensor? _imageEmbedding;
        private IList<Tensor>? _highResFeatures;
        private List<long[]>? _origHw;
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
            using var noGrad = no_grad();
            ResetPredictor();

            long h, w;
            if (image.dim() != 3)
            {
                throw new ArgumentException("Image must be a 3D RGB tensor in [H,W,3] or [3,H,W] layout.", nameof(image));
            }

            // HWC is the preferred public contract and wins for [3,W,3].
            if (image.size(2) == 3)
            {
                h = image.size(0);
                w = image.size(1);
            }
            else if (image.size(0) == 3)
            {
                h = image.size(1);
                w = image.size(2);
            }
            else
            {
                throw new ArgumentException("Image must have exactly 3 RGB channels in [H,W,3] or [3,H,W] layout.", nameof(image));
            }

            _origHw = new List<long[]> { new long[] { h, w } };

            using var transformedImage = _transforms.__call(image);
            using var inputImage = transformedImage.unsqueeze(0).to(_device);

            SetBackboneFeatures(inputImage);
            _isBatch = false;
            _isImageSet = true;
        }

        /// <summary>
        /// Set a batch of RGB images for prediction. The images may have different
        /// original sizes; all backbone embeddings are computed in one batch.
        /// </summary>
        public void SetImageBatch(IReadOnlyList<Tensor> images)
        {
            ArgumentNullException.ThrowIfNull(images);
            if (images.Count == 0)
                throw new ArgumentException("At least one image is required.", nameof(images));

            using var noGrad = no_grad();
            ResetPredictor();

            _origHw = new List<long[]>(images.Count);
            var transformedImages = new List<Tensor>(images.Count);
            try
            {
                foreach (var image in images)
                {
                    var (height, width) = GetImageSize(image);
                    _origHw.Add(new long[] { height, width });
                    transformedImages.Add(_transforms.__call(image));
                }

                using var imageBatch = stack(transformedImages.ToArray()).to(_device);
                SetBackboneFeatures(imageBatch);
                _isBatch = true;
                _isImageSet = true;
            }
            finally
            {
                foreach (var transformedImage in transformedImages)
                    transformedImage.Dispose();
            }
        }

        private static (long Height, long Width) GetImageSize(Tensor image)
        {
            if (image.dim() != 3)
                throw new ArgumentException("Image must be a 3D RGB tensor in [H,W,3] or [3,H,W] layout.", nameof(image));

            if (image.size(2) == 3)
                return (image.size(0), image.size(1));
            if (image.size(0) == 3)
                return (image.size(1), image.size(2));

            throw new ArgumentException("Image must have exactly 3 RGB channels in [H,W,3] or [3,H,W] layout.", nameof(image));
        }

        private void SetBackboneFeatures(Tensor inputImages)
        {
            var backboneOut = _model.ForwardImage(inputImages);
            var (visionFeats, _, featureSizes) = _model.PrepareBackboneFeatures(backboneOut);

            if (_model.directly_add_no_mem_embed)
            {
                visionFeats[visionFeats.Count - 1] = visionFeats[visionFeats.Count - 1] + _model.no_mem_embed;
            }

            var feats = new Tensor[visionFeats.Count];
            for (int i = 0; i < visionFeats.Count; i++)
            {
                var feat = visionFeats[i];
                var (featureHeight, featureWidth) = featureSizes[i];
                feats[i] = feat.permute(new long[] { 1, 2, 0 }).view(
                    new long[] { inputImages.size(0), -1, featureHeight, featureWidth });
            }

            _imageEmbedding = feats[^1];
            _highResFeatures = feats.Take(feats.Length - 1).ToList();
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
            bool returnLogits = false,
            bool normalizeCoordinates = true)
        {
            using var noGrad = no_grad();
            if (!_isImageSet)
                throw new InvalidOperationException("An image must be set with SetImage before prediction.");
            if (_isBatch)
                throw new InvalidOperationException("A batch is set; use PredictBatch for prediction.");

            return PredictInternal(pointCoords, pointLabels, box, maskInput,
                multimaskOutput, returnLogits, normalizeCoordinates, imageIndex: 0);
        }

        /// <summary>
        /// Predict masks for every image previously supplied to <see cref="SetImageBatch"/>.
        /// Each prompt list must either be null or contain one entry per image.
        /// </summary>
        public IReadOnlyList<(Tensor Masks, Tensor IouPredictions, Tensor LowResMasks)> PredictBatch(
            IReadOnlyList<Tensor?>? pointCoordsBatch = null,
            IReadOnlyList<Tensor?>? pointLabelsBatch = null,
            IReadOnlyList<Tensor?>? boxBatch = null,
            IReadOnlyList<Tensor?>? maskInputBatch = null,
            bool multimaskOutput = true,
            bool returnLogits = false,
            bool normalizeCoordinates = true)
        {
            using var noGrad = no_grad();
            if (!_isImageSet)
                throw new InvalidOperationException("An image batch must be set with SetImageBatch before prediction.");
            if (!_isBatch)
                throw new InvalidOperationException("A single image is set; use Predict for prediction.");

            int imageCount = _origHw!.Count;
            ValidatePromptBatch(pointCoordsBatch, imageCount, nameof(pointCoordsBatch));
            ValidatePromptBatch(pointLabelsBatch, imageCount, nameof(pointLabelsBatch));
            ValidatePromptBatch(boxBatch, imageCount, nameof(boxBatch));
            ValidatePromptBatch(maskInputBatch, imageCount, nameof(maskInputBatch));

            var results = new List<(Tensor, Tensor, Tensor)>(imageCount);
            for (int imageIndex = 0; imageIndex < imageCount; imageIndex++)
            {
                results.Add(PredictInternal(
                    GetPrompt(pointCoordsBatch, imageIndex),
                    GetPrompt(pointLabelsBatch, imageIndex),
                    GetPrompt(boxBatch, imageIndex),
                    GetPrompt(maskInputBatch, imageIndex),
                    multimaskOutput,
                    returnLogits,
                    normalizeCoordinates,
                    imageIndex));
            }

            return results;
        }

        private static void ValidatePromptBatch(IReadOnlyList<Tensor?>? prompts, int imageCount, string parameterName)
        {
            if (prompts is not null && prompts.Count != imageCount)
                throw new ArgumentException($"{parameterName} must contain one entry per image ({imageCount}).", parameterName);
        }

        private static Tensor? GetPrompt(IReadOnlyList<Tensor?>? prompts, int imageIndex)
            => prompts is null ? null : prompts[imageIndex];

        private (Tensor Masks, Tensor IouPredictions, Tensor LowResMasks) PredictInternal(
            Tensor? pointCoords,
            Tensor? pointLabels,
            Tensor? box,
            Tensor? maskInput,
            bool multimaskOutput,
            bool returnLogits,
            bool normalizeCoordinates,
            int imageIndex)
        {
            var originalSize = _origHw![imageIndex];

            Tensor? unnormCoords = null;
            Tensor? labels = null;
            Tensor? unnormBox = null;

            if (pointCoords is not null)
            {
                if (pointLabels is null)
                    throw new ArgumentException("point_labels must be supplied if point_coords is supplied.");

                unnormCoords = normalizeCoordinates
                    ? _transforms.TransformCoordinates(pointCoords, originalSize[0], originalSize[1]).to(_device)
                    : pointCoords.to(_device);
                labels = pointLabels.to(_device).to(ScalarType.Int32);

                if (unnormCoords.dim() == 2) unnormCoords = unnormCoords.unsqueeze(0);
                if (labels.dim() == 1) labels = labels.unsqueeze(0);
            }

            if (box is not null)
            {
                var boxCoordinates = box.reshape(-1, 2, 2);
                unnormBox = (normalizeCoordinates
                    ? _transforms.TransformCoordinates(boxCoordinates, originalSize[0], originalSize[1])
                    : boxCoordinates).to(_device).reshape(-1, 2, 2);
                var boxLabels = tensor(new long[] { 2, 3 }, dtype: ScalarType.Int32, device: _device)
                    .reshape(1, 2).repeat(unnormBox.size(0), 1);
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

            using var imageEmbedding = _isBatch
                ? _imageEmbedding![imageIndex].unsqueeze(0)
                : _imageEmbedding!.alias();
            var highResFeatures = _isBatch
                ? _highResFeatures!.Select(feature => feature[imageIndex].unsqueeze(0)).ToList()
                : (_highResFeatures ?? new List<Tensor>()).Select(feature => feature.alias()).ToList();

            Tensor lowResMultimasks;
            Tensor ious;
            try
            {
                (lowResMultimasks, ious, _, _) = _model.sam_mask_decoder.forward(
                    imageEmbedding,
                    _model.sam_prompt_encoder.get_dense_pe(),
                    sparseEmbeddings,
                    denseEmbeddings,
                    multimaskOutput,
                    batchedMode,
                    highResFeatures);
            }
            finally
            {
                foreach (var feature in highResFeatures)
                    feature.Dispose();
            }

            var fullResolutionMasks = _transforms.PostprocessMasks(
                lowResMultimasks, originalSize[0], originalSize[1]);
            Tensor masks;
            if (returnLogits)
            {
                masks = fullResolutionMasks;
            }
            else
            {
                masks = (fullResolutionMasks > MaskThreshold).to(ScalarType.Float32);
            }

            Tensor lowResMasksOut = clamp(lowResMultimasks, -32.0, 32.0);

            return (masks, ious, lowResMasksOut);
        }

        public void ResetPredictor()
        {
            _imageEmbedding?.Dispose();
            if (_highResFeatures is not null)
                foreach (var feature in _highResFeatures)
                    feature.Dispose();
            _isImageSet = false;
            _isBatch = false;
            _imageEmbedding = null;
            _highResFeatures = null;
            _origHw = null;
        }

        public void Dispose()
        {
            ResetPredictor();
            _model.Dispose();
        }
    }
}
