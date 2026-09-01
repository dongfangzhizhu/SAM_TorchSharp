using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn.functional;
using System.Collections.Generic;
using System;
using SAMTorchSharp.Modeling.Sam2;

namespace SAMTorchSharp
{
    /// <summary>
    /// Automatic mask generation for SAM2 images.
    /// Generates a grid of point prompts over the image, then filters
    /// low quality and duplicate masks.
    /// </summary>
    public class SAM2AutomaticMaskGenerator : IDisposable
    {
        private readonly SAM2ImagePredictor _predictor;
        private readonly double _predIouThresh;
        private readonly double _stabilityScore;
        private readonly double _stabilityScoreOffset;
        private readonly int _pointsPerBatch;
        private readonly float _maskThreshold;
        private readonly float _boxNmsThresh;
        private readonly float _cropNmsThresh;
        private readonly int _cropNLayers;
        private readonly double _cropOverlapRatio;
        private readonly int _cropNPointsDownscaleFactor;
        private readonly long _minMaskRegionArea;
        private readonly bool _useM2M;
        private readonly bool _multimaskOutput;
        private readonly double[][] _pointGrids;

        public SAM2AutomaticMaskGenerator(Sam2Base model,
            int pointsPerSide = 32,
            int pointsPerBatch = 64,
            double predIouThresh = 0.8,
            double stabilityScoreThreshold = 0.95,
            double stabilityScoreOffset = 1.0,
            float maskThreshold = 0.0f,
            float boxNmsThresh = 0.7f,
            int cropNLayers = 0,
            float cropNmsThresh = 0.7f,
            double cropOverlapRatio = 512.0 / 1500.0,
            int cropNPointsDownscaleFactor = 1,
            long minMaskRegionArea = 0,
            bool useM2M = false,
            bool multimaskOutput = true)
        {
            _predictor = new SAM2ImagePredictor(model);
            _pointsPerBatch = pointsPerBatch;
            _predIouThresh = predIouThresh;
            _stabilityScore = stabilityScoreThreshold;
            _stabilityScoreOffset = stabilityScoreOffset;
            _maskThreshold = maskThreshold;
            _boxNmsThresh = boxNmsThresh;
            _cropNmsThresh = cropNmsThresh;
            _cropNLayers = cropNLayers;
            _cropOverlapRatio = cropOverlapRatio;
            _cropNPointsDownscaleFactor = cropNPointsDownscaleFactor;
            _minMaskRegionArea = minMaskRegionArea;
            _useM2M = useM2M;
            _multimaskOutput = multimaskOutput;

            _pointGrids = AMGUtiities.BuildAllLayerPointGrids(
                pointsPerSide, cropNLayers, cropNPointsDownscaleFactor);
        }

        public List<MaskRecord> Generate(Tensor image)
        {
            var maskData = _GenerateMasks(image);

            var boxes = maskData.GetTensor("boxes");
            var iouPreds = maskData.GetTensor("iou_preds");
            var points = maskData.GetTensor("points");
            var stabilityScores = maskData.GetTensor("stability_score");
            var cropBoxes = maskData.GetTensor("crop_boxes");
            var rles = maskData.GetRles("rles");

            var currAnns = new List<MaskRecord>();
            int n = (int)boxes.size(0);
            for (int i = 0; i < n; i++)
            {
                var seg = rles[i];
                currAnns.Add(new MaskRecord
                {
                    Segmentation = seg,
                    Area = AMGUtiities.AreaFromRle(seg),
                    BBox = TensorToArray(AMGUtiities.BoxXYXYToXYWH(boxes.index(new TensorIndex[] { i }))),
                    PredictedIou = iouPreds[i].item<float>(),
                    PointCoords = TensorToArray(points[i]),
                    StabilityScore = stabilityScores[i].item<float>(),
                    CropBox = TensorToArray(AMGUtiities.BoxXYXYToXYWH(cropBoxes.index(new TensorIndex[] { i })))
                });
            }

            return currAnns;
        }

        private MaskData _GenerateMasks(Tensor image)
        {
            int origH = (int)image.size(0);
            int origW = (int)image.size(1);

            var (cropBoxes, layerIndices) = AMGUtiities.GenerateCropBoxes(
                origH, origW, _cropNLayers, _cropOverlapRatio);

            var data = new MaskData();
            for (int i = 0; i < cropBoxes.Length; i++)
            {
                var cropData = _ProcessCrop(image, cropBoxes[i], layerIndices[i], origH, origW);
                data.Cat(cropData);
            }

            if (cropBoxes.Length > 1 && (int)data.GetTensor("boxes").size(0) > 0)
            {
                var boxes = data.GetTensor("boxes");
                var xywh = AMGUtiities.BoxXYXYToXYWH(boxes);
                var areaX = xywh.index(new TensorIndex[] { TensorIndex.Ellipsis, 2 });
                var areaY = xywh.index(new TensorIndex[] { TensorIndex.Ellipsis, 3 });
                var invArea = 1.0 / (areaX * areaY + 1e-6);

                var categories = zeros(boxes.size(0), dtype: ScalarType.Int64, device: boxes.device);
                var keep = _BatchedNms(boxes, invArea, categories, _cropNmsThresh);
                data.Filter(keep.to(ScalarType.Bool));
            }

            return data;
        }

        private MaskData _ProcessCrop(Tensor image, int[] cropBox, int cropLayerIdx, int origSizeH, int origSizeW)
        {
            int x0 = cropBox[0];
            int y0 = cropBox[1];
            int x1 = cropBox[2];
            int y1 = cropBox[3];

            var croppedIm = image
                .narrow(0, y0, y1 - y0)
                .narrow(1, x0, x1 - x0);

            var croppedH = (int)croppedIm.size(0);
            var croppedW = (int)croppedIm.size(1);

            _predictor.SetImage(croppedIm);

            double[] pointGrid = _pointGrids[cropLayerIdx];
            double[] scaledPoints = new double[pointGrid.Length];
            for (int i = 0; i < pointGrid.Length; i += 2)
            {
                scaledPoints[i] = pointGrid[i] * croppedW;
                scaledPoints[i + 1] = pointGrid[i + 1] * croppedH;
            }

            var data = new MaskData();
            foreach (var batch in AMGUtiities.BatchIteratorDouble(_pointsPerBatch, scaledPoints))
            {
                var batchData = _ProcessBatch(batch, (int)croppedH, (int)croppedW, cropBox, origSizeH, origSizeW);
                data.Cat(batchData);
            }

            _predictor.ResetPredictor();

            if ((int)data.GetTensor("boxes").size(0) > 0)
            {
                var boxes = data.GetTensor("boxes");
                var iouPreds = data.GetTensor("iou_preds");
                var categories = zeros(boxes.size(0), dtype: ScalarType.Int64, device: boxes.device);
                var keep = _BatchedNms(boxes, iouPreds, categories, _boxNmsThresh);
                data.Filter(keep.to(ScalarType.Bool));
            }

            if ((int)data.GetTensor("boxes").size(0) > 0)
            {
                data.Set("boxes", AMGUtiities.UncropBoxes(data.GetTensor("boxes"), x0, y0));
                data.Set("points", AMGUtiities.UncropPoints(data.GetTensor("points"), x0, y0));
            }

            if ((int)data.GetTensor("boxes").size(0) > 0)
            {
                var cropBoxTensor = tensor(new[] { (float)x0, (float)y0, (float)x1, (float)y1 });
                int count = (int)data.GetTensor("boxes").size(0);
                var cropBoxesExpanded = cropBoxTensor.unsqueeze(0).expand(new long[] { count, 4 });
                data.Set("crop_boxes", cropBoxesExpanded);
            }

            return data;
        }

        private MaskData _ProcessBatch(double[] points, int imH, int imW, int[] cropBox, int origH, int origW)
        {
            var ptsArray = new float[points.Length];
            for (int i = 0; i < points.Length; i++)
                ptsArray[i] = (float)points[i];
            Tensor ptsTensor = tensor(ptsArray).view(-1, 2).to(ScalarType.Float32);

            var labels = ones(new long[] { ptsTensor.size(0) }, dtype: ScalarType.Int32);
            var (masks, iouPreds, lowResMasks) = _predictor.Predict(
                pointCoords: ptsTensor.unsqueeze(1),
                pointLabels: labels.to(ScalarType.Int32).unsqueeze(1),
                multimaskOutput: _multimaskOutput,
                returnLogits: true);

            int nPoints = (int)ptsTensor.size(0);
            int numMasks = (int)masks.size(1);

            var flatMasks = masks.flatten(0, 1);
            var flatIous = iouPreds.flatten(0, 1);
            var flatLowRes = lowResMasks.flatten(0, 1);
            var flatPoints = ptsTensor.repeat_interleave((long)numMasks, dim: 0);

            var data = new MaskData();
            data.Set("masks", flatMasks);
            data.Set("iou_preds", flatIous);
            data.Set("points", flatPoints);
            data.Set("low_res_masks", flatLowRes);

            if (!_useM2M)
            {
                if (_predIouThresh > 0.0)
                {
                    var iuKeep = data.GetTensor("iou_preds") > _predIouThresh;
                    data.Filter(iuKeep);
                }

                var stabilityScores = AMGUtiities.CalculateStabilityScore(
                    data.GetTensor("masks"), _maskThreshold, (float)_stabilityScoreOffset);
                data.Set("stability_score", stabilityScores);

                if (_stabilityScore > 0.0)
                {
                    var stKeep = data.GetTensor("stability_score") >= _stabilityScore;
                    data.Filter(stKeep);
                }
            }

            var threshed = (data.GetTensor("masks") > _maskThreshold);
            data.Set("masks", threshed.to(ScalarType.Bool));
            data.Set("boxes", AMGUtiities.BatchedMaskToBox(threshed));

            var origBox = new[] { 0, 0, origW, origH };
            var keepMaskArr = ~AMGUtiities.IsBoxNearCropEdge(
                data.GetTensor("boxes"), cropBox, origBox);
            if (!keepMaskArr.to(ScalarType.Bool).all().item<bool>())
            {
                data.Filter(keepMaskArr);
            }

            int xc0 = cropBox[0], yc0 = cropBox[1], xc1 = cropBox[2], yc1 = cropBox[3];
            Tensor uncropped = AMGUtiities.UncropMasks(
                data.GetTensor("masks").to(ScalarType.Float32), xc0, yc0, xc1, yc1, origH, origW);
            data.Set("rles", AMGUtiities.MaskToRle(uncropped.to(ScalarType.Bool)));
            data.Remove("masks");

            return data;
        }

        private Tensor _BatchedNms(Tensor boxes, Tensor scores, Tensor categories, float iouThresh)
        {
            var (sortedScores, sortedIdx) = scores.sort(descending: true);
            var sortedBoxes = boxes[sortedIdx].to(ScalarType.Float32);
            var sortedCategories = categories[sortedIdx];

            int n = (int)sortedBoxes.size(0);
            var keepFlags = new bool[n];
            for (int i = 0; i < n; i++) keepFlags[i] = true;

            var keepList = new List<Tensor>();

            for (int i = 0; i < n; i++)
            {
                if (!keepFlags[i]) continue;

                keepList.Add(sortedIdx[i]);

                for (int j = i + 1; j < n; j++)
                {
                    if (!keepFlags[j]) continue;
                    if (sortedCategories[i].item<long>() != sortedCategories[j].item<long>()) continue;

                    var iouVal = _ComputeSingleBoxIoU(sortedBoxes.index(new TensorIndex[] { i }), sortedBoxes.index(new TensorIndex[] { j }));
                    if (iouVal >= iouThresh)
                        keepFlags[j] = false;
                }
            }

            if (keepList.Count == 0)
                return zeros(0, dtype: ScalarType.Int64, device: boxes.device);

            return stack(keepList).to(ScalarType.Int64);
        }

        private float _ComputeSingleBoxIoU(Tensor box1, Tensor box2)
        {
            float x1Left = box1[0].item<float>();
            float y1Top = box1[1].item<float>();
            float x2Right = box1[2].item<float>();
            float y2Bottom = box1[3].item<float>();

            float x1r = box2[0].item<float>();
            float y1t = box2[1].item<float>();
            float x2r = box2[2].item<float>();
            float y2b = box2[3].item<float>();

            float ltX = x1Left > x1r ? x1Left : x1r;
            float ltY = y1Top > y1t ? y1Top : y1t;
            float rbX = x2Right < x2r ? x2Right : x2r;
            float rbY = y2Bottom < y2b ? y2Bottom : y2b;

            float interW = Math.Max(0f, rbX - ltX);
            float interH = Math.Max(0f, rbY - ltY);
            float inter = interW * interH;

            float area1 = (x2Right - x1Left) * (y2Bottom - y1Top);
            float area2 = (x2r - x1r) * (y2b - y1t);
            float union = area1 + area2 - inter;

            return inter / (union + 1e-6f);
        }

        private Tensor _ComputeIoU(Tensor box1, Tensor box2)
        {
            long N = box1.size(0);
            long M = box2.size(0);

            var x1Left = box1.index(new TensorIndex[] { TensorIndex.Ellipsis, 0 }).unsqueeze(1).expand(N, M);
            var y1Top = box1.index(new TensorIndex[] { TensorIndex.Ellipsis, 1 }).unsqueeze(1).expand(N, M);
            var x2Right = box1.index(new TensorIndex[] { TensorIndex.Ellipsis, 2 }).unsqueeze(1).expand(N, M);
            var y2Bottom = box1.index(new TensorIndex[] { TensorIndex.Ellipsis, 3 }).unsqueeze(1).expand(N, M);

            var x1r = box2.index(new TensorIndex[] { TensorIndex.Ellipsis, 0 }).unsqueeze(0).expand(N, M);
            var y1t = box2.index(new TensorIndex[] { TensorIndex.Ellipsis, 1 }).unsqueeze(0).expand(N, M);
            var x2r = box2.index(new TensorIndex[] { TensorIndex.Ellipsis, 2 }).unsqueeze(0).expand(N, M);
            var y2b = box2.index(new TensorIndex[] { TensorIndex.Ellipsis, 3 }).unsqueeze(0).expand(N, M);

            var ltX = where(x1Left > x1r, x1Left, x1r);
            var ltY = where(y1Top > y1t, y1Top, y1t);
            var rbX = where(x2Right < x2r, x2Right, x2r);
            var rbY = where(y2Bottom < y2b, y2Bottom, y2b);

            var interW = (rbX - ltX).clamp(min: 0);
            var interH = (rbY - ltY).clamp(min: 0);
            var inter = interW * interH;

            var area1 = (x2Right - x1Left) * (y2Bottom - y1Top);
            var area2 = (x2r - x1r) * (y2b - y1t);
            var union = area1 + area2 - inter;

            return inter / (union + 1e-6);
        }

        private float[] TensorToArray(Tensor t)
        {
            var arr = new float[(int)t.size(0)];
            for (int i = 0; i < (int)t.size(0); i++)
                arr[i] = t[i].item<float>();
            return arr;
        }

        public void Dispose()
        {
            _predictor.Dispose();
        }
    }

    /// <summary>
    /// Record for a single generated mask.
    /// </summary>
    public class MaskRecord
    {
        public RleElement? Segmentation { get; set; }
        public int Area { get; set; }
        public float[] BBox { get; set; } = null!;
        public float PredictedIou { get; set; }
        public float[] PointCoords { get; set; } = null!;
        public float StabilityScore { get; set; }
        public float[] CropBox { get; set; } = null!;
    }
}
