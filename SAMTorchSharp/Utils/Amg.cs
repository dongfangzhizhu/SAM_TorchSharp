using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static System.Runtime.InteropServices.JavaScript.JSType;
using static TorchSharp.torch;

namespace SAMTorchSharp.Utils
{
    public class MaskData
    {
        private Dictionary<string, object> _stats;

        public MaskData(params KeyValuePair<string, object>[] kwargs)
        {
            _stats = new Dictionary<string, object>();
            foreach (var kvp in kwargs)
            {
                var value = kvp.Value;
                if (!(value is IList || value is Array || value is Tensor))
                {
                    throw new ArgumentException("MaskData only supports list, arrays, and torch tensors.");
                }

                _stats[kvp.Key] = value;
            }
        }

        public object this[string key]
        {
            get => _stats[key];
            set
            {
                if (!(value is IList || value is Array || value is Tensor))
                {
                    throw new ArgumentException("MaskData only supports list, arrays, and torch tensors.");
                }

                _stats[key] = value;
            }
        }

        public void Remove(string key)
        {
            _stats.Remove(key);
        }

        public ICollection<string> Keys => _stats.Keys;

        public ICollection<object> Values => _stats.Values;

        public IEnumerable<KeyValuePair<string, object>> Items => _stats.AsEnumerable();

        public void Filter(Tensor keep)
        {
            foreach (var key in _stats.Keys.ToList())
            {
                var value = _stats[key];
                if (value == null)
                {
                    _stats[key] = null;
                }
                else if (value is Tensor tensorValue)
                {
                    _stats[key] = tensorValue.index_select(0, keep.to(tensorValue.device));
                }
                else if (value is Array arrayValue)
                {
                    var keepArray = keep.cpu().data<int>().ToArray();
                    _stats[key] = arrayValue.Cast<object>().Where((v, i) => keepArray[i] != 0).ToArray();
                }
                else if (value is IList listValue)
                {
                    if (keep.dtype == ScalarType.Bool)
                    {
                        _stats[key] = listValue.Cast<object>().Where((v, i) => keep[i].ToBoolean()).ToList();
                    }
                    else
                    {
                        var keepIndices = keep.cpu().data<int>().ToArray();
                        _stats[key] = keepIndices.Select(i => listValue[i]).ToList();
                    }
                }
                else
                {
                    throw new ArgumentException($"MaskData key {key} has an unsupported type {value.GetType()}.");
                }
            }
        }

        public void Cat(MaskData newStats)
        {
            foreach (var kvp in newStats.Items)
            {
                var key = kvp.Key;
                var value = kvp.Value;
                if (!_stats.ContainsKey(key) || _stats[key] == null)
                {
                    _stats[key] = value is ICloneable cloneableValue ? cloneableValue.Clone() : value;
                }
                else if (value is Tensor tensorValue)
                {
                    _stats[key] = torch.cat(new[] { (Tensor)_stats[key], tensorValue }, 0);
                }
                else if (value is Array arrayValue)
                {
                    var combined = Array.CreateInstance(arrayValue.GetType().GetElementType(),
                        ((Array)_stats[key]).Length + arrayValue.Length);
                    ((Array)_stats[key]).CopyTo(combined, 0);
                    arrayValue.CopyTo(combined, ((Array)_stats[key]).Length);
                    _stats[key] = combined;
                }
                else if (value is IList listValue)
                {
                    ((IList)_stats[key]).AddRange(listValue.Cast<object>()
                        .Select(v => v is ICloneable cloneableV ? cloneableV.Clone() : v));
                }
                else
                {
                    throw new ArgumentException($"MaskData key {key} has an unsupported type {value.GetType()}.");
                }
            }
        }

        public void ToNumpy()
        {
            foreach (var key in _stats.Keys.ToList())
            {
                if (_stats[key] is Tensor tensorValue)
                {
                    _stats[key] = tensorValue.cpu().numpy();
                }
            }
        }
    }

    public static class Extensions
    {
        public static void AddRange(this IList list, IEnumerable<object> items)
        {
            foreach (var item in items)
            {
                list.Add(item);
            }
        }
    }

    internal static class Amg
    {
        public static Tensor IsBoxNearCropEdge(Tensor boxes, List<int> cropBox, List<int> origBox, float atol = 20.0f)
        {
            var cropBoxTorch = cropBox.ToTorchTensor(new long[] { 1 }, boxes.Device).ToType(ScalarType.Float);
            var origBoxTorch = origBox.ToTorchTensor(new long[] { 1 }, boxes.Device).ToType(ScalarType.Float);
            boxes = UncropBoxesXYXY(boxes, cropBox).ToType(ScalarType.Float);

            var nearCropEdge = boxes.IsClose(cropBoxTorch, atol, 0f);
            var nearImageEdge = boxes.IsClose(origBoxTorch, atol, 0f);
            nearCropEdge = nearCropEdge.And(nearImageEdge.Not());

            return nearCropEdge.Any(dim: 1);
        }

        public static Tensor BoxXYXYToXYWH(Tensor boxXYXY)
        {
            var boxXYWH = boxXYXY.Clone();
            boxXYWH.SetData(boxXYWH.GetSlice(2), boxXYWH.GetSlice(2) - boxXYWH.GetSlice(0));
            boxXYWH.SetData(boxXYWH.GetSlice(3), boxXYWH.GetSlice(3) - boxXYWH.GetSlice(1));
            return boxXYWH;
        }

        public static IEnumerable<List<T>> BatchIterator<T>(int batchSize, params List<T>[] args)
        {
            if (args.Length == 0 || args.Any(arg => arg.Count != args[0].Count))
                throw new ArgumentException("All input lists must be of the same size.");

            int nBatches = (args[0].Count + batchSize - 1) / batchSize;
            for (int b = 0; b < nBatches; b++)
            {
                yield return Enumerable.Range(b * batchSize, Math.Min(batchSize, args[0].Count - b * batchSize))
                    .Select(i => args.Select(arg => arg[i]).ToList())
                    .ToList();
            }
        }

        public static List<Dictionary<string, object>> MaskToRLEPyTorch(TorchTensor tensor)
        {
            int b = tensor.Shape[0];
            int h = tensor.Shape[1];
            int w = tensor.Shape[2];

            // Permute and flatten
            var tensorPermuted = tensor.Permute(0, 2, 1).Reshape(new long[] { b, h * w });

            // Compute change indices
            var diff = tensorPermuted.GetSlice(1, null) ^ tensorPermuted.GetSlice(0, -1);
            var changeIndices = diff.NonZeroAsTuple();

            List<Dictionary<string, object>> outList = new List<Dictionary<string, object>>();
            for (int i = 0; i < b; i++)
            {
                var curIdxs = changeIndices.Item1.Where(idx => idx.Item0 == i).Select(idx => idx.Item1).ToArray();
                curIdxs = torch.tensor(new long[] { 0 })
                    .Add(curIdxs.Select(idx => idx + 1).Concat(new long[] { h * w })).ToArray();

                var btwIdxs = curIdxs.Skip(1).Zip(curIdxs, (a, b) => (long)(b - a)).ToArray();
                var counts = tensor[i, 0].Item<bool>() ? new List<long>() : new List<long> { 0 };
                counts.AddRange(btwIdxs);
                outList.Add(new Dictionary<string, object>
                {
                    { "size", new List<int> { h, w } },
                    { "counts", counts }
                });
            }

            return outList;
        }

        public static NDArray RLEToMask(Dictionary<string, object> rle)
        {
            int h = (int)rle["size"][0];
            int w = (int)rle["size"][1];
            bool[] mask = new bool[h * w];
            int idx = 0;
            bool parity = false;
            foreach (var count in (List<long>)rle["counts"])
            {
                Array.Fill(mask, idx, (int)count, parity);
                idx += count;
                parity = !parity;
            }

            return NP.array(mask).Reshape(w, h).Transpose();
        }

        public static int AreaFromRLE(Dictionary<string, object> rle)
        {
            return ((List<long>)rle["counts"]).Where((_, index) => index % 2 == 1).Sum();
        }

        public static Tensor CalculateStabilityScore(Tensor masks, float maskThreshold, float thresholdOffset)
        {
            // Cast to byte type (equivalent to int16 in PyTorch) to save memory
            var highThreshMasks = masks.Greater(maskThreshold + thresholdOffset).CastByte();
            var lowThreshMasks = masks.Greater(maskThreshold - thresholdOffset).CastByte();

            // Sum over last two dimensions to get intersection and union counts
            var intersections = highThreshMasks.Sum(new long[] { -1, -2 }, dtype: ScalarType.Int32);
            var unions = lowThreshMasks.Sum(new long[] { -1, -2 }, dtype: ScalarType.Int32);

            // Compute IoU (stability score)
            return intersections.Div(unions);
        }

        public static NDArray BuildPointGrid(int nPerSide)
        {
            float offset = 1f / (2 * nPerSide);
            float[] pointsOneSide = np.linspace(offset, 1 - offset, nPerSide).ToArray();
            float[,] pointsX = np.repeat(pointsOneSide, nPerSide, axis: 0);
            float[,] pointsY = np.tile(pointsOneSide, nPerSide);
            float[,,] points = np.dstack(new[] { pointsX, pointsY });
            return points.reshape(nPerSide * nPerSide, 2);
        }

        public static List<NDArray> BuildAllLayerPointGrids(int nPerSide, int nLayers, int scalePerLayer)
        {
            List<NDArray> pointsByLayer = new List<NDArray>();
            for (int i = 0; i <= nLayers; i++)
            {
                int nPoints = nPerSide / (int)Math.Pow(scalePerLayer, i);
                pointsByLayer.Add(BuildPointGrid(nPoints));
            }

            return pointsByLayer;
        }

        public static (List<(int, int, int, int)>, List<int>) GenerateCropBoxes(int[] imSize, int nLayers,
            float overlapRatio)
        {
            int imH = imSize[0];
            int imW = imSize[1];
            int shortSide = Math.Min(imH, imW);

            // Original image
            var cropBoxes = new List<(int, int, int, int)> { (0, 0, imW, imH) };
            var layerIdxs = new List<int> { 0 };

            Func<int, int, int, int> cropLen = (origLen, nCrops, overlap) =>
                (int)Math.Ceiling((overlap * (nCrops - 1) + origLen) / nCrops);

            for (int iLayer = 0; iLayer < nLayers; iLayer++)
            {
                int nCropsPerSide = (int)Math.Pow(2, iLayer + 1);
                int overlap = (int)(overlapRatio * shortSide * (2 / nCropsPerSide));

                int cropW = cropLen(imW, nCropsPerSide, overlap);
                int cropH = cropLen(imH, nCropsPerSide, overlap);

                int[] cropBoxX0 = Enumerable.Range(0, nCropsPerSide)
                    .Select(i => (int)((cropW - overlap) * i / nCropsPerSide)).ToArray();
                int[] cropBoxY0 = Enumerable.Range(0, nCropsPerSide)
                    .Select(i => (int)((cropH - overlap) * i / nCropsPerSide)).ToArray();

                // Crops in XYWH format
                foreach (var x0 in cropBoxX0)
                {
                    foreach (var y0 in cropBoxY0)
                    {
                        int x1 = Math.Min(x0 + cropW, imW);
                        int y1 = Math.Min(y0 + cropH, imH);
                        cropBoxes.Add((x0, y0, x1, y1));
                        layerIdxs.Add(iLayer + 1);
                    }
                }
            }

            return (cropBoxes, layerIdxs);
        }

        static Tensor UncropBoxesXYXY(Tensor boxes, List<int> cropBox)
        {
            int x0 = cropBox[0], y0 = cropBox[1];
            var offset = torch.tensor(new float[,] { { x0, y0, x0, y0 } }).to(boxes.device);

            if (boxes.shape.Length == 3)
            {
                offset = offset.unsqueeze(1);
            }

            return boxes + offset;
        }

        static Tensor UncropPoints(Tensor points, List<int> cropBox)
        {
            int x0 = cropBox[0], y0 = cropBox[1];
            var offset = torch.tensor(new float[,] { { x0, y0 } }).to(points.device);

            if (points.shape.Length == 3)
            {
                offset = offset.unsqueeze(1);
            }

            return points + offset;
        }

        static Tensor UncropMasks(Tensor masks, List<int> cropBox, int origH, int origW)
        {
            int x0 = cropBox[0], y0 = cropBox[1], x1 = cropBox[2], y1 = cropBox[3];

            if (x0 == 0 && y0 == 0 && x1 == origW && y1 == origH)
            {
                return masks;
            }

            int padX = origW - (x1 - x0);
            int padY = origH - (y1 - y0);
            var pad = new long[] { x0, padX - x0, y0, padY - y0 };

            return torch.nn.functional.pad(masks, pad, value: 0);
        }

        public static (Mat, bool) RemoveSmallRegions(Mat mask, double areaThresh, string mode)
        {
            if (mode != "holes" && mode != "islands")
            {
                throw new ArgumentException("Invalid mode. Use 'holes' or 'islands'.");
            }

            bool correctHoles = mode == "holes";
            Mat workingMask = new Mat();
            Cv2.BitwiseXor(mask, new Scalar(correctHoles ? 255 : 0), workingMask);
            workingMask.ConvertTo(workingMask, MatType.CV_8UC1);

            Mat labels = new Mat();
            Mat stats = new Mat();
            Mat centroids = new Mat();
            int nLabels = Cv2.ConnectedComponentsWithStats(workingMask, labels, stats, centroids, Connectivity.Eight);

            var sizes = stats.ColRange(4, 5).RowRange(1, stats.Rows).ToArray<int>(); // Row 0 is background label
            var smallRegions = new List<int>();
            for (int i = 0; i < sizes.Length; i++)
            {
                if (sizes[i] < areaThresh)
                {
                    smallRegions.Add(i + 1);
                }
            }

            if (smallRegions.Count == 0)
            {
                return (mask, false);
            }

            var fillLabels = new List<int> { 0 };
            fillLabels.AddRange(smallRegions);
            if (!correctHoles)
            {
                fillLabels = Enumerable.Range(0, nLabels).Where(i => !fillLabels.Contains(i)).ToList();
                if (fillLabels.Count == 0)
                {
                    fillLabels.Add(Array.IndexOf(sizes, sizes.Max()) + 1);
                }
            }

            var regions = labels.ToMat();
            var maskArr = regions.ToArray<int>().Select(val => fillLabels.Contains(val) ? 1 : 0).ToArray();
            mask = new Mat(regions.Rows, regions.Cols, MatType.CV_8UC1, maskArr);

            return (mask, true);
        }

        public static Dictionary<string, object> CocoEncodeRLE(Dictionary<string, object> uncompressedRle)
        {
            var maskUtils = new pycocotools.Mask(); // Assuming you have a binding for pycocotools in C#
            int h = ((int[])uncompressedRle["size"])[0];
            int w = ((int[])uncompressedRle["size"])[1];
            var rle = maskUtils.frPyObjects(uncompressedRle, h, w);
            rle["counts"] = rle["counts"].ToString(); // Necessary to serialize with JSON

            return rle;
        }

        public static Tensor BatchedMaskToBox(Tensor masks)
        {
            if (masks.numel() == 0)
            {
                return torch.zeros(new long[] { masks.shape[0], masks.shape[1], 4 }, device: masks.device);
            }

            var shape = masks.shape;
            int h = (int)shape[^2];
            int w = (int)shape[^1];

            if (shape.Length > 2)
            {
                masks = masks.flatten(0, -3);
            }
            else
            {
                masks = masks.unsqueeze(0);
            }

            var inHeight = masks.max(dim: -1).values;
            var inHeightCoords =
                inHeight * arange(h, ScalarType.Float32, inHeight.device)
                    .unsqueeze(0); //TODO: 暂时用ScalarType.Float32，不确定类型
            var bottomEdges = inHeightCoords.max(dim: -1).values;
            inHeightCoords += h * inHeight.bitwise_not();
            var topEdges = inHeightCoords.min(dim: -1).values;

            var inWidth = masks.max(dim: -2).values;
            var inWidthCoords =
                inWidth * arange(w, ScalarType.Float32, inWidth.device)
                    .unsqueeze(0); //TODO: 暂时用ScalarType.Float32，不确定类型
            var rightEdges = inWidthCoords.max(dim: -1).values;
            inWidthCoords += w * inWidth.bitwise_not();
            var leftEdges = inWidthCoords.min(dim: -1).values;

            var emptyFilter = (rightEdges < leftEdges) | (bottomEdges < topEdges);
            var outTensor = torch.stack(new Tensor[] { leftEdges, topEdges, rightEdges, bottomEdges }, dim: -1);
            outTensor *= emptyFilter.bitwise_not().unsqueeze(-1);

            if (shape.Length > 2)
            {
                outTensor = outTensor.reshape(shape.SkipLast(2).Append(4).ToArray());
            }
            else
            {
                outTensor = outTensor[0];
            }

            return outTensor;
        }
    }
}