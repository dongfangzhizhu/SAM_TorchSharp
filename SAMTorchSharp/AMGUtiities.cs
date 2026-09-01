using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn.functional;
using System.Collections.Generic;

namespace SAMTorchSharp
{
    /// <summary>
    /// A structure for storing masks and their related data in batched format.
    /// Implements basic filtering and concatenation.
    /// Corresponds to sam2/utils/amg.py: MaskData.
    /// </summary>
    public class MaskData
    {
        private readonly Dictionary<string, object> _stats = new();

        public void Set(string key, Tensor value)
        {
            _stats[key] = value;
        }

        public void Set(string key, double[] value)
        {
            _stats[key] = torch.tensor(value);
        }

        public void Set(string key, float[] value)
        {
            _stats[key] = torch.tensor(value);
        }

        public void Set(string key, int[] value)
        {
            _stats[key] = torch.tensor(value);
        }

        public void Set(string key, long[] value)
        {
            _stats[key] = torch.tensor(value);
        }

        public void Set(string key, bool[] value)
        {
            _stats[key] = torch.tensor(value);
        }

        public void Set(string key, List<RleElement> value)
        {
            _stats[key] = value;
        }

        public T Get<T>(string key)
        {
            return (T)_stats[key];
        }

        public Tensor GetTensor(string key)
        {
            return (Tensor)_stats[key];
        }

        public List<RleElement> GetRles(string key)
        {
            return (List<RleElement>)_stats[key];
        }

        public void Remove(string key)
        {
            _stats.Remove(key);
        }

        public bool ContainsKey(string key)
        {
            return _stats.ContainsKey(key);
        }

        /// <summary>Filter entries by a boolean mask.</summary>
        public void Filter(Tensor keep)
        {
            var keepTensor = keep.dtype == ScalarType.Bool
                ? keep
                : keep.to(ScalarType.Int64);
            var indices = keepTensor.dtype == ScalarType.Bool
                ? keepTensor.nonzero().flatten().data<long>().ToArray()
                : keepTensor.flatten().data<long>().ToArray();
            foreach (var kvp in _stats)
            {
                if (kvp.Value is Tensor t)
                {
                    _stats[kvp.Key] = t[keepTensor];
                }
                else if (kvp.Value is List<RleElement> rles)
                {
                    _stats[kvp.Key] = indices.Select(index => rles[checked((int)index)]).ToList();
                }
            }
        }

        /// <summary>Concatenate another MaskData into this one.</summary>
        public void Cat(MaskData other)
        {
            foreach (var kvp in other._stats)
            {
                if (!_stats.ContainsKey(kvp.Key) || _stats[kvp.Key] is null)
                {
                    _stats[kvp.Key] = kvp.Value;
                }
                else if (kvp.Value is Tensor v && _stats[kvp.Key] is Tensor existing)
                {
                    _stats[kvp.Key] = cat(new[] { existing, v }, dim: 0);
                }
                else if (kvp.Value is List<RleElement> rles && _stats[kvp.Key] is List<RleElement> existingRles)
                {
                    existingRles.AddRange(rles);
                }
            }
        }
    }

    /// <summary>
    /// RLE-encoded mask (uncompressed, COCO format).
    /// </summary>
    public class RleElement
    {
        public long[] Size { get; set; } = null!;
        public int[] Counts { get; set; } = null!;
    }

    /// <summary>
    /// AMG Utilities: RLE encoding, stability score, mask-to-box, crop box generation, etc.
    /// </summary>
    public static class AMGUtiities
    {
        /// <summary>
        /// Encodes masks to an uncompressed RLE, in the format expected by pycoco tools.
        /// </summary>
        public static List<RleElement> MaskToRle(Tensor masks)
        {
            var B = masks.size(0);
            var H = masks.size(1);
            var W = masks.size(2);
            var flat = masks.permute(new long[] { 0, 2, 1 }).flatten(1);
            var flatBool = flat.to(ScalarType.Bool);

            var results = new List<RleElement>((int)B);
            for (int b = 0; b < (int)B; b++)
            {
                var row = flatBool[b];
                long[] arr = new long[row.size(0)];
                for (long i = 0; i < row.size(0); i++)
                {
                    arr[i] = row[i].item<bool>() ? 1 : 0;
                }

                var counts = new List<int>();
                bool prev = arr[0] == 1;
                int runStart = 0;

                if (prev)
                    counts.Add(0);

                for (int i = 1; i < arr.Length; i++)
                {
                    bool cur = arr[i] == 1;
                    if (cur != prev)
                    {
                        counts.Add(i - runStart);
                        runStart = i;
                        prev = cur;
                    }
                }
                counts.Add(arr.Length - runStart);

                results.Add(new RleElement
                {
                    Size = new[] { H, W },
                    Counts = counts.ToArray()
                });
            }

            return results;
        }

        /// <summary>
        /// Computes the stability score for a batch of masks.
        /// </summary>
        public static Tensor CalculateStabilityScore(Tensor masks, float maskThreshold, float thresholdOffset)
        {
            Tensor high = (masks > (maskThreshold + thresholdOffset)).to(ScalarType.Float32);
            Tensor low = (masks > (maskThreshold - thresholdOffset)).to(ScalarType.Float32);

            var intersections = high.sum(new long[] { -2, -1 }).to(ScalarType.Int16).to(ScalarType.Int32);
            var unions = low.sum(new long[] { -2, -1 }).to(ScalarType.Int16).to(ScalarType.Int32);

            return intersections.to(ScalarType.Float32) / (unions.to(ScalarType.Float32) + 1e-6);
        }

        /// <summary>
        /// Calculates bounding boxes in XYXY format around masks.
        /// </summary>
        public static Tensor BatchedMaskToBox(Tensor masks)
        {
            if (masks.size(0) == 0)
                return zeros(new long[] { 0, 4 }, device: masks.device);

            if (masks.dim() == 4)
                masks = masks.squeeze(1);
            masks = masks.to(ScalarType.Bool);

            var H = masks.shape[1];
            var W = masks.shape[2];
            var B = masks.shape[0];

            Tensor inHeight = masks.max(dim: 2).values;
            Tensor hCoords = arange(H, device: masks.device).unsqueeze(0).expand(B, H);
            Tensor bottomEdges = (inHeight * hCoords).max(dim: 1).values;
            Tensor inHeightCoords = inHeight * hCoords + H * (~inHeight);
            Tensor topEdges = inHeightCoords.min(dim: 1).values;

            Tensor inWidth = masks.max(dim: 1).values;
            Tensor wCoords = arange(W, device: masks.device).unsqueeze(0).expand(B, W);
            Tensor rightEdges = (inWidth * wCoords).max(dim: 1).values;
            Tensor inWidthCoords = inWidth * wCoords + W * (~inWidth);
            Tensor leftEdges = inWidthCoords.min(dim: 1).values;

            Tensor emptyFilter = (rightEdges < leftEdges) | (bottomEdges < topEdges);
            var result = stack(new[] { leftEdges, topEdges, rightEdges, bottomEdges }, dim: -1);
            result = result * (~emptyFilter).to(result.dtype).unsqueeze(-1);

            return result;
        }

        /// <summary>Converts a box from XYXY to XYWH format.</summary>
        public static Tensor BoxXYXYToXYWH(Tensor boxXYXY)
        {
            var result = boxXYXY.clone();
            var x0r = result.index(new TensorIndex[] { TensorIndex.Ellipsis, 0 });
            var y0r = result.index(new TensorIndex[] { TensorIndex.Ellipsis, 1 });
            var x1r = result.index(new TensorIndex[] { TensorIndex.Ellipsis, 2 });
            var y1r = result.index(new TensorIndex[] { TensorIndex.Ellipsis, 3 });
            result.index(new TensorIndex[] { TensorIndex.Ellipsis, 2 }).copy_(x1r - x0r);
            result.index(new TensorIndex[] { TensorIndex.Ellipsis, 3 }).copy_(y1r - y0r);
            return result;
        }

        /// <summary>
        /// Generates crop boxes for multi-scale mask generation.
        /// </summary>
        public static (int[][] CropBoxes, int[] LayerIndices) GenerateCropBoxes(
            int imH, int imW, int cropNLayers, double overlapRatio)
        {
            var cropBoxes = new List<int[]>();
            var layerIndices = new List<int>();

            cropBoxes.Add(new[] { 0, 0, imW, imH });
            layerIndices.Add(0);

            int shortSide = Math.Min(imH, imW);

            for (int iLayer = 0; iLayer < cropNLayers; iLayer++)
            {
                int nCropsPerSide = 1 << (iLayer + 1);
                int cropLen = (int)Math.Ceiling((overlapRatio * (nCropsPerSide - 1) + imW) / nCropsPerSide);

                double overlap = overlapRatio * shortSide * (2.0 / nCropsPerSide);
                var cropBoxX0 = new List<int>();
                var cropBoxY0 = new List<int>();
                for (int i = 0; i < nCropsPerSide; i++)
                {
                    cropBoxX0.Add((int)(i * (cropLen - overlap)));
                    cropBoxY0.Add((int)(i * (cropLen - overlap)));
                }

                foreach (var x0 in cropBoxX0)
                {
                    foreach (var y0 in cropBoxY0)
                    {
                        cropBoxes.Add(new[] {
                            x0, y0,
                            Math.Min(x0 + cropLen, imW),
                            Math.Min(y0 + cropLen, imH)
                        });
                        layerIndices.Add(iLayer + 1);
                    }
                }
            }

            return (cropBoxes.ToArray(), layerIndices.ToArray());
        }

        /// <summary>Uncrop boxes from crop coordinates back to original image coordinates.</summary>
        public static Tensor UncropBoxes(Tensor boxes, int x0, int y0, Device? device = null)
        {
            device ??= boxes.device;
            var offset = tensor(new[] { (float)x0, (float)y0, (float)x0, (float)y0 }, device: device);
            if (boxes.dim() == 3)
                offset = offset.unsqueeze(0);
            return boxes + offset;
        }

        /// <summary>Uncrop points from crop coordinates back to original image coordinates.</summary>
        public static Tensor UncropPoints(Tensor points, int x0, int y0, Device? device = null)
        {
            device ??= points.device;
            var offset = tensor(new[] { (float)x0, (float)y0 }, device: device);
            if (points.dim() == 3)
                offset = offset.unsqueeze(0);
            return points + offset;
        }

        /// <summary>Uncrop masks by padding them back to original size.</summary>
        public static Tensor UncropMasks(Tensor masks, int x0, int y0, int x1, int y1, int origH, int origW)
        {
            if (x0 == 0 && y0 == 0 && x1 == origW && y1 == origH)
                return masks;

            long padXSize = origW - (x1 - x0);
            long padYSize = origH - (y1 - y0);
            long padLeft = (long)x0;
            long padRight = padXSize - x0;
            long padTop = (long)y0;
            long padBottom = padYSize - y0;
            var padArgs = new long[] { padLeft, padRight, padTop, padBottom };

            return pad(masks, padArgs, value: 0);
        }

        /// <summary>Check if a box is near the edge of a crop but not the edge of the original image.</summary>
        public static Tensor IsBoxNearCropEdge(Tensor boxes, int[] cropBox, int[] origBox, float atol = 20.0f)
        {
            var uncropped = UncropBoxes(boxes, cropBox[0], cropBox[1]).to(ScalarType.Float32);

            Tensor cropBoxT = tensor(cropBox, dtype: ScalarType.Float32, device: boxes.device).unsqueeze(0);
            Tensor origBoxT = tensor(origBox, dtype: ScalarType.Float32, device: boxes.device).unsqueeze(0);

            var nearCropEdge = isclose(uncropped, cropBoxT, atol: atol);
            var nearImageEdge = isclose(uncropped, origBoxT, atol: atol);
            var nearCropEdgeOnly = nearCropEdge & (~nearImageEdge);

            return nearCropEdgeOnly.any(dim: 1);
        }

        /// <summary>Compute area from RLE.</summary>
        public static int AreaFromRle(RleElement rle)
        {
            int area = 0;
            for (int i = 1; i < rle.Counts.Length; i += 2)
            {
                area += rle.Counts[i];
            }
            return area;
        }

        /// <summary>Decode an uncompressed COCO RLE into a row-major binary mask.</summary>
        public static bool[,] RleToMask(RleElement rle)
        {
            int height = checked((int)rle.Size[0]);
            int width = checked((int)rle.Size[1]);
            var mask = new bool[height, width];
            int offset = 0;
            bool value = false;
            foreach (int count in rle.Counts)
            {
                for (int index = 0; index < count; index++, offset++)
                {
                    int x = offset / height;
                    int y = offset % height;
                    mask[y, x] = value;
                }
                value = !value;
            }
            if (offset != height * width)
                throw new InvalidDataException("RLE counts do not match the declared mask size.");
            return mask;
        }

        /// <summary>Build a 2D grid of points evenly spaced in [0,1]x[0,1].</summary>
        public static double[] BuildPointGrid(int nPerSide)
        {
            var points = new double[nPerSide * nPerSide, 2];
            for (int y = 0; y < nPerSide; y++)
            {
                for (int x = 0; x < nPerSide; x++)
                {
                    points[y * nPerSide + x, 0] = (x + 0.5) / nPerSide;
                    points[y * nPerSide + x, 1] = (y + 0.5) / nPerSide;
                }
            }

            var result = new double[points.Length];
            for (int i = 0; i < points.Length; i++)
                result[i] = points[i / 2, i % 2];
            return result;
        }

        /// <summary>Generate point grids for all crop layers.</summary>
        public static double[][] BuildAllLayerPointGrids(int nPerSide, int nLayers, int scalePerLayer)
        {
            var result = new List<double[]>();
            for (int i = 0; i <= nLayers; i++)
            {
                int nPoints = (int)Math.Ceiling((double)nPerSide / Math.Pow(scalePerLayer, i));
                result.Add(BuildPointGrid(nPoints));
            }
            return result.ToArray();
        }

        /// <summary>Batch iterator for double arrays.</summary>
        public static IEnumerable<double[]> BatchIteratorDouble(int batchSize, double[] points)
        {
            int n = points.Length / 2;
            for (int b = 0; b < n; b += batchSize)
            {
                int end = Math.Min(b + batchSize, n);
                int count = (end - b) * 2;
                var chunk = new double[count];
                Array.Copy(points, b * 2, chunk, 0, count);
                yield return chunk;
            }
        }
    }
}
