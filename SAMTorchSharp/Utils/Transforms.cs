using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn.functional;

namespace SAMTorchSharp.Utils
{
    internal class ResizeLongestSide
    {
        private int target_length;

        public ResizeLongestSide(int target_length)
        {
            this.target_length = target_length;
        }

        public torch.Tensor ApplyImage(torch.Tensor image)
        {
            var target_size = GetPreprocessShape(image.shape[2], image.shape[3], target_length);
            return interpolate(image, size: new long[] { target_size.Item1, target_size.Item2 }, mode: torch.InterpolationMode.Bilinear, align_corners: false);
        }

        public torch.Tensor ApplyCoords(torch.Tensor coords, (long, long) original_size)
        {
            var (old_h, old_w) = original_size;
            var (new_h, new_w) = GetPreprocessShape(original_size.Item1, original_size.Item2, target_length);

            coords = coords.clone().to(float32);
            coords.index_put_(coords[torch.TensorIndex.Ellipsis, torch.TensorIndex.Single(0)] * (new_w / (double)old_w), torch.TensorIndex.Ellipsis, torch.TensorIndex.Single(0));
            coords.index_put_(coords[torch.TensorIndex.Ellipsis, torch.TensorIndex.Single(1)] * (new_h / (double)old_h), torch.TensorIndex.Ellipsis, torch.TensorIndex.Single(1));
            return coords;
        }

        public torch.Tensor ApplyBoxes(torch.Tensor boxes, (long, long) original_size)
        {
            boxes = ApplyCoords(boxes.view(-1, 2, 2), original_size).view(-1, 4);
            return boxes;
        }

        public static (long, long) GetPreprocessShape(long oldh, long oldw, long long_side_length)
        {
            var scale = long_side_length * 1.0 / Math.Max(oldh, oldw);
            var newh = (long)(oldh * scale + 0.5);
            var neww = (long)(oldw * scale + 0.5);
            return (newh, neww);
        }
    }
}
