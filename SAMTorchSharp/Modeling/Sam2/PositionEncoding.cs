using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SAMTorchSharp.Modeling.Sam2
{
    /// <summary>
    /// 对应 sam2/modeling/position_encoding.py: PositionEmbeddingSine。
    /// 用于给 FpnNeck 输出的每个特征层生成正弦位置编码（NeRF/Transformer 风格），
    /// 仅依赖输入张量的空间尺寸，不含可训练参数。
    /// </summary>
    public class PositionEmbeddingSine : Module<Tensor, Tensor>
    {
        private readonly long numPosFeats;
        private readonly long temperature;
        private readonly bool normalize;
        private readonly double scale;

        public PositionEmbeddingSine(
            long numPosFeats,
            long temperature = 10000,
            bool normalize = true,
            double? scale = null,
            string name = "PositionEmbeddingSine") : base(name)
        {
            if (numPosFeats % 2 != 0)
            {
                throw new ArgumentException("Expecting even model width");
            }
            this.numPosFeats = numPosFeats / 2;
            this.temperature = temperature;
            this.normalize = normalize;

            if (scale.HasValue && !normalize)
            {
                throw new ArgumentException("normalize should be True if scale is passed");
            }
            this.scale = scale ?? 2 * Math.PI;

            RegisterComponents();
        }

        /// <summary>
        /// 对应 Python 端 _pe：给定 batch 大小和空间尺寸 (H,W)，生成 [B, 2*num_pos_feats, H, W] 的位置编码。
        /// </summary>
        private Tensor Pe(long B, Device device, long H, long W)
        {
            using var _ = no_grad();

            Tensor yEmbed = arange(1, H + 1, dtype: ScalarType.Float32, device: device)
                .view(1, -1, 1).repeat(B, 1, W);
            Tensor xEmbed = arange(1, W + 1, dtype: ScalarType.Float32, device: device)
                .view(1, 1, -1).repeat(B, H, 1);

            if (normalize)
            {
                double eps = 1e-6;
                yEmbed = yEmbed / (yEmbed[TensorIndex.Colon, TensorIndex.Slice(-1, null), TensorIndex.Colon] + eps) * scale;
                xEmbed = xEmbed / (xEmbed[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(-1, null)] + eps) * scale;
            }

            // Python: dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
            // 对非负整数值的浮点张量而言，(dim_t // 2) * 2 等价于 floor(dim_t / 2) * 2。
            Tensor dimT = arange(numPosFeats, dtype: ScalarType.Float32, device: device);
            Tensor dimTFloorHalf = (dimT / 2.0).floor() * 2.0;
            dimT = pow((double)temperature, dimTFloorHalf / numPosFeats);

            Tensor posX = xEmbed[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Colon, TensorIndex.None] / dimT;
            Tensor posY = yEmbed[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Colon, TensorIndex.None] / dimT;

            Tensor posXSin = posX[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(0, null, 2)].sin();
            Tensor posXCos = posX[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(1, null, 2)].cos();
            posX = stack(new[] { posXSin, posXCos }, dim: 4).flatten(3);

            Tensor posYSin = posY[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(0, null, 2)].sin();
            Tensor posYCos = posY[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(1, null, 2)].cos();
            posY = stack(new[] { posYSin, posYCos }, dim: 4).flatten(3);

            Tensor pos = cat(new[] { posY, posX }, dim: 3).permute(0, 3, 1, 2);
            return pos;
        }

        public override Tensor forward(Tensor x)
        {
            long B = x.shape[0];
            long H = x.shape[^2];
            long W = x.shape[^1];
            return Pe(B, x.device, H, W);
        }
    }
}
