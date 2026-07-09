using TorchSharp;
using static TorchSharp.torch;

namespace SAMTorchSharp.Modeling.Sam2
{
    /// <summary>
    /// 对应 sam2/modeling/position_encoding.py 中的 2D 轴向旋转位置编码 (RoPE) 相关函数
    /// (compute_axial_cis / apply_rotary_enc)。
    ///
    /// Python 端用 torch 的复数张量(torch.polar/view_as_complex)实现，TorchSharp 没有对等的
    /// 复数张量API，这里改用等价的实数 cos/sin 表 + 手动实部虚部旋转公式实现：
    ///   给定复数频率 freqs_cis = cos(theta) + i*sin(theta)，
    ///   对 x 的最后一维按相邻两个元素 (x0, x1) 视为复数 (real=x0, imag=x1)，
    ///   旋转后: out_real = x0*cos - x1*sin, out_imag = x0*sin + x1*cos，
    ///   再交错写回 (out0=out_real, out1=out_imag)。
    /// 这与 torch.view_as_complex(x) * freqs_cis 后 view_as_real 展开完全等价。
    /// </summary>
    public static class RotaryPositionEncoding
    {
        /// <summary>
        /// 对应 init_t_xy：返回长度 end_x*end_y 的 (t_x, t_y) 坐标序列，
        /// t_x[n] = n % end_x, t_y[n] = n / end_x （行主序展开的网格坐标）。
        /// </summary>
        private static (Tensor tx, Tensor ty) InitTxy(long endX, long endY)
        {
            Tensor t = arange(endX * endY, dtype: ScalarType.Float32);
            Tensor tx = t.remainder(endX);
            Tensor ty = (t / endX).floor();
            return (tx, ty);
        }

        /// <summary>
        /// 对应 compute_axial_cis：返回 (cos, sin) 表，形状均为 [end_x*end_y, dim/2]。
        /// dim 是每个注意力头的通道数（internal_dim / num_heads）。
        /// </summary>
        public static (Tensor cos, Tensor sin) ComputeAxialFreqs(long dim, long endX, long endY, double theta = 10000.0)
        {
            long quarterDim = dim / 4;
            Tensor idx = arange(0, dim, 4, dtype: ScalarType.Float32)[TensorIndex.Slice(0, quarterDim)];
            Tensor freqBase = 1.0 / pow(theta, idx / dim); // [dim/4]

            var (tx, ty) = InitTxy(endX, endY);
            Tensor freqsX = tx.unsqueeze(-1) * freqBase.unsqueeze(0); // [N, dim/4]
            Tensor freqsY = ty.unsqueeze(-1) * freqBase.unsqueeze(0); // [N, dim/4]
            Tensor freqs = cat(new[] { freqsX, freqsY }, dim: -1); // [N, dim/2]

            return (freqs.cos(), freqs.sin());
        }

        /// <summary>
        /// 沿 N (token) 维重复频率表 r 次（对应 Python 端 rope_k_repeat=true 时的 freqs_cis.repeat）。
        /// cos/sin 形状 [N, D] -> [N*r, D]。
        /// </summary>
        public static (Tensor cos, Tensor sin) RepeatAlongTokenDim(Tensor cos, Tensor sin, long r)
        {
            Tensor cosR = cos.unsqueeze(0).expand(r, -1, -1).reshape(-1, cos.shape[^1]);
            Tensor sinR = sin.unsqueeze(0).expand(r, -1, -1).reshape(-1, sin.shape[^1]);
            return (cosR, sinR);
        }

        /// <summary>
        /// 对 x（形状 [..., N, D]，D 为偶数）按相邻两元素做旋转编码。
        /// cos/sin 形状为 [N, D/2]，会自动 broadcast 到 x 除最后两维之外的所有维度。
        /// </summary>
        public static Tensor ApplyRotary(Tensor x, Tensor cos, Tensor sin)
        {
            long d = x.shape[^1];
            Tensor xEven = x[TensorIndex.Ellipsis, TensorIndex.Slice(0, null, 2)]; // real, [..., N, D/2]
            Tensor xOdd = x[TensorIndex.Ellipsis, TensorIndex.Slice(1, null, 2)]; // imag, [..., N, D/2]

            // cos/sin: [N, D/2] -> reshape 成可以 broadcast 到 x 的形状 [1,...,1,N,D/2]
            // xEven/xOdd 与 x 的维度数相同（只是最后一维变成 D/2），所以 broadcastShape 长度应为 x.ndim。
            int extraDims = (int)x.ndim - 2;
            long[] broadcastShape = new long[x.ndim];
            for (int i = 0; i < extraDims; i++) broadcastShape[i] = 1;
            broadcastShape[extraDims] = cos.shape[0];
            broadcastShape[extraDims + 1] = cos.shape[1];
            Tensor cosB = cos.reshape(broadcastShape);
            Tensor sinB = sin.reshape(broadcastShape);

            Tensor outReal = xEven * cosB - xOdd * sinB;
            Tensor outImag = xEven * sinB + xOdd * cosB;

            Tensor outT = stack(new[] { outReal, outImag }, dim: -1).flatten(-2);
            return outT;
        }
    }
}
