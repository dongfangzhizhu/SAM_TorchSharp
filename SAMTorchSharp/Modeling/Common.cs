using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
namespace SAMTorchSharp.Modeling
{
    public abstract class ImageEncoderViTBase : Module<Tensor, Tensor>
    {
        public readonly int imgSize;
        public ImageEncoderViTBase(int imgSize,string name):base(name)
        {
            this.imgSize = imgSize;
        }
    }
    /// <summary>
    /// 已经检验对比过
    /// </summary>
    internal class MLPBlock : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> lin1;
        private readonly Module<Tensor, Tensor> lin2;
        private readonly Module<Tensor, Tensor> act;
        public MLPBlock(string name, int embeddingDim, int mlpDim, Func<Module<Tensor, Tensor>> act = null) : base(name)
        {
            this.lin1 = nn.Linear(embeddingDim, mlpDim);
            this.lin2 = nn.Linear(mlpDim, embeddingDim);
            this.act = act is null ? nn.GELU() : act();
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            return this.lin2.forward(this.act.forward(this.lin1.forward(x)));
        }
    }

    internal class LayerNorm2d : Module<Tensor, Tensor>
    {
        private readonly Tensor weight;
        private readonly Tensor bias;
        private readonly double eps;

        public LayerNorm2d(long numChannels, double eps = 1e-6, string name = "LayerNorm2d") : base(name)
        {
            // 创建权重和偏置张量，设置requires_grad为true
            this.weight = torch.ones(numChannels, requires_grad: true);
            this.bias = torch.zeros(numChannels, requires_grad: true);
            this.eps = eps;
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            // 计算均值和方差
            var u = x.mean(new long[] { 1 }, keepdim: true);
            var s = (x - u).pow(2).mean(new long[] { 1 }, keepdim: true);
            x = (x - u) / torch.sqrt(s + eps);
            x = this.weight[TensorIndex.Colon, TensorIndex.None, TensorIndex.None] * x + this.bias[TensorIndex.Colon, TensorIndex.None, TensorIndex.None];

            return x;
        }
    }
}
