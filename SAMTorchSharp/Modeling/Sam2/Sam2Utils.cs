using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SAMTorchSharp.Modeling.Sam2
{
    /// <summary>
    /// 对应 sam2/modeling/sam2_utils.py 中的通用小组件：DropPath、MLP、LayerNorm2d。
    /// 与 SAM1 版本的等价实现区别在于：MLP 支持自定义激活函数（Hiera 中使用 GELU，
    /// 而 SAM1 的 MaskDecoder.MLP 固定使用 ReLU），因此这里单独实现，不与 Sam1 共享。
    /// </summary>
    public class DropPath : Module<Tensor, Tensor>
    {
        private readonly double dropProb;
        private readonly bool scaleByKeep;

        public DropPath(double dropProb = 0.0, bool scaleByKeep = true, string name = "DropPath") : base(name)
        {
            this.dropProb = dropProb;
            this.scaleByKeep = scaleByKeep;
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            if (dropProb == 0.0 || !training)
            {
                return x;
            }
            double keepProb = 1 - dropProb;
            var shape = new long[] { x.shape[0] }.Concat(Enumerable.Repeat(1L, (int)x.ndim - 1)).ToArray();
            Tensor randomTensor = x.new_empty(shape).bernoulli_(keepProb);
            if (keepProb > 0.0 && scaleByKeep)
            {
                randomTensor.div_(keepProb);
            }
            return x * randomTensor;
        }
    }

    /// <summary>
    /// 对应 sam2_utils.py: MLP。支持指定隐藏层激活函数（Hiera 使用 GELU，SAM 解码器头部使用 ReLU），
    /// 输出层不经过激活，sigmoidOutput=true 时对最终输出额外过一次 sigmoid。
    /// </summary>
    public class MLP : Module<Tensor, Tensor>
    {
        private readonly bool sigmoidOutput;
        private readonly ModuleList<Linear> layers;
        private readonly Module<Tensor, Tensor> act;

        public MLP(
            int inputDim,
            int hiddenDim,
            int outputDim,
            int numLayers,
            Func<Module<Tensor, Tensor>>? activation = null,
            bool sigmoidOutput = false,
            string name = "MLP") : base(name)
        {
            this.sigmoidOutput = sigmoidOutput;
            act = activation is null ? ReLU() : activation();

            layers = new ModuleList<Linear>();
            layers.append(Linear(inputDim, hiddenDim));
            for (int i = 1; i < numLayers - 1; i++)
            {
                layers.append(Linear(hiddenDim, hiddenDim));
            }
            layers.append(Linear(hiddenDim, outputDim));

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            int len = layers.Count;
            for (int i = 0; i < len - 1; i++)
            {
                x = act.forward(layers[i].forward(x));
            }
            x = layers[len - 1].forward(x);
            if (sigmoidOutput)
            {
                x = sigmoid(x);
            }
            return x;
        }
    }

    /// <summary>
    /// 对应 sam2_utils.py: LayerNorm2d（在通道维度上做 LayerNorm，输入布局为 [B,C,H,W]）。
    /// </summary>
    public class LayerNorm2d : Module<Tensor, Tensor>
    {
        private readonly Parameter weight;
        private readonly Parameter bias;
        private readonly double eps;

        public LayerNorm2d(long numChannels, double eps = 1e-6, string name = "LayerNorm2d") : base(name)
        {
            this.eps = eps;
            weight = Parameter(ones(numChannels));
            bias = Parameter(zeros(numChannels));
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            Tensor u = x.mean(new long[] { 1 }, keepdim: true);
            Tensor s = (x - u).pow(2).mean(new long[] { 1 }, keepdim: true);
            x = (x - u) / sqrt(s + eps);
            x = weight[TensorIndex.Colon, TensorIndex.None, TensorIndex.None] * x
                + bias[TensorIndex.Colon, TensorIndex.None, TensorIndex.None];
            return x;
        }
    }
}
