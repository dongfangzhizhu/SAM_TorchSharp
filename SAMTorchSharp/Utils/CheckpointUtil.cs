using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;

namespace SAMTorchSharp.Utils
{
    /// <summary>
    /// TODO:实现类似pytorch里的checkpoint效果
    /// </summary>
    internal class CheckpointUtil
    {
        public static Tensor Checkpoint(Module<Tensor, Tensor> module, Tensor x)
        {
            return module.forward(x);
        }
    }
}
