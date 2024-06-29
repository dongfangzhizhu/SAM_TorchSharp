using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SAMTorchSharp.Modeling
{

    public class MaskDecoder : Module<Tensor, Tensor, Tensor, Tensor,bool, (Tensor, Tensor)>
    {
        private readonly int transformerDim;
        private readonly Module<Tensor,Tensor,Tensor, (Tensor, Tensor)> transformer;
        private readonly int numMaskTokens;
        private readonly Module<Tensor, Tensor> activation;
        private readonly int iouHeadDepth;
        private readonly int iouHeadHiddenDim;

        private readonly Embedding iou_token;
        private readonly Embedding mask_tokens;
        private readonly Module<Tensor, Tensor> output_upscaling;
        private readonly ModuleList<Module<Tensor, Tensor>> output_hypernetworks_mlps;
        private readonly Module<Tensor, Tensor> iou_prediction_head;

        public MaskDecoder(int transformerDim, Module<Tensor, Tensor, Tensor, (Tensor, Tensor)> transformer,int numMultimaskOutputs = 3,Module<Tensor, Tensor> activation = null,int iouHeadDepth = 3,int iouHeadHiddenDim = 256,string name= "MaskDecoder") : base(name)
        {
            this.transformerDim = transformerDim;
            this.transformer = transformer;

            this.iou_token = Embedding(1, transformerDim);
            //this.register_module("iou_token", iou_token);

            numMaskTokens = numMultimaskOutputs + 1;
            this.mask_tokens = Embedding(numMaskTokens, transformerDim);
            //this.register_module("mask_tokens", mask_tokens);

            this.activation = activation ?? GELU();
            this.iouHeadDepth = iouHeadDepth;
            this.iouHeadHiddenDim = iouHeadHiddenDim;

            this.output_upscaling = Sequential(
                ConvTranspose2d(transformerDim, transformerDim / 4, 2, 2),
                new LayerNorm2d(transformerDim / 4 ),
                this.activation,
                ConvTranspose2d(transformerDim / 4, transformerDim / 8, 2, 2),
                this.activation
            );

            this.output_hypernetworks_mlps = new ModuleList<Module<Tensor, Tensor>>();
            for (int i = 0; i < numMaskTokens; i++)
            {
                var mlp = new MLP(transformerDim, transformerDim, transformerDim / 8, 3);
                output_hypernetworks_mlps.append(mlp);
            }

            this.iou_prediction_head = new MLP(transformerDim, iouHeadHiddenDim, numMaskTokens, iouHeadDepth);
            RegisterComponents();
        }

        public override (Tensor, Tensor) forward(Tensor imageEmbeddings, Tensor imagePe, Tensor sparsePromptEmbeddings, Tensor densePromptEmbeddings, bool multimaskOutput)
        {
            var (masks, iou_pred) = this.predict_masks(imageEmbeddings, imagePe, sparsePromptEmbeddings, densePromptEmbeddings);
            var mask_slice = multimaskOutput ? TensorIndex.Slice(1) : TensorIndex.Slice(0, 1);
            masks = masks[TensorIndex.Colon, mask_slice, TensorIndex.Colon];
            iou_pred = iou_pred[TensorIndex.Colon, mask_slice];
            return (masks, iou_pred);
        }

        public Tuple<Tensor, Tensor> predict_masks(Tensor imageEmbeddings, Tensor imagePe, Tensor sparsePromptEmbeddings, Tensor densePromptEmbeddings)
        {
            // Concatenate output tokens
            Tensor outputTokens = torch.cat(new Tensor[] { this.iou_token.weight, this.mask_tokens.weight }, 0);
            outputTokens = outputTokens.unsqueeze(0).expand(sparsePromptEmbeddings.size(0), -1, -1);
            Tensor tokens = torch.cat(new Tensor[] { outputTokens, sparsePromptEmbeddings }, 1);

            // Expand per-image data in batch direction to be per-mask
            Tensor src = torch.repeat_interleave(imageEmbeddings, tokens.size(0), 0);
            src.add_(densePromptEmbeddings);
            Tensor posSrc = torch.repeat_interleave(imagePe, tokens.size(0), 0);
            long b = src.size(0);
            long c = src.size(1);
            long h = src.size(2);
            long w = src.size(3);
            // Run the transformer
            Tensor hs;
            (hs, src) = this.transformer.forward(src, posSrc, tokens);
            Tensor iouTokenOut = hs[TensorIndex.Colon, 0, TensorIndex.Colon];
            Tensor maskTokensOut = hs[TensorIndex.Colon, TensorIndex.Slice(1, 1+ numMaskTokens), TensorIndex.Colon];

            // Upscale mask embeddings and predict masks using the mask tokens
            src = src.transpose(1, 2).view(b, c, h, w);


            Tensor upscaledEmbedding = this.output_upscaling.forward(src);
            List<Tensor> hyperInList = new List<Tensor>();
            for (int i = 0; i < this.numMaskTokens; i++)
            {
                hyperInList.Add(this.output_hypernetworks_mlps[i].forward(maskTokensOut.index(new[] { TensorIndex.Colon, i, TensorIndex.Colon })));
            }
            Tensor hyperIn = torch.stack(hyperInList, 1);
            b = upscaledEmbedding.size(0);
            c = upscaledEmbedding.size(1);
            h = upscaledEmbedding.size(2);
            w = upscaledEmbedding.size(3);
            Tensor masks = (hyperIn.matmul(upscaledEmbedding.view(b, c, h * w))).view(b, -1, h, w);

            // Generate mask quality predictions
            Tensor iouPred = this.iou_prediction_head.forward(iouTokenOut);

            return new Tuple<Tensor, Tensor>(masks, iouPred);
        }
    }

    public class MLP : Module<Tensor, Tensor>
    {
        private readonly bool sigmoidOutput;
        private readonly ModuleList<Linear> layers;

        public MLP(int inputDim, int hiddenDim, int outputDim, int numLayers, bool sigmoidOutput = false,string name= "MLP") : base(name)
        {
            this.sigmoidOutput = sigmoidOutput;
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
            var len = layers.Count;
            for (int i = 0; i < (len-1); i++)
            {
                x = functional.relu(layers[i].forward(x));
            }

            x = layers[len - 1].forward(x);
            if (sigmoidOutput)
            {
                x = sigmoid(x);
            }
            return x;
        }
    }
}
