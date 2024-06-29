using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SAMTorchSharp.Modeling
{
    public class Sam : Module<IList<IDictionary<string, object>>, bool, IList<IDictionary<string, Tensor>>>
    {
        public readonly float mask_threshold = 0.0f;
        public string image_format = "RGB";
        private readonly int imgSize;
        //ImageEncoderViT image_encoder;
        public readonly ImageEncoderViTBase image_encoder;
        public readonly PromptEncoder prompt_encoder;
        public readonly MaskDecoder mask_decoder;
        Tensor pixel_mean;
        Tensor pixel_std;

        public Sam(ImageEncoderViTBase image_encoder,PromptEncoder prompt_encoder,MaskDecoder mask_decoder,float[] pixel_mean = null, float[] pixel_std = null,string name="sam") : base(name)
        {
            if (pixel_mean is null)
            {
                pixel_mean = [123.675f, 116.28f, 103.53f];
            }

            if (pixel_std is null)
            {
                pixel_std = [58.395f, 57.12f, 57.375f];
            }
            imgSize = image_encoder.imgSize;
            this.image_encoder = image_encoder;
            this.prompt_encoder = prompt_encoder;
            this.mask_decoder = mask_decoder;
            this.pixel_mean = tensor(pixel_mean).view(-1, 1, 1);
            this.pixel_std = tensor(pixel_std).view(-1, 1, 1);
            register_buffer("pixel_mean", pixel_mean,false);
            register_buffer("pixel_std", pixel_std, false);
            RegisterComponents();
        }

        public Device device()
        {
            return this.pixel_mean.device;
        }

        public override IList<IDictionary<string, Tensor>> forward(IList<IDictionary<string, object>> batched_input,bool multimask_output)
        {
            var input_images = stack(batched_input.Select(x => preprocess(x["image"] as Tensor)), 0);
            Tensor image_embeddings = image_encoder.forward(input_images);

            var outputs = new List<IDictionary<string, Tensor>>();
            for (int i = 0; i < batched_input.Count; i++)
            {
                Tensor sparse_embeddings, dense_embeddings;
                Tuple<Tensor, Tensor> points = null;
                if (batched_input[i].ContainsKey("point_coords"))
                {
                    points = Tuple.Create(batched_input[i]["point_coords"] as Tensor, batched_input[i]["point_labels"] as torch.Tensor);
                }

                batched_input[i].TryGetValue("boxes", out object boxes);
                batched_input[i].TryGetValue("mask_inputs", out object mask_inputs);
                (sparse_embeddings, dense_embeddings) = prompt_encoder.forward(points, boxes as Tensor,mask_inputs as Tensor);

                Tensor low_res_masks, iou_predictions;
                (low_res_masks, iou_predictions) = mask_decoder.forward(
                    imageEmbeddings: image_embeddings[i].unsqueeze(0),
                    imagePe: prompt_encoder.get_dense_pe(),
                    sparsePromptEmbeddings: sparse_embeddings,
                    densePromptEmbeddings: dense_embeddings,
                    multimaskOutput: multimask_output
                );

                Tensor masks = postprocess_masks(
                    low_res_masks,
                    input_size: (batched_input[i]["image"] as Tensor).shape[^2..],
                    original_size: batched_input[i]["original_size"] as long[]
                );

                masks = masks > mask_threshold;

                outputs.Add(new Dictionary<string, Tensor>
                {
                    { "masks", masks },
                    { "iou_predictions", iou_predictions },
                    { "low_res_logits", low_res_masks }
                });
            }
            return outputs;
        }

        public Tensor postprocess_masks(Tensor masks, long[] input_size, long[] original_size)
        {
            masks = functional.interpolate(masks, [imgSize, imgSize], mode:InterpolationMode.Bilinear, align_corners: false);
            masks = masks[TensorIndex.Ellipsis, TensorIndex.Slice(0, input_size[0]), TensorIndex.Slice(0, input_size[1])];
            masks = functional.interpolate(masks, original_size, mode: InterpolationMode.Bilinear, align_corners: false);
            return masks;
        }

        public Tensor preprocess(Tensor x)
        {
            // Normalize colors
            x = (x - this.pixel_mean) / this.pixel_std;

            var shape = x.shape;
            var shapelen = shape.Length;
            var h = shape[shapelen - 2];
            var w = shape[shapelen - 1];
            var padh = imgSize - h;
            var padw = imgSize - w;
            x = functional.pad(x, (0, padw, 0, padh));

            return x;
        }
    }
}
