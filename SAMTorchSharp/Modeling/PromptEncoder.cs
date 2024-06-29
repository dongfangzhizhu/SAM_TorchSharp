using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SAMTorchSharp.Modeling
{
    public class PromptEncoder : Module<Tuple<Tensor, Tensor>,Tensor,Tensor,Tuple<Tensor,Tensor>>
    {
        private int embed_dim;
        private Tuple<int, int> input_image_size;
        private Tuple<int, int> image_embedding_size;
        private PositionEmbeddingRandom pe_layer;
        private int num_point_embeddings;
        private ModuleList<Embedding> point_embeddings;
        private Embedding not_a_point_embed;
        private Tuple<int, int> mask_input_size;
        private Sequential mask_downscaling;
        private Embedding no_mask_embed;
        private readonly Module<Tensor, Tensor> activation;

        public PromptEncoder(
            int embed_dim,
            Tuple<int, int> image_embedding_size,
            Tuple<int, int> input_image_size,
            int mask_in_chans,
            Func<Module<Tensor, Tensor>> activation = null,
            string name= "PromptEncoder"
        ) : base(name)
        {

            this.activation = activation == null ? nn.GELU() : activation();
            this.embed_dim = embed_dim;
            this.input_image_size = input_image_size;
            this.image_embedding_size = image_embedding_size;
            this.pe_layer = new PositionEmbeddingRandom(embed_dim / 2);

            this.num_point_embeddings = 4; // pos/neg point + 2 box corners
            Embedding[] point_embeddings = new Embedding[num_point_embeddings];
            for (int i = 0; i < this.num_point_embeddings; i++)
            {
                point_embeddings[i] = Embedding(1, embed_dim);
            }
            this.point_embeddings = new ModuleList<Embedding>(point_embeddings);
            this.not_a_point_embed = Embedding(1, embed_dim);

            this.mask_input_size = new Tuple<int, int>(4 * image_embedding_size.Item1, 4 * image_embedding_size.Item2);
        
            this.mask_downscaling = Sequential( Conv2d(1, mask_in_chans / 4, 2, 2),
                LayerNorm(mask_in_chans / 4),
                this.activation,
                Conv2d(mask_in_chans / 4, mask_in_chans, 2, 2),
                LayerNorm(mask_in_chans),
               this.activation,
                Conv2d(mask_in_chans, embed_dim, 1));

            this.no_mask_embed = Embedding(1, embed_dim);
            RegisterComponents();
        }


        public Tensor get_dense_pe()
        {
            return pe_layer.forward((image_embedding_size.Item1, image_embedding_size.Item2)).unsqueeze(0);
        }

        private Tensor _embed_points(Tensor points, Tensor labels, bool pad)
        {
            // Shift to center of pixel
            points = points.add(0.5f);
            if (pad)
            {
                var padding_point = zeros(new long[] { points.shape[0], 1, 2 }, device: points.device);
                var padding_label = ones(new long[] { labels.shape[0], 1 }, device: labels.device).mul(-1);
                points = torch.cat(new Tensor[] { points, padding_point }, dim: 1);
                labels = torch.cat(new Tensor[] { labels, padding_label }, dim: 1);
            }

            var point_embedding = pe_layer.forwardWithCoords(points, input_image_size);
            point_embedding[labels.eq(-1)] = 0.0f;
            point_embedding[labels.eq(-1)].add_(not_a_point_embed.weight);
            point_embedding[labels.eq(0)].add_(point_embeddings[0].weight);
            point_embedding[labels.eq(1)].add_(point_embeddings[1].weight);

            return point_embedding;
        }

        private Tensor _embed_boxes(Tensor boxes)
        {
            // Shift to center of pixel
            boxes = boxes.add(0.5f);
            var coords = boxes.view(-1, 2, 2);
            var corner_embedding = pe_layer.forwardWithCoords(coords, input_image_size);
            corner_embedding[TensorIndex.Colon, TensorIndex.Single(0), TensorIndex.Colon].add_(point_embeddings[2].weight);
            corner_embedding[TensorIndex.Colon, TensorIndex.Single(1), TensorIndex.Colon].add_(point_embeddings[3].weight);
            return corner_embedding;
        }

        private Tensor _embed_masks(Tensor masks)
        {
            return mask_downscaling.forward(masks);
        }

        private long _get_batch_size(Tuple<Tensor, Tensor>? points = null, Tensor boxes = null, Tensor masks = null)
        {
            if (points is not null)
            {
                return points.Item1.shape[0];
            }
            else if (boxes is not null)
            {
                return boxes.shape[0];
            }
            else if (masks is not null)
            {
                return masks.shape[0];
            }
            else
            {
                return 1;
            }
        }

        private Device _get_device()
        {
            return point_embeddings[0].weight.device;
        }

        public override Tuple<Tensor, Tensor> forward(Tuple<Tensor, Tensor>? points = null, Tensor? boxes = null, Tensor? masks = null)
        {
            var bs = _get_batch_size(points, boxes, masks);
            var sparse_embeddings = empty(new long[] { bs, 0, embed_dim }, dtype: ScalarType.Float32, device: _get_device());

            if (points is not null)
            {
                var point_embeddings = _embed_points(points.Item1, points.Item2, pad: (boxes is null));
                sparse_embeddings = torch.cat(new Tensor[] { sparse_embeddings, point_embeddings }, dim: 1);
            }

            if (boxes is not null)
            {
                var box_embeddings = _embed_boxes(boxes);
                sparse_embeddings = torch.cat(new Tensor[] { sparse_embeddings, box_embeddings }, dim: 1);
            }

            var dense_embeddings = masks is not null ? _embed_masks(masks) : no_mask_embed.weight.view(new long[] { 1, -1, 1, 1 }).expand(new long[] { bs, -1, image_embedding_size.Item1, image_embedding_size.Item2 }, true);

            return new Tuple<Tensor, Tensor>(sparse_embeddings, dense_embeddings);
        }
    }
    public class PositionEmbeddingRandom : Module<(long,long),Tensor>
    {
        private Parameter positionalEncodingGaussianMatrix;
        private float scale;

        public PositionEmbeddingRandom(int numPosFeats = 64, float? scale = null,string name= "PositionEmbeddingRandom") : base(name)
        {
            if (scale == null || scale.Value <= 0.0f)
            {
                this.scale = 1.0f;
            }
            else
            {
                this.scale = scale.Value;
            }

            var parm = new Parameter(torch.rand(new long[] { 2, numPosFeats }).mul(this.scale));
            this.positionalEncodingGaussianMatrix = parm;
            register_parameter("positional_encoding_gaussian_matrix", this.positionalEncodingGaussianMatrix);
            RegisterComponents();
        }

        private Tensor _peEncoding(Tensor coords)
        {
            // Assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
            coords = 2 * coords - 1;
            coords = coords.matmul(positionalEncodingGaussianMatrix);
            coords = coords.mul((float)(2 * Math.PI));
            // Outputs d_1 x ... x d_n x C shape
            var sinCoords = coords.sin();
            var cosCoords = coords.cos();
            return torch.cat(new Tensor[] { sinCoords, cosCoords }, -1);
        }

        public override Tensor forward((long,long) input)
        {
            var (h,w) = input;
            var device = positionalEncodingGaussianMatrix.device;
            var grid = ones(new long[] { h, w }, dtype: ScalarType.Float32, device: device);
            var y_embed = torch.cumsum(grid, 0).sub(0.5f);
            var x_embed = torch.cumsum(grid, 1).sub(0.5f);
            y_embed = y_embed.div(h);
            x_embed = x_embed.div(w);
            var pe = _peEncoding(torch.stack(new Tensor[] { x_embed, y_embed },dim:-1));
            return pe.permute(new long[] { 2, 0, 1 }); // C x H x W
        }
        public Tensor forwardWithCoords(Tensor coordsInput, Tuple<int, int> imageSize)
        {
            Tensor coords = coordsInput.clone();
            coords.index(TensorIndex.Ellipsis, TensorIndex.Single(0)).div_(imageSize.Item2); // Divide by image width
            coords.index(TensorIndex.Ellipsis, TensorIndex.Single(1)).div_(imageSize.Item1);
            return _peEncoding(coords.to(ScalarType.Float32)); // 指定数据类型为浮点型
        }
    }
}
