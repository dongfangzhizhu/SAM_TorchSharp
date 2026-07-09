using SAMTorchSharp.Modeling.Sam1;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SAMTorchSharp.Modeling.Sam2
{
    /// <summary>
    /// 对应 sam2/modeling/sam/prompt_encoder.py: PromptEncoder。
    /// 结构与 SAM1 版本基本一致（复用 Sam1.PositionEmbeddingRandom），主要区别在于
    /// _embed_points 使用 torch.where 按 label(-1/0/1/2/3) 分支赋值，而不是原地索引累加；
    /// 数值结果等价，这里按 Python 端写法实现以保持逐行可对照。
    /// </summary>
    public class PromptEncoder : Module<Tuple<Tensor, Tensor>?, Tensor?, Tensor?, Tuple<Tensor, Tensor>>
    {
        public readonly int embed_dim;
        private readonly (int H, int W) input_image_size;
        public readonly (int H, int W) image_embedding_size;
        public readonly PositionEmbeddingRandom pe_layer;
        private readonly int num_point_embeddings;
        public readonly ModuleList<Embedding> point_embeddings;
        public readonly Embedding not_a_point_embed;
        public readonly (int H, int W) mask_input_size;
        public readonly Sequential mask_downscaling;
        public readonly Embedding no_mask_embed;

        public PromptEncoder(
            int embed_dim,
            (int, int) image_embedding_size,
            (int, int) input_image_size,
            int mask_in_chans,
            Func<Module<Tensor, Tensor>>? activation = null,
            string name = "PromptEncoder") : base(name)
        {
            var act = activation is null ? GELU() : activation();

            this.embed_dim = embed_dim;
            this.input_image_size = input_image_size;
            this.image_embedding_size = image_embedding_size;
            pe_layer = new PositionEmbeddingRandom(embed_dim / 2);

            num_point_embeddings = 4; // pos/neg point + 2 box corners
            var embeddings = new Embedding[num_point_embeddings];
            for (int i = 0; i < num_point_embeddings; i++)
            {
                embeddings[i] = Embedding(1, embed_dim);
            }
            point_embeddings = new ModuleList<Embedding>(embeddings);
            not_a_point_embed = Embedding(1, embed_dim);

            mask_input_size = (4 * image_embedding_size.Item1, 4 * image_embedding_size.Item2);

            mask_downscaling = Sequential(
                Conv2d(1, mask_in_chans / 4, kernelSize: 2, stride: 2),
                new LayerNorm2d(mask_in_chans / 4),
                act,
                Conv2d(mask_in_chans / 4, mask_in_chans, kernelSize: 2, stride: 2),
                new LayerNorm2d(mask_in_chans),
                act,
                Conv2d(mask_in_chans, embed_dim, kernelSize: 1));

            no_mask_embed = Embedding(1, embed_dim);

            RegisterComponents();
        }

        public Tensor get_dense_pe()
        {
            return pe_layer.forward((image_embedding_size.H, image_embedding_size.W)).unsqueeze(0);
        }

        private Tensor _embed_points(Tensor points, Tensor labels, bool pad)
        {
            points = points + 0.5f;
            if (pad)
            {
                Tensor paddingPoint = zeros(new long[] { points.shape[0], 1, 2 }, device: points.device);
                Tensor paddingLabel = -ones(new long[] { labels.shape[0], 1 }, device: labels.device);
                points = cat(new[] { points, paddingPoint }, dim: 1);
                labels = cat(new[] { labels, paddingLabel }, dim: 1);
            }

            Tensor pointEmbedding = pe_layer.forwardWithCoords(points, Tuple.Create(input_image_size.H, input_image_size.W));

            pointEmbedding = where(
                (labels == -1).unsqueeze(-1),
                zeros_like(pointEmbedding) + not_a_point_embed.weight!,
                pointEmbedding);
            pointEmbedding = where(
                (labels == 0).unsqueeze(-1),
                pointEmbedding + point_embeddings[0].weight!,
                pointEmbedding);
            pointEmbedding = where(
                (labels == 1).unsqueeze(-1),
                pointEmbedding + point_embeddings[1].weight!,
                pointEmbedding);
            pointEmbedding = where(
                (labels == 2).unsqueeze(-1),
                pointEmbedding + point_embeddings[2].weight!,
                pointEmbedding);
            pointEmbedding = where(
                (labels == 3).unsqueeze(-1),
                pointEmbedding + point_embeddings[3].weight!,
                pointEmbedding);

            return pointEmbedding;
        }

        private Tensor _embed_boxes(Tensor boxes)
        {
            boxes = boxes + 0.5f;
            Tensor coords = boxes.reshape(-1, 2, 2);
            Tensor cornerEmbedding = pe_layer.forwardWithCoords(coords, Tuple.Create(input_image_size.H, input_image_size.W));
            cornerEmbedding[TensorIndex.Colon, TensorIndex.Single(0), TensorIndex.Colon] =
                cornerEmbedding[TensorIndex.Colon, TensorIndex.Single(0), TensorIndex.Colon] + point_embeddings[2].weight!;
            cornerEmbedding[TensorIndex.Colon, TensorIndex.Single(1), TensorIndex.Colon] =
                cornerEmbedding[TensorIndex.Colon, TensorIndex.Single(1), TensorIndex.Colon] + point_embeddings[3].weight!;
            return cornerEmbedding;
        }

        private Tensor _embed_masks(Tensor masks)
        {
            return mask_downscaling.forward(masks);
        }

        private long _get_batch_size(Tuple<Tensor, Tensor>? points, Tensor? boxes, Tensor? masks)
        {
            if (points is not null) return points.Item1.shape[0];
            if (boxes is not null) return boxes.shape[0];
            if (masks is not null) return masks.shape[0];
            return 1;
        }

        private Device _get_device()
        {
            return point_embeddings[0].weight!.device;
        }

        public override Tuple<Tensor, Tensor> forward(Tuple<Tensor, Tensor>? points, Tensor? boxes, Tensor? masks)
        {
            long bs = _get_batch_size(points, boxes, masks);
            Tensor sparseEmbeddings = empty(new long[] { bs, 0, embed_dim }, device: _get_device());

            if (points is not null)
            {
                Tensor pointEmbeddings = _embed_points(points.Item1, points.Item2, pad: boxes is null);
                sparseEmbeddings = cat(new[] { sparseEmbeddings, pointEmbeddings }, dim: 1);
            }

            if (boxes is not null)
            {
                Tensor boxEmbeddings = _embed_boxes(boxes);
                sparseEmbeddings = cat(new[] { sparseEmbeddings, boxEmbeddings }, dim: 1);
            }

            Tensor denseEmbeddings = masks is not null
                ? _embed_masks(masks)
                : no_mask_embed.weight!.reshape(1, -1, 1, 1).expand(bs, -1, image_embedding_size.H, image_embedding_size.W);

            return Tuple.Create(sparseEmbeddings, denseEmbeddings);
        }
    }
}
