using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SAMTorchSharp.Modeling.Sam2
{
    /// <summary>
    /// 对应 sam2/modeling/memory_attention.py: MemoryAttentionLayer。
    /// self_attn 对当前帧特征做自注意力，cross_attn_image 让当前帧 attend 到历史 memory，
    /// 最后过一个标准 FFN。self_attn/cross_attn_image 通常都是 RoPEAttention。
    /// </summary>
    public class MemoryAttentionLayer : Module
    {
        public readonly long d_model;
        public readonly long dim_feedforward;
        public readonly double dropout_value;
        public readonly RoPEAttention self_attn;
        public readonly RoPEAttention cross_attn_image;
        public readonly Linear linear1;
        public readonly Dropout dropout;
        public readonly Linear linear2;
        public readonly LayerNorm norm1;
        public readonly LayerNorm norm2;
        public readonly LayerNorm norm3;
        public readonly Dropout dropout1;
        public readonly Dropout dropout2;
        public readonly Dropout dropout3;
        private readonly Func<Tensor, Tensor> activation;
        public readonly bool pos_enc_at_attn;
        public readonly bool pos_enc_at_cross_attn_queries;
        public readonly bool pos_enc_at_cross_attn_keys;

        public MemoryAttentionLayer(
            string activation,
            RoPEAttention crossAttention,
            long dModel,
            long dimFeedforward,
            double dropout,
            bool posEncAtAttn,
            bool posEncAtCrossAttnKeys,
            bool posEncAtCrossAttnQueries,
            RoPEAttention selfAttention,
            string name = "MemoryAttentionLayer") : base(name)
        {
            d_model = dModel;
            dim_feedforward = dimFeedforward;
            dropout_value = dropout;
            self_attn = selfAttention;
            cross_attn_image = crossAttention;

            linear1 = Linear(dModel, dimFeedforward);
            this.dropout = Dropout(dropout);
            linear2 = Linear(dimFeedforward, dModel);

            norm1 = LayerNorm(dModel);
            norm2 = LayerNorm(dModel);
            norm3 = LayerNorm(dModel);
            dropout1 = Dropout(dropout);
            dropout2 = Dropout(dropout);
            dropout3 = Dropout(dropout);

            this.activation = activation switch
            {
                "relu" => (x => functional.relu(x)),
                "gelu" => (x => functional.gelu(x)),
                _ => throw new ArgumentException($"未知激活函数: {activation}")
            };

            pos_enc_at_attn = posEncAtAttn;
            pos_enc_at_cross_attn_queries = posEncAtCrossAttnQueries;
            pos_enc_at_cross_attn_keys = posEncAtCrossAttnKeys;

            RegisterComponents();
        }

        private Tensor ForwardSa(Tensor tgt, Tensor queryPos)
        {
            Tensor tgt2 = norm1.forward(tgt);
            Tensor q = pos_enc_at_attn ? tgt2 + queryPos : tgt2;
            Tensor k = q;
            tgt2 = self_attn.forward(q, k, tgt2, 0);
            tgt = tgt + dropout1.forward(tgt2);
            return tgt;
        }

        private Tensor ForwardCa(Tensor tgt, Tensor memory, Tensor queryPos, Tensor pos, long numKExcludeRope)
        {
            Tensor tgt2 = norm2.forward(tgt);
            Tensor q = pos_enc_at_cross_attn_queries ? tgt2 + queryPos : tgt2;
            Tensor k = pos_enc_at_cross_attn_keys ? memory + pos : memory;
            tgt2 = cross_attn_image.forward(q, k, memory, numKExcludeRope);
            tgt = tgt + dropout2.forward(tgt2);
            return tgt;
        }

        public Tensor forward(Tensor tgt, Tensor memory, Tensor pos, Tensor queryPos, long numKExcludeRope = 0)
        {
            tgt = ForwardSa(tgt, queryPos);
            tgt = ForwardCa(tgt, memory, queryPos, pos, numKExcludeRope);

            Tensor tgt2 = norm3.forward(tgt);
            tgt2 = linear2.forward(dropout.forward(activation(linear1.forward(tgt2))));
            tgt = tgt + dropout3.forward(tgt2);
            return tgt;
        }
    }

    /// <summary>
    /// 对应 sam2/modeling/memory_attention.py: MemoryAttention。
    /// 堆叠若干 MemoryAttentionLayer，将当前帧特征（curr）与历史 memory 融合。
    /// batch_first=true 时内部把 (seq,batch,dim) 转成 (batch,seq,dim) 处理再转回来。
    /// </summary>
    public class MemoryAttention : Module
    {
        public readonly long d_model;
        public readonly ModuleList<MemoryAttentionLayer> layers;
        public readonly int num_layers;
        public readonly LayerNorm norm;
        public readonly bool pos_enc_at_input;
        public readonly bool batch_first;

        public MemoryAttention(
            long dModel,
            bool posEncAtInput,
            MemoryAttentionLayer layer,
            int numLayers,
            bool batchFirst = true,
            string name = "MemoryAttention") : base(name)
        {
            d_model = dModel;
            layers = new ModuleList<MemoryAttentionLayer>();
            for (int i = 0; i < numLayers; i++)
            {
                layers.Add(i == 0 ? layer : CloneLayer(layer));
            }
            num_layers = numLayers;
            norm = LayerNorm(dModel);
            pos_enc_at_input = posEncAtInput;
            batch_first = batchFirst;

            RegisterComponents();
        }

        private static MemoryAttentionLayer CloneLayer(MemoryAttentionLayer template)
        {
            // 对应 Python 端 get_clones(copy.deepcopy)：结构相同的新实例，随后由 load_safetensors
            // 用真实权重逐层覆盖，构造参数在此不影响最终数值结果。
            var selfAttn = new RoPEAttention(
                embeddingDim: template.self_attn.embedding_dim,
                numHeads: template.self_attn.num_heads,
                downsampleRate: template.self_attn.embedding_dim / template.self_attn.internal_dim,
                ropeTheta: 10000.0,
                featSizes: (64, 64));
            var crossAttn = new RoPEAttention(
                embeddingDim: template.cross_attn_image.embedding_dim,
                numHeads: template.cross_attn_image.num_heads,
                downsampleRate: template.cross_attn_image.embedding_dim / template.cross_attn_image.internal_dim,
                kvInDim: template.cross_attn_image.kv_in_dim,
                ropeTheta: 10000.0,
                featSizes: (64, 64),
                ropeKRepeat: true);

            return new MemoryAttentionLayer(
                activation: "relu",
                crossAttention: crossAttn,
                dModel: template.d_model,
                dimFeedforward: template.dim_feedforward,
                dropout: template.dropout_value,
                posEncAtAttn: template.pos_enc_at_attn,
                posEncAtCrossAttnKeys: template.pos_enc_at_cross_attn_keys,
                posEncAtCrossAttnQueries: template.pos_enc_at_cross_attn_queries,
                selfAttention: selfAttn);
        }

        /// <summary>
        /// curr/curr_pos/memory/memory_pos 形状均为 (seq, batch, dim)（与 Python 端签名一致）。
        /// </summary>
        public Tensor forward(Tensor curr, Tensor memory, Tensor? currPos, Tensor? memoryPos, long numObjPtrTokens = 0)
        {
            if (curr.shape[1] != memory.shape[1])
            {
                throw new ArgumentException("curr 和 memory 的 batch 维必须一致");
            }

            Tensor output = curr;
            if (pos_enc_at_input && currPos is not null)
            {
                output = output + 0.1 * currPos;
            }

            Tensor curCurrPos = currPos!;
            Tensor curMemory = memory;
            Tensor curMemoryPos = memoryPos!;

            if (batch_first)
            {
                output = output.transpose(0, 1);
                curCurrPos = curCurrPos.transpose(0, 1);
                curMemory = curMemory.transpose(0, 1);
                curMemoryPos = curMemoryPos.transpose(0, 1);
            }

            for (int i = 0; i < layers.Count; i++)
            {
                output = layers[i].forward(output, curMemory, curMemoryPos, curCurrPos, numObjPtrTokens);
            }

            Tensor normedOutput = norm.forward(output);

            if (batch_first)
            {
                normedOutput = normedOutput.transpose(0, 1);
            }

            return normedOutput;
        }
    }
}
