// Copyright (c) Sapiens AI. All rights reserved.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SAMTorchSharp.Modeling.Sam3;

/// <summary>
/// Simplified BPE tokenizer for SAM3 CLIP-style text encoding.
/// Adapted from open_clip / SAM3 tokenizer_ve.py.
/// </summary>
public class Sam3TokenizerVE
{
    public const int DefaultContextLength = 77;

    private readonly Dictionary<string, int> encoder;
    private readonly Dictionary<string, int> bpeRanks;
    private readonly List<string> specialTokens;
    private readonly int sotTokenId;
    private readonly int eotTokenId;
    private readonly int vocabSize;
    private readonly int contextLength;
    private readonly Regex pat;
    private readonly Dictionary<byte, string> byteEncoder;
    private readonly Dictionary<string, byte> byteDecoder;

    public Sam3TokenizerVE(string? bpeFilePath = null, int contextLength = DefaultContextLength)
    {
        this.contextLength = contextLength;

        var (byteEnc, byteDec) = BytesToUnicode();
        byteEncoder = byteEnc;
        byteDecoder = byteDec;

        var vocab = new List<string>(byteEncoder.Values.Distinct());

        var baseVocab = vocab.ToList();
        foreach (var v in baseVocab)
        {
            if (!v.EndsWith("</w>"))
                vocab.Add(v + "</w>");
        }

        bpeRanks = new Dictionary<string, int>();
        int rank = 0;
        if (!string.IsNullOrEmpty(bpeFilePath) && File.Exists(bpeFilePath))
        {
            var lines = File.ReadAllLines(bpeFilePath);
            var limit = Math.Min(lines.Length, 49152 - 256 - 2 + 1);
            for (int i = 1; i < limit; i++)
            {
                var parts = lines[i].Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length == 2)
                {
                    vocab.Add(parts[0] + parts[1]);
                    bpeRanks[parts[0] + " " + parts[1]] = rank++;
                }
            }
        }

        specialTokens = new List<string> { "<start_of_text>", "<end_of_text>" };
        vocab.AddRange(specialTokens);

        encoder = vocab.Distinct().Select((v, i) => (v, i)).ToDictionary(x => x.v, x => x.i);
        vocabSize = encoder.Count;
        sotTokenId = encoder[specialTokens[0]];
        eotTokenId = encoder[specialTokens[1]];

        var specialPattern = string.Join("|", specialTokens.Select(t => Regex.Escape(t)));
        pat = new Regex(
            $"{specialPattern}|'s|'t|'re|'ve|'m|'ll|'d|[\\p{{L}}]+|[\\p{{N}}]|[^\\s\\p{{L}}\\p{{N}}]+",
            RegexOptions.Compiled | RegexOptions.IgnoreCase);
    }

    private (Dictionary<byte, string>, Dictionary<string, byte>) BytesToUnicode()
    {
        var bs = new List<byte>();
        var cs = new List<byte>();
        for (int b = '!'; b <= '~'; b++) bs.Add((byte)b);
        for (int b = 0xa1; b <= 0xac; b++) bs.Add((byte)b);
        for (int b = 0xae; b <= 0xff; b++) bs.Add((byte)b);

        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!bs.Contains((byte)b))
            {
                bs.Add((byte)b);
                cs.Add((byte)(256 + n));
                n++;
            }
        }

        var enc = new Dictionary<byte, string>();
        var dec = new Dictionary<string, byte>();
        for (int i = 0; i < cs.Count; i++)
        {
            var ch = (char)cs[i];
            enc[(byte)i] = ch.ToString();
            dec[ch.ToString()] = (byte)i;
        }
        return (enc, dec);
    }

    private string Bpe(string token)
    {
        if (encoder.ContainsKey(token))
            return token;

        // Build word as array of strings (tokens)
        var word = new List<string>();
        for (int i = 0; i < token.Length - 1; i++)
            word.Add(token[i].ToString());
        word.Add(token[token.Length - 1] + "</w>");

        var pairs = GetPairs(word);
        if (!pairs.Any())
            return token + "</w>";

        while (true)
        {
            var bigram = pairs.FirstOrDefault(p => bpeRanks.ContainsKey(p));
            if (!bpeRanks.ContainsKey(bigram))
                break;

            var parts = bigram.Split(' ');
            var first = parts[0];
            var second = parts[1];

            var newWord = new List<string>();
            int i = 0;
            while (i < word.Count)
            {
                var idx = word.IndexOf(first, i);
                if (idx == -1)
                {
                    newWord.AddRange(word.GetRange(i, word.Count - i));
                    break;
                }
                newWord.AddRange(word.GetRange(i, idx - i));
                i = idx;
                if (i < word.Count && word[i] == first && i + 1 < word.Count && word[i + 1] == second)
                {
                    newWord.Add(first + second);
                    i += 2;
                }
                else
                {
                    newWord.Add(word[i]);
                    i++;
                }
            }

            word = newWord;
            if (word.Count == 1)
                break;
            pairs = GetPairs(word);
        }

        return string.Join(" ", word);
    }

    private HashSet<string> GetPairs(List<string> word)
    {
        var pairs = new HashSet<string>();
        for (int i = 0; i < word.Count - 1; i++)
            pairs.Add($"{word[i]} {word[i + 1]}");
        return pairs;
    }

    private string CleanText(string text)
    {
        text = text.Replace("_", " ");
        text = Regex.Replace(text, @"[\p{P}]", "");
        text = text.ToLowerInvariant();
        text = Regex.Replace(text, @"\s+", " ").Trim();
        return text;
    }

    private List<int> Encode(string text)
    {
        text = CleanText(text);
        var matches = pat.Matches(text);
        var bpeTokens = new List<int>();
        foreach (Match match in matches)
        {
            var token = match.Value;
            var encodedBytes = Encoding.UTF8.GetBytes(token);
            var unicodeString = new string(encodedBytes.Select(b => (char)b).ToArray());
            var bpeToken = Bpe(unicodeString);
            foreach (var t in bpeToken.Split(' ', StringSplitOptions.RemoveEmptyEntries))
            {
                if (encoder.TryGetValue(t, out int id))
                    bpeTokens.Add(id);
            }
        }
        return bpeTokens;
    }

    public Tensor EncodeTexts(IList<string> texts, int? contextLength = null)
    {
        var cl = contextLength ?? this.contextLength;
        var allTokens = texts.Select(text =>
        {
            var tokens = new List<int> { sotTokenId };
            tokens.AddRange(Encode(text));
            tokens.Add(eotTokenId);
            return tokens;
        }).ToList();

        var result = torch.zeros(new long[] { (long)allTokens.Count, (long)cl }, dtype: int64);
        for (int i = 0; i < allTokens.Count; i++)
        {
            var tokens = allTokens[i];
            if (tokens.Count > cl)
            {
                tokens = tokens.Take(cl).ToList();
                tokens[cl - 1] = eotTokenId;
            }
            for (int j = 0; j < tokens.Count && j < cl; j++)
            {
                result[i, j] = (long)tokens[j];
            }
        }
        return result;
    }

    public int VocabSize => vocabSize;
    public int SotTokenId => sotTokenId;
    public int EotTokenId => eotTokenId;
    public int ContextLength => contextLength;
}

/// <summary>
/// Residual Attention Block for CLIP-style text transformer.
/// Equivalent to SAM3's ResidualAttentionBlock.
/// TorchSharp MultiheadAttention uses [seq, batch, dim] format (not batch_first).
/// </summary>
public class Sam3ResidualAttentionBlock : Module
{
    private readonly MultiheadAttention attn;
    private readonly LayerNorm ln1;
    private readonly LayerNorm ln2;
    private readonly Linear c_fc;
    private readonly Linear c_proj1;
    private readonly Module<Tensor, Tensor> gelu;

    public Sam3ResidualAttentionBlock(
        int dModel,
        int nHead,
        float mlpRatio = 4.0f)
        : base(nameof(Sam3ResidualAttentionBlock))
    {
        var mlpWidth = (int)(dModel * mlpRatio);
        
        // TorchSharp MultiheadAttention: [seq, batch, dim] format
        attn = MultiheadAttention(dModel, nHead, dropout: 0.0f);
        ln1 = LayerNorm(dModel);
        ln2 = LayerNorm(dModel);
        
        c_fc = Linear(dModel, mlpWidth);
        gelu = GELU();
        c_proj1 = Linear(mlpWidth, dModel);
    }

    public Tensor forward(Tensor q_x, Tensor? attn_mask = null)
    {
        // TorchSharp MA expects [seq, batch, dim], so transpose from [batch, seq, dim]
        var q_normed = ln1.forward(q_x);
        var qTransposed = q_normed.transpose(0, 1);
        
        var attnOut = attn.forward(qTransposed, qTransposed, qTransposed,
            key_padding_mask: null, need_weights: false, attn_mask: attn_mask).Item1;
        // attnOut is [seq, batch, dim], transpose back to [batch, seq, dim]
        var attnScaled = attnOut.transpose(0, 1);
        
        var x1 = q_x + attnScaled;

        // MLP
        var mlpIn = ln2.forward(x1);
        var mlpOut = c_proj1.forward(gelu.forward(c_fc.forward(mlpIn)));
        var x2 = x1 + mlpOut;

        return x2;
    }
}

/// <summary>
/// CLIP-style Transformer with residual attention blocks.
/// </summary>
public class Sam3TextTransformer : Module
{
    private readonly List<Sam3ResidualAttentionBlock> resblocks;
    private readonly LayerNorm lnFinal;

    public Sam3TextTransformer(
        int width = 1024,
        int layers = 24,
        int heads = 16,
        float mlpRatio = 4.0f)
        : base(nameof(Sam3TextTransformer))
    {
        resblocks = new List<Sam3ResidualAttentionBlock>();
        for (int i = 0; i < layers; i++)
        {
            resblocks.Add(new Sam3ResidualAttentionBlock(width, heads, mlpRatio));
        }
        lnFinal = LayerNorm(width);
    }

    public Tensor forward(Tensor x, Tensor? attn_mask = null)
    {
        foreach (var rb in resblocks)
        {
            x = rb.forward(x, attn_mask);
        }
        x = lnFinal.forward(x);
        return x;
    }
}

/// <summary>
/// CLIP Text Transformer with token embedding, positional encoding, and projection.
/// </summary>
public class Sam3CLIPTextTransformer : Module
{
    private readonly Embedding tokenEmbedding;
    private readonly Parameter positionalEmbedding;
    private readonly Sam3TextTransformer transformer;
    private readonly Module<Tensor, Tensor> lnFinal;
    private readonly Linear? textProjection;
    private readonly Tensor? causalMask;
    private readonly int contextLength;
    private readonly int width;
    private readonly int outputDim;

    public Sam3CLIPTextTransformer(
        int contextLength = 32,
        int vocabSize = 49408,
        int width = 1024,
        int heads = 16,
        int layers = 24,
        float mlpRatio = 4.0f,
        int outputDim = 1024,
        bool noCausalMask = false,
        bool useLnPost = true,
        bool useBiasProj = true)
        : base(nameof(Sam3CLIPTextTransformer))
    {
        this.contextLength = contextLength;
        this.width = width;
        this.outputDim = outputDim;

        tokenEmbedding = Embedding(vocabSize, width);
        positionalEmbedding = Parameter(zeros(new long[] { (long)contextLength, (long)width }));
        
        transformer = new Sam3TextTransformer(width, layers, heads, mlpRatio);
        lnFinal = useLnPost ? (Module<Tensor, Tensor>)LayerNorm(width) : Identity();

        if (!noCausalMask)
        {
            var mask = full(new long[] { (long)contextLength, (long)contextLength }, double.NegativeInfinity);
            causalMask = mask.triu(diagonal: 1);
        }

        if (useBiasProj)
        {
            textProjection = Linear(width, outputDim);
        }
    }

    public Tensor forward(Tensor text)
    {
        var (pooled, _) = ForwardInternal(text);
        return pooled;
    }

    public Tuple<Tensor, Tensor> ForwardWithTokens(Tensor text)
    {
        return ForwardInternal(text);
    }

    private Tuple<Tensor, Tensor> ForwardInternal(Tensor text)
    {
        var seqLen = text.size(1);
        
        var x = tokenEmbedding.forward(text);
        x = x + positionalEmbedding.narrow(0, 0, seqLen).unsqueeze(0);

        Tensor? attnMask = null;
        if (causalMask is not null)
        {
            attnMask = causalMask.narrow(0, 0, seqLen).narrow(1, 0, seqLen);
        }

        x = transformer.forward(x, attnMask);
        x = ((Module<Tensor, Tensor>)lnFinal).forward(x);

        var pooled = x.select(-2, seqLen - 1);

        if (textProjection is not null)
        {
            pooled = textProjection.forward(pooled);
        }

        return Tuple.Create(pooled, x);
    }
}

/// <summary>
/// VE Text Encoder for SAM3.
/// Wraps the CLIP text transformer and adds a linear resizer.
/// </summary>
public class Sam3VETextEncoder : Module
{
    private readonly Sam3CLIPTextTransformer encoder;
    private readonly Linear resizer;
    private readonly Sam3TokenizerVE? tokenizer;
    private readonly int contextLength;

    public Sam3VETextEncoder(
        int dModel = 1024,
        int textWidth = 1024,
        int textHeads = 16,
        int textLayers = 24,
        int contextLength = 32,
        int vocabSize = 49408,
        bool useLnPost = true,
        Sam3TokenizerVE? tokenizer = null)
        : base(nameof(Sam3VETextEncoder))
    {
        this.contextLength = contextLength;
        this.tokenizer = tokenizer;

        encoder = new Sam3CLIPTextTransformer(
            contextLength: contextLength,
            vocabSize: vocabSize,
            width: textWidth,
            heads: textHeads,
            layers: textLayers,
            outputDim: textWidth,
            useLnPost: useLnPost);

        resizer = Linear(textWidth, dModel);
    }

    /// <summary>
    /// Forward pass for text strings.
    /// Returns (text_attention_mask, text_memory_resized, text_inputs_embeds).
    /// </summary>
    public Tuple<Tensor, Tensor, Tensor> ForwardText(IList<string> texts, Device? device = null)
    {
        var dev = device ?? CPU;
        
        Tensor tokenized;
        if (tokenizer is not null)
        {
            tokenized = tokenizer.EncodeTexts(texts).to(dev);
        }
        else
        {
            tokenized = zeros(new long[] { (long)texts.Count, (long)contextLength }, dtype: int64, device: dev);
        }

        var (pooled, tokens) = encoder.ForwardWithTokens(tokenized);
        // tokens: [batch, seq_len, text_width]

        var textAttentionMask = (tokenized != 0).logical_not();

        // [seq_len, batch, text_width] -> [seq_len, batch, dModel]
        var textMemoryResized = resizer.forward(tokens.transpose(0, 1));

        // [batch, seq_len, dModel]
        var inputsEmbeds = tokens.transpose(0, 1);

        return Tuple.Create(textAttentionMask, textMemoryResized, inputsEmbeds);
    }

    public Tuple<Tensor, Tensor, Tensor> ForwardEncoded(Tensor tokenized, Device? device = null)
    {
        var dev = device ?? CPU;
        tokenized = tokenized.to(dev);

        var (pooled, tokens) = encoder.ForwardWithTokens(tokenized);

        var textAttentionMask = (tokenized != 0).logical_not();
        var textMemoryResized = resizer.forward(tokens.transpose(0, 1));
        var inputsEmbeds = tokens.transpose(0, 1);

        return Tuple.Create(textAttentionMask, textMemoryResized, inputsEmbeds);
    }
}
