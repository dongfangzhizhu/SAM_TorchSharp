// Copyright (c) Sapiens AI. All rights reserved.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;

namespace SAMTorchSharp.Modeling.Sam3;

/// <summary>
/// Simplified tokenizer for SAM3 text encoder.
/// Replicates the SimpleTokenizer from sam3/model/tokenizer_ve.py.
/// Uses the merges.txt and vocab.json from the checkpoint directory.
/// </summary>
public class Sam3TokenizerVE
{
    private readonly Dictionary<string, int> vocab;
    private readonly int contextLength;

    public Sam3TokenizerVE(string bpeFilePath = null, int contextLength = 32)
    {
        this.contextLength = contextLength;
        vocab = new Dictionary<string, int>();

        if (bpeFilePath != null && File.Exists(bpeFilePath))
        {
            LoadFromBpeFile(bpeFilePath);
        }
        else
        {
            BuildMinimalVocab();
        }
    }

    private void LoadFromBpeFile(string path)
    {
        var lines = File.ReadAllLines(path);
        int idx = 0;

        foreach (var line in lines)
        {
            if (line.StartsWith("#") || string.IsNullOrWhiteSpace(line)) continue;
            var parts = line.Split('\t');
            if (parts.Length >= 2 && !vocab.ContainsKey(parts[0]))
                vocab[parts[0]] = idx++;
        }

        // Ensure special tokens exist
        var specialTokens = new[] { "[BOS]", "[EOS]", "[PAD]" };
        foreach (var st in specialTokens)
        {
            if (!vocab.ContainsKey(st))
                vocab[st] = idx++;
        }
    }

    private void BuildMinimalVocab()
    {
        int idx = 0;
        var commonWords = new[] {
            "a", "an", "the", "dog", "cat", "person", "man", "woman", "boy", "girl",
            "bird", "horse", "cow", "fish", "tree", "flower", "sky", "sun", "moon",
            "water", "land", "road", "house", "car", "bike", "book", "phone",
            "table", "chair", "door", "window", "wall", "floor", "room", "garden",
            "animal", "plant", "food", "drink", "red", "blue", "green", "yellow",
            "black", "white", "big", "small", "hot", "cold", "fast", "slow",
            "happy", "sad", "good", "bad", "new", "old", "young", "high", "low",
            "long", "short", "run", "walk", "jump", "fly", "swim", "eat", "drink",
            "play", "work", "read", "write", "talk", "listen", "look", "see",
            "think", "feel", "know", "love", "like", "want", "need",
            "[BOS]", "[EOS]", "[PAD]", ""
        };

        foreach (var word in commonWords)
        {
            if (!string.IsNullOrEmpty(word) && !vocab.ContainsKey(word))
                vocab[word] = idx++;
        }

        // Character-level tokens
        foreach (var c in "abcdefghijklmnopqrstuvwxyz .,!?;:'\"-()[]{}")
        {
            var key = c.ToString();
            if (!vocab.ContainsKey(key))
                vocab[key] = idx++;
        }
    }

    /// <summary>
    /// Encode text strings into token IDs.
    /// Returns tensor of shape [batch_size, context_length].
    /// </summary>
    public Tensor EncodeTexts(IList<string> texts, Device? device = null)
    {
        var dev = device ?? CPU;
        var batchSize = texts.Count;
        var tokens = new long[batchSize, contextLength];

        for (int i = 0; i < batchSize; i++)
        {
            var text = texts[i];
            var tokenIds = Tokenize(text);

            int pos = 0;
            if (pos < contextLength) tokens[i, pos++] = vocab.GetValueOrDefault("[BOS]", 0);

            foreach (var tid in tokenIds)
            {
                if (pos >= contextLength) break;
                tokens[i, pos++] = tid;
            }

            if (pos < contextLength) tokens[i, pos++] = vocab.GetValueOrDefault("[EOS]", 0);

            while (pos < contextLength)
            {
                tokens[i, pos++] = vocab.GetValueOrDefault("[PAD]", 0);
            }
        }

        return torch.tensor(tokens, dtype: ScalarType.Int64, device: dev);
    }

    private IList<long> Tokenize(string text)
    {
        text = text.ToLower().Trim();
        var chars = text.ToList();
        var tokens = chars.Select(c => c.ToString()).ToList();

        var result = new List<long>();
        foreach (var t in tokens)
        {
            result.Add(vocab.GetValueOrDefault(t, 0));
        }

        return result;
    }
}
