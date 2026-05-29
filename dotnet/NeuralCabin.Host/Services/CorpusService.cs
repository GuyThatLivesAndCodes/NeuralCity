using NeuralCabin.Core.Models;
using NeuralCabin.Host.Ipc;
using NeuralCabin.Host.State;

namespace NeuralCabin.Host.Services;

/// <summary>
/// Corpus management (port of <c>set_corpus</c> / <c>get_corpus</c> /
/// <c>corpus_stats</c>). Feed-forward corpora are fully handled; next-token text
/// / pairs are stored now, with vocabulary building and stats arriving with the
/// tokenizer port. Transformer corpora await the transformer engine.
/// </summary>
public sealed class CorpusService
{
    private readonly AppState _state;

    public CorpusService(AppState state) => _state = state;

    public CorpusStats Set(SetCorpusRequest request)
    {
        var net = _state.GetNetwork(request.NetworkId) ?? throw new CommandException("Network not found");

        Corpus corpus;
        if (net.Kind == NetworkKinds.Feedforward)
        {
            var ff = request.Feedforward
                ?? throw new CommandException("Feedforward network requires a feedforward corpus");
            if (ff.InDim != net.InputDim)
                throw new CommandException($"corpus in_dim {ff.InDim} doesn't match network input_dim {net.InputDim}");
            if (ff.OutDim != net.OutputDim)
                throw new CommandException($"corpus out_dim {ff.OutDim} doesn't match network output_dim {net.OutputDim}");
            if (ff.Features.Count != ff.Rows * ff.InDim)
                throw new CommandException($"features length {ff.Features.Count} != rows*in_dim ({ff.Rows}*{ff.InDim})");
            if (ff.Targets.Count != ff.Rows * ff.OutDim)
                throw new CommandException($"targets length {ff.Targets.Count} != rows*out_dim ({ff.Rows}*{ff.OutDim})");

            corpus = new Corpus
            {
                NetworkId = net.Id, Kind = net.Kind, UpdatedAt = DateTimeOffset.UtcNow, Feedforward = ff,
            };
        }
        else if (net.Kind == NetworkKinds.NextToken)
        {
            corpus = new Corpus
            {
                NetworkId = net.Id, Kind = net.Kind, UpdatedAt = DateTimeOffset.UtcNow,
                Text = request.Text, Pairs = request.Pairs, Stage = request.Stage ?? "pretrain",
            };
        }
        else
        {
            throw new NotYetMigratedException("set_corpus (transformer)");
        }

        _state.Corpora[net.Id] = corpus;
        Persistence.SaveState(_state);
        return BuildStats(corpus, net);
    }

    public Corpus? Get(string networkId) =>
        _state.Corpora.TryGetValue(networkId, out var c) ? c : null;

    public CorpusStats? StatsFor(string networkId)
    {
        if (!_state.Corpora.TryGetValue(networkId, out var corpus)) return null;
        var net = _state.GetNetwork(networkId);
        return net is null ? null : BuildStats(corpus, net);
    }

    private static CorpusStats BuildStats(Corpus corpus, Network net)
    {
        if (net.Kind == NetworkKinds.Feedforward)
        {
            var ff = corpus.Feedforward!;
            return new CorpusStats
            {
                Kind = net.Kind, Rows = ff.Rows, InDim = ff.InDim, OutDim = ff.OutDim,
                TrainingExamples = ff.Rows, VocabReady = true, ModelReady = true,
            };
        }

        // next_token: vocabulary + materialized model arrive with the tokenizer port.
        return new CorpusStats
        {
            Kind = net.Kind,
            Stage = corpus.Stage,
            TextChars = corpus.Text?.Length,
            PairCount = corpus.Pairs?.Count,
            VocabReady = false,
            ModelReady = false,
        };
    }
}
