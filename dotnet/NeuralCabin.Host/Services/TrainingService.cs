using System.Collections.Concurrent;
using System.Diagnostics;
using NeuralCabin.Core.Ipc;
using NeuralCabin.Core.Models;
using NeuralCabin.Host.Ipc;
using NeuralCabin.Host.State;
using NeuralCabin.Engine;

namespace NeuralCabin.Host.Services;

/// <summary>
/// Feed-forward training (port of <c>start_training</c> / <c>run_training_loop</c>).
/// Runs each job on a background task, streaming <c>training_update</c> events
/// per epoch and a terminal <c>training_finished</c> / <c>training_error</c>.
/// Stop keeps the in-progress weights; Abort rolls back to the pre-training
/// snapshot. Next-token / transformer training is added in later slices.
/// </summary>
public sealed class TrainingService
{
    private sealed class Handle
    {
        public volatile bool Cancel;
        public volatile bool Rollback;
        public readonly object Gate = new();
        public TrainingStatusResponse Status = new();
    }

    private readonly AppState _state;
    private readonly ConcurrentDictionary<string, Handle> _trainers = new(StringComparer.Ordinal);

    public TrainingService(AppState state) => _state = state;

    public StartTrainingResponse Start(StartTrainingRequest request, IEventEmitter events)
    {
        var net = _state.GetNetwork(request.NetworkId) ?? throw new CommandException("Network not found");
        if (!_state.Corpora.TryGetValue(request.NetworkId, out var corpus))
            throw new CommandException("No corpus attached. Add training data on the Corpus tab first.");

        if (net.Kind == NetworkKinds.Transformer || net.Kind == NetworkKinds.NextToken)
            throw new NotYetMigratedException($"start_training ({net.Kind})");

        var ff = corpus.Feedforward ?? throw new CommandException("Feedforward network requires a feedforward corpus");
        var lossKind = Losses.Parse(request.Config.Loss);
        var x = new Tensor(new[] { ff.Rows, ff.InDim }, ff.Features.ToArray());
        var y = new Tensor(new[] { ff.Rows, ff.OutDim }, ff.Targets.ToArray());

        if (ff.Rows == 0) throw new CommandException("No training examples produced from the current corpus");
        if (request.Config.Epochs == 0) throw new CommandException("epochs must be > 0");
        if (net.ParameterCount == 0) throw new CommandException("Network has no parameters yet.");

        var model = _state.GetModel(request.NetworkId) ?? throw new CommandException("Model not materialized");
        var opt = BuildOptimizer(request.Config.Optimizer);
        var batchSize = Math.Max(1, request.Config.BatchSize);

        var trainingId = Guid.NewGuid().ToString();
        var handle = new Handle
        {
            Status = new TrainingStatusResponse
            {
                TrainingId = trainingId, Status = "running", TotalEpochs = request.Config.Epochs,
            },
        };
        _trainers[trainingId] = handle;

        var frozen = ParseFrozenLinearIndices(request.Config.FrozenLayers);
        var startedAt = DateTimeOffset.UtcNow;
        _ = Task.Run(() => RunLoop(trainingId, request.NetworkId, request.Config, startedAt,
            model.Clone(), x, y, lossKind, batchSize, opt, frozen, handle, events));

        return new StartTrainingResponse { TrainingId = trainingId, Status = "running" };
    }

    public bool Stop(string trainingId)
    {
        if (!_trainers.TryGetValue(trainingId, out var h)) return false;
        h.Cancel = true;
        return true;
    }

    public bool Abort(string trainingId)
    {
        if (!_trainers.TryGetValue(trainingId, out var h)) return false;
        h.Rollback = true;
        h.Cancel = true;
        return true;
    }

    public TrainingStatusResponse Status(string trainingId)
    {
        if (!_trainers.TryGetValue(trainingId, out var h)) throw new CommandException("Training run not found");
        lock (h.Gate) return h.Status with { LossHistory = new List<float>(h.Status.LossHistory) };
    }

    public IReadOnlyList<TrainingRun> History(string networkId)
    {
        var runs = _state.TrainingHistory.TryGetValue(networkId, out var list)
            ? new List<TrainingRun>(list)
            : new List<TrainingRun>();
        runs.Reverse(); // newest first
        return runs;
    }

    public void ClearHistory(string networkId)
    {
        _state.TrainingHistory.TryRemove(networkId, out _);
        Persistence.SaveState(_state);
    }

    // ── Training loop ─────────────────────────────────────────────────────────

    private void RunLoop(
        string trainingId, string networkId, TrainingConfig cfg, DateTimeOffset startedAt,
        Model working, Tensor x, Tensor y, Loss lossKind, int batchSize,
        OptimizerSpec opt, int[] frozen, Handle handle, IEventEmitter events)
    {
        var initial = working.Clone();
        var n = x.Rows;
        int inDim = x.Cols, outDim = y.Cols;
        var sw = Stopwatch.StartNew();
        var lossHistory = new List<float>(cfg.Epochs);
        var rng = new SplitMix64(unchecked((ulong)cfg.Seed + 0xA5A55A5A5A5AA5A5UL));
        var indices = Enumerable.Range(0, n).ToArray();
        var trainer = new MlpTrainer(working, opt, lossKind, frozen);

        try
        {
            for (var epoch = 1; epoch <= cfg.Epochs; epoch++)
            {
                if (handle.Cancel)
                {
                    FinishCancelled(trainingId, networkId, cfg, startedAt, epoch - 1, lossHistory,
                        sw, working, initial, handle, events);
                    return;
                }

                Shuffle(indices, rng);
                var epochLoss = 0.0f;
                var batches = 0;
                for (var start = 0; start < n; start += batchSize)
                {
                    if (handle.Cancel) break;
                    var len = Math.Min(batchSize, n - start);
                    var bx = new float[len * inDim];
                    var by = new float[len * outDim];
                    for (var r = 0; r < len; r++)
                    {
                        var src = indices[start + r];
                        Array.Copy(x.Data, src * inDim, bx, r * inDim, inDim);
                        Array.Copy(y.Data, src * outDim, by, r * outDim, outDim);
                    }
                    var loss = trainer.Step(new Tensor(new[] { len, inDim }, bx), new Tensor(new[] { len, outDim }, by));
                    if (!float.IsFinite(loss))
                    {
                        RecordRun(networkId, trainingId, cfg, startedAt, epoch,
                            lossHistory.Count > 0 ? lossHistory[^1] : 0f, (float)sw.Elapsed.TotalSeconds, lossHistory, "error");
                        EmitError(events, trainingId, $"Loss diverged to {loss} at epoch {epoch}", handle);
                        return;
                    }
                    epochLoss += loss;
                    batches++;
                }

                var meanLoss = epochLoss / Math.Max(1, batches);
                lossHistory.Add(meanLoss);
                _state.SetModel(networkId, working.Clone()); // per-epoch snapshot for inference

                var elapsed = (float)sw.Elapsed.TotalSeconds;
                lock (handle.Gate)
                {
                    handle.Status = handle.Status with
                    {
                        Epoch = epoch, LastLoss = meanLoss, LossHistory = new List<float>(lossHistory), ElapsedSecs = elapsed,
                    };
                }
                events.Emit(Events.TrainingUpdate, new TrainingUpdate
                {
                    TrainingId = trainingId, Epoch = epoch, TotalEpochs = cfg.Epochs,
                    Loss = meanLoss, LossHistory = new List<float>(lossHistory), ElapsedSecs = elapsed,
                });
            }

            // Completed.
            var finalElapsed = (float)sw.Elapsed.TotalSeconds;
            var finalLoss = lossHistory.Count > 0 ? lossHistory[^1] : 0f;
            _state.SetModel(networkId, working.Clone());
            if (_state.DataDir is { } dir) Persistence.SaveModel(dir, networkId, working);
            MarkTrained(networkId, pretrained: true);
            RecordRun(networkId, trainingId, cfg, startedAt, lossHistory.Count, finalLoss, finalElapsed, lossHistory, "completed");
            lock (handle.Gate) handle.Status = handle.Status with { Status = "completed", ElapsedSecs = finalElapsed };
            events.Emit(Events.TrainingFinished, new TrainingFinished
            {
                TrainingId = trainingId, Status = "completed", FinalLoss = finalLoss,
                TotalEpochs = cfg.Epochs, ElapsedSecs = finalElapsed,
            });
        }
        catch (Exception e)
        {
            EmitError(events, trainingId, e.Message, handle);
        }
    }

    private void FinishCancelled(
        string trainingId, string networkId, TrainingConfig cfg, DateTimeOffset startedAt,
        int epochsRun, List<float> lossHistory, Stopwatch sw, Model working, Model initial,
        Handle handle, IEventEmitter events)
    {
        var aborted = handle.Rollback;
        if (aborted) _state.SetModel(networkId, initial.Clone());
        else _state.SetModel(networkId, working.Clone());

        var elapsed = (float)sw.Elapsed.TotalSeconds;
        var finalLoss = lossHistory.Count > 0 ? lossHistory[^1] : 0f;
        var status = aborted ? "aborted" : "cancelled";

        if (!aborted && epochsRun > 0)
        {
            MarkTrained(networkId, pretrained: false);
            if (_state.DataDir is { } dir) Persistence.SaveModel(dir, networkId, working);
        }
        RecordRun(networkId, trainingId, cfg, startedAt, epochsRun, finalLoss, elapsed, lossHistory, status);
        lock (handle.Gate) handle.Status = handle.Status with { Status = status, ElapsedSecs = elapsed };
        events.Emit(Events.TrainingFinished, new TrainingFinished
        {
            TrainingId = trainingId, Status = status, FinalLoss = finalLoss,
            TotalEpochs = cfg.Epochs, ElapsedSecs = elapsed,
        });
    }

    private void MarkTrained(string networkId, bool pretrained)
    {
        if (_state.Networks.TryGetValue(networkId, out var net))
            _state.Networks[networkId] = net with { Trained = true, Pretrained = net.Pretrained || pretrained };
        Persistence.SaveState(_state);
    }

    private void EmitError(IEventEmitter events, string trainingId, string message, Handle handle)
    {
        lock (handle.Gate) handle.Status = handle.Status with { Status = "error" };
        events.Emit(Events.TrainingError, new TrainingError { TrainingId = trainingId, Message = message });
    }

    private void RecordRun(
        string networkId, string trainingId, TrainingConfig cfg, DateTimeOffset startedAt,
        int epochsRun, float finalLoss, float elapsed, List<float> lossHistory, string status)
    {
        var run = new TrainingRun
        {
            Id = trainingId, NetworkId = networkId, StartedAt = startedAt, FinishedAt = DateTimeOffset.UtcNow,
            Status = status, TotalEpochs = cfg.Epochs, EpochsRun = epochsRun, FinalLoss = finalLoss, ElapsedSecs = elapsed,
            LossHistory = Downsample(lossHistory, 500),
            ConfigSummary = new TrainingConfigSummary
            {
                Optimizer = cfg.Optimizer.Kind, Lr = cfg.Optimizer.Lr, BatchSize = cfg.BatchSize, Epochs = cfg.Epochs,
            },
        };
        var list = _state.TrainingHistory.GetOrAdd(networkId, _ => new List<TrainingRun>());
        lock (list) list.Add(run);
        Persistence.SaveState(_state);
    }

    private static List<float> Downsample(List<float> history, int max)
    {
        if (history.Count <= max) return new List<float>(history);
        var outList = new List<float>(max);
        var stride = (double)history.Count / max;
        for (var i = 0; i < max; i++) outList.Add(history[(int)(i * stride)]);
        return outList;
    }

    private static void Shuffle(int[] indices, SplitMix64 rng)
    {
        for (var i = indices.Length - 1; i >= 1; i--)
        {
            var j = (int)(rng.NextU64() % (ulong)(i + 1));
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }
    }

    private static OptimizerSpec BuildOptimizer(OptimizerConfig c) =>
        OptimizerSpec.FromConfig(c.Kind, c.Lr, c.Beta1, c.Beta2, c.Eps, c.Momentum, c.WeightDecay);

    /// <summary>Parse frozen-layer keys ("linear:N" or bare "N") into linear-layer indices.</summary>
    private static int[] ParseFrozenLinearIndices(IReadOnlyList<string>? keys)
    {
        if (keys is null || keys.Count == 0) return Array.Empty<int>();
        var result = new List<int>();
        foreach (var key in keys)
        {
            var k = key.Trim();
            if (k.StartsWith("linear:", StringComparison.OrdinalIgnoreCase)) k = k["linear:".Length..];
            if (int.TryParse(k, out var idx)) result.Add(idx);
        }
        return result.ToArray();
    }
}
