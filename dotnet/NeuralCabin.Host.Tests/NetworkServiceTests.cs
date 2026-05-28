using NeuralCabin.Core.Models;
using NeuralCabin.Host.Ipc;
using NeuralCabin.Host.Services;
using Xunit;

namespace NeuralCabin.Host.Tests;

/// <summary>
/// Verifies the network CRUD port reproduces the Rust <c>create_network</c>
/// semantics: parameter counting, dim derivation, and validation errors.
/// </summary>
public class NetworkServiceTests
{
    private static CreateNetworkRequest Feedforward(params LayerDef[] layers) => new()
    {
        Name = "Net",
        Kind = NetworkKinds.Feedforward,
        Seed = 1,
        InputDim = layers.OfType<LinearLayer>().FirstOrDefault()?.InDim ?? 2,
        Layers = layers.ToList(),
    };

    [Fact]
    public void Create_feedforward_counts_parameters_and_derives_output_dim()
    {
        var service = new NetworkService();
        var network = service.Create(Feedforward(
            new LinearLayer { InDim = 3, OutDim = 4 },
            new ActivationLayer { Activation = "relu" },
            new LinearLayer { InDim = 4, OutDim = 2 }));

        // (3*4 + 4) + (4*2 + 2) = 16 + 10 = 26.
        Assert.Equal(26, network.ParameterCount);
        Assert.Equal(3, network.InputDim);
        Assert.Equal(2, network.OutputDim);
        Assert.False(network.Trained);
        Assert.False(network.Pretrained);
        Assert.NotEqual(string.Empty, network.Id);
        Assert.Null(network.HiddenLayers);
    }

    [Fact]
    public void Create_feedforward_rejects_dim_mismatch()
    {
        var service = new NetworkService();
        var ex = Assert.Throws<CommandException>(() => service.Create(Feedforward(
            new LinearLayer { InDim = 3, OutDim = 4 },
            new LinearLayer { InDim = 5, OutDim = 2 }))); // 5 != running dim 4
        Assert.Contains("doesn't match running dim", ex.Message);
    }

    [Fact]
    public void Create_feedforward_requires_at_least_one_layer()
    {
        var service = new NetworkService();
        var request = new CreateNetworkRequest { Name = "N", Kind = NetworkKinds.Feedforward, InputDim = 2 };
        var ex = Assert.Throws<CommandException>(() => service.Create(request));
        Assert.Contains("at least one layer", ex.Message);
    }

    [Fact]
    public void Create_feedforward_requires_positive_input_dim()
    {
        var service = new NetworkService();
        var request = new CreateNetworkRequest
        {
            Name = "N",
            Kind = NetworkKinds.Feedforward,
            InputDim = 0,
            Layers = new List<LayerDef> { new LinearLayer { InDim = 0, OutDim = 2 } },
        };
        var ex = Assert.Throws<CommandException>(() => service.Create(request));
        Assert.Contains("input_dim > 0", ex.Message);
    }

    [Fact]
    public void Create_rejects_empty_name_and_unknown_kind()
    {
        var service = new NetworkService();
        Assert.Throws<CommandException>(() => service.Create(new CreateNetworkRequest
        {
            Name = "   ",
            Kind = NetworkKinds.Feedforward,
            InputDim = 2,
            Layers = new List<LayerDef> { new LinearLayer { InDim = 2, OutDim = 1 } },
        }));

        Assert.Throws<CommandException>(() => service.Create(new CreateNetworkRequest
        {
            Name = "N",
            Kind = "wobbly",
        }));
    }

    [Fact]
    public void Create_next_token_stores_hidden_chain_without_engine()
    {
        var service = new NetworkService();
        var network = service.Create(new CreateNetworkRequest
        {
            Name = "Gen",
            Kind = NetworkKinds.NextToken,
            Seed = 9,
            ContextSize = 16,
            Layers = new List<LayerDef> { new LinearLayer { InDim = 8, OutDim = 8 } },
        });

        Assert.Equal(NetworkKinds.NextToken, network.Kind);
        Assert.Equal(0, network.ParameterCount);
        Assert.Equal(16, network.ContextSize);
        Assert.NotNull(network.HiddenLayers);
        Assert.Single(network.HiddenLayers!);
        Assert.Empty(network.Layers);
    }

    [Fact]
    public void Create_transformer_is_deferred_to_engine_port()
    {
        var service = new NetworkService();
        Assert.Throws<NotYetMigratedException>(() => service.Create(new CreateNetworkRequest
        {
            Name = "T",
            Kind = NetworkKinds.Transformer,
            Transformer = new TransformerHParams { NCtx = 32, NEmbd = 16, NLayers = 2, NHeads = 2, NFf = 32 },
        }));
    }

    [Fact]
    public void List_get_and_delete_round_trip()
    {
        var service = new NetworkService();
        var a = service.Create(Feedforward(new LinearLayer { InDim = 2, OutDim = 1 }));
        var b = service.Create(Feedforward(new LinearLayer { InDim = 2, OutDim = 1 }));

        Assert.Equal(2, service.List().Count);
        Assert.Equal(a.Id, service.Get(a.Id).Id);

        Assert.True(service.Delete(a.Id));
        Assert.False(service.Delete(a.Id)); // already gone
        Assert.Single(service.List());
        Assert.Throws<CommandException>(() => service.Get(a.Id));
        Assert.Equal(b.Id, service.Get(b.Id).Id);
    }
}
