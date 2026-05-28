using NeuralCabin.Core.Ipc;
using NeuralCabin.Core.Models;
using NeuralCabin.Host.Ipc;
using NeuralCabin.Host.Services;

namespace NeuralCabin.Host.Commands;

/// <summary>Registers the network CRUD commands against a <see cref="NetworkService"/>.</summary>
public static class NetworkCommands
{
    public static void Register(CommandRegistry registry, NetworkService service)
    {
        registry.Register(CommandNames.CreateNetwork, context =>
            Task.FromResult<object?>(service.Create(context.Arg<CreateNetworkRequest>("req"))));

        registry.Register(CommandNames.ListNetworks, _ =>
            Task.FromResult<object?>(service.List()));

        registry.Register(CommandNames.GetNetwork, context =>
            Task.FromResult<object?>(service.Get(context.ArgString("id"))));

        registry.Register(CommandNames.DeleteNetwork, context =>
            Task.FromResult<object?>(service.Delete(context.ArgString("id"))));
    }
}
