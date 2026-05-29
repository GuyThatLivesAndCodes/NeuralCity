using System.Text.Json;
using System.Text.Json.Serialization;

namespace NeuralCabin.Engine;

/// <summary>
/// JSON (de)serialization for the engine's model types, byte-compatible with
/// the Rust serde output so model files written by either backend interoperate:
///  - <see cref="Tensor"/>   → <c>{ "shape": [...], "data": [...] }</c>
///  - <see cref="Activation"/> → variant name string (e.g. <c>"ReLU"</c>)
///  - <see cref="Layer"/>     → externally tagged: <c>{"Linear":{...}}</c> / <c>{"Activation":"Tanh"}</c>
///  - <see cref="Model"/>     → snake_case fields (<c>input_dim</c>, <c>layers</c>, <c>seed</c>)
/// </summary>
public static class EngineModelJson
{
    public static JsonSerializerOptions Options { get; } = Create();

    private static JsonSerializerOptions Create()
    {
        var options = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
            WriteIndented = true, // Rust uses serde_json::to_string_pretty for model files
        };
        options.Converters.Add(new TensorConverter());
        options.Converters.Add(new ActivationConverter());
        options.Converters.Add(new LayerConverter());
        return options;
    }
}

public sealed class TensorConverter : JsonConverter<Tensor>
{
    public override Tensor Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        int[]? shape = null;
        float[]? data = null;
        if (reader.TokenType != JsonTokenType.StartObject) throw new JsonException("expected tensor object");
        while (reader.Read() && reader.TokenType != JsonTokenType.EndObject)
        {
            var prop = reader.GetString();
            reader.Read();
            if (prop == "shape") shape = JsonSerializer.Deserialize<int[]>(ref reader, options);
            else if (prop == "data") data = JsonSerializer.Deserialize<float[]>(ref reader, options);
            else reader.Skip();
        }
        if (shape is null || data is null) throw new JsonException("tensor missing shape/data");
        return new Tensor(shape, data);
    }

    public override void Write(Utf8JsonWriter writer, Tensor value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();
        writer.WritePropertyName("shape");
        JsonSerializer.Serialize(writer, value.Shape, options);
        writer.WritePropertyName("data");
        JsonSerializer.Serialize(writer, value.Data, options);
        writer.WriteEndObject();
    }
}

public sealed class ActivationConverter : JsonConverter<Activation>
{
    public override Activation Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        var s = reader.GetString() ?? throw new JsonException("expected activation string");
        return s.ToLowerInvariant() switch
        {
            "identity" => Activation.Identity,
            "relu" => Activation.ReLU,
            "sigmoid" => Activation.Sigmoid,
            "tanh" => Activation.Tanh,
            "softmax" => Activation.Softmax,
            _ => throw new JsonException($"unknown activation '{s}'"),
        };
    }

    public override void Write(Utf8JsonWriter writer, Activation value, JsonSerializerOptions options) =>
        writer.WriteStringValue(value.Name());
}

public sealed class LayerConverter : JsonConverter<Layer>
{
    public override Layer Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType != JsonTokenType.StartObject) throw new JsonException("expected layer object");
        reader.Read();
        if (reader.TokenType != JsonTokenType.PropertyName) throw new JsonException("expected layer tag");
        var tag = reader.GetString();
        reader.Read();

        Layer layer;
        if (tag == "Linear")
        {
            int inDim = 0, outDim = 0;
            Tensor? w = null, b = null;
            if (reader.TokenType != JsonTokenType.StartObject) throw new JsonException("expected linear body");
            while (reader.Read() && reader.TokenType != JsonTokenType.EndObject)
            {
                var prop = reader.GetString();
                reader.Read();
                switch (prop)
                {
                    case "in_dim": inDim = reader.GetInt32(); break;
                    case "out_dim": outDim = reader.GetInt32(); break;
                    case "w": w = JsonSerializer.Deserialize<Tensor>(ref reader, options); break;
                    case "b": b = JsonSerializer.Deserialize<Tensor>(ref reader, options); break;
                    default: reader.Skip(); break;
                }
            }
            if (w is null || b is null) throw new JsonException("linear layer missing weights");
            layer = new LinearLayer(inDim, outDim, w, b);
        }
        else if (tag == "Activation")
        {
            var act = JsonSerializer.Deserialize<Activation>(ref reader, options);
            layer = new ActivationLayer(act);
        }
        else
        {
            throw new JsonException($"unknown layer tag '{tag}'");
        }

        reader.Read(); // consume EndObject of the wrapper
        return layer;
    }

    public override void Write(Utf8JsonWriter writer, Layer value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();
        switch (value)
        {
            case LinearLayer ll:
                writer.WritePropertyName("Linear");
                writer.WriteStartObject();
                writer.WriteNumber("in_dim", ll.InDim);
                writer.WriteNumber("out_dim", ll.OutDim);
                writer.WritePropertyName("w");
                JsonSerializer.Serialize(writer, ll.W, options);
                writer.WritePropertyName("b");
                JsonSerializer.Serialize(writer, ll.B, options);
                writer.WriteEndObject();
                break;
            case ActivationLayer al:
                writer.WritePropertyName("Activation");
                JsonSerializer.Serialize(writer, al.Activation, options);
                break;
            default:
                throw new JsonException($"unknown layer type {value.GetType().Name}");
        }
        writer.WriteEndObject();
    }
}
