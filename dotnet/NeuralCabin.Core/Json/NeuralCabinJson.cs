using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace NeuralCabin.Core.Json;

/// <summary>
/// Central JSON configuration. Produces output byte-compatible with the Rust
/// backend's serde serialization so the existing React frontend cannot tell
/// which backend answered an <c>invoke</c> call.
///
/// serde defaults used by the Tauri layer:
///  - struct fields serialize as-is (already snake_case in Rust), so we use
///    <see cref="JsonNamingPolicy.SnakeCaseLower"/>;
///  - <c>Option&lt;T&gt;</c> fields without <c>skip_serializing_if</c> emit
///    <c>null</c> — which is System.Text.Json's default, so we keep nulls;
///  - <c>chrono</c> <c>DateTime&lt;Utc&gt;</c> serializes as an RFC 3339 string
///    ending in <c>Z</c> — see <see cref="Rfc3339DateTimeOffsetConverter"/>.
/// </summary>
public static class NeuralCabinJson
{
    /// <summary>Shared, thread-safe options instance for all (de)serialization.</summary>
    public static JsonSerializerOptions Options { get; } = Create();

    public static JsonSerializerOptions Create()
    {
        var options = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
            DictionaryKeyPolicy = JsonNamingPolicy.SnakeCaseLower,
            // serde matches field names exactly; mirror that on the way in.
            PropertyNameCaseInsensitive = false,
            // Match serde: emit null for Option::None (except where a DTO opts
            // out per-property via [JsonIgnore(WhenWritingNull)]).
            DefaultIgnoreCondition = JsonIgnoreCondition.Never,
            WriteIndented = false,
        };
        options.Converters.Add(new Rfc3339DateTimeOffsetConverter());
        return options;
    }
}

/// <summary>
/// Serializes timestamps the way chrono + serde do: an RFC 3339 / ISO 8601
/// instant in UTC ending with a literal <c>Z</c>. JavaScript's <c>Date</c>
/// parses this identically to the Tauri output.
/// </summary>
public sealed class Rfc3339DateTimeOffsetConverter : JsonConverter<DateTimeOffset>
{
    public override DateTimeOffset Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        var raw = reader.GetString()
            ?? throw new JsonException("expected an RFC 3339 timestamp string");
        return DateTimeOffset.Parse(
            raw, CultureInfo.InvariantCulture,
            DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal);
    }

    public override void Write(Utf8JsonWriter writer, DateTimeOffset value, JsonSerializerOptions options)
    {
        writer.WriteStringValue(
            value.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.ffffffZ", CultureInfo.InvariantCulture));
    }
}
