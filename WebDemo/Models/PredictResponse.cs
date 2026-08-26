namespace WebDemo.Models;

public sealed record PredictResponse(
    string Model,
    string? Image,
    string Message,
    object? Data = null);