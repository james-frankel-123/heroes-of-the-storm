using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Media;

namespace HotsFever.Overlay;

/// <summary>A single recommendation row bound into the overlay list.</summary>
public sealed class RecItem
{
    public string Portrait { get; set; } = "";
    public string Hero { get; set; } = "";
    public string Subtitle { get; set; } = "";
    public string WinDelta { get; set; } = "";
    public Brush WinDeltaBrush { get; set; } = new SolidColorBrush();
    public bool IsAiPick { get; set; }
    public Visibility AiPickVisibility => IsAiPick ? Visibility.Visible : Visibility.Collapsed;
}
