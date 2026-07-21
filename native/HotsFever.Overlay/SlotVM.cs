using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Media;

namespace HotsFever.Overlay;

/// <summary>One ban/pick slot in the mini draft board (mirrors buildDraftView).</summary>
public sealed class SlotVM
{
    public string Portrait { get; set; } = "";
    public string Name { get; set; } = "";
    public Brush Background { get; set; } = new SolidColorBrush();
    public Brush BorderColor { get; set; } = new SolidColorBrush();
    public Thickness BorderT { get; set; }
    public Brush NameForeground { get; set; } = new SolidColorBrush();
}
