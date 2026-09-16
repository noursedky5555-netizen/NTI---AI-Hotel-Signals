"""Create a small inline logo for the sidebar."""

from PIL import Image, ImageDraw
from pathlib import Path

output_dir = Path(__file__).resolve().parents[1] / "assets"

def create_inline_logo():
    """Create a small logo to display inline with text."""
    
    # Create small icon (64x64) for inline display
    print("Creating inline logo (64x64)...")
    size = (64, 64)
    img = Image.new('RGBA', size, color=(255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Dark navy background
    bg_color = (0, 26, 53)
    draw.rounded_rectangle(
        [(2, 2), (size[0]-2, size[1]-2)],
        radius=8,
        fill=bg_color
    )
    
    # Draw house with upward trend line (simplified icon)
    padding = 10
    
    # House outline
    house_left = padding
    house_top = padding + 5
    house_right = size[0] - padding
    house_bottom = size[1] - padding
    
    # Roof (white triangle)
    roof_points = [
        (size[0]//2, house_top),  # peak
        (house_left, house_top + 12),  # left
        (house_right, house_top + 12)  # right
    ]
    draw.polygon(roof_points, fill=(255, 255, 255))
    
    # House body (cyan rectangle)
    draw.rectangle(
        [(house_left, house_top + 12), (house_right, house_bottom)],
        fill=(0, 188, 212)
    )
    
    # Trending line (upward curve in white)
    trend_points = [
        (house_left + 8, house_top + 20),
        (size[0]//2 - 8, house_top + 12),
        (size[0] - padding - 5, house_top + 5)
    ]
    draw.line(trend_points, fill=(255, 255, 255), width=2)
    
    # Save as PNG
    img.save(output_dir / "hotel_signals_inline.png", "PNG", quality=95)
    print(f"✅ Inline logo saved (64x64)")

if __name__ == "__main__":
    create_inline_logo()
