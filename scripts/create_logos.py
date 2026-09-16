"""Create placeholder logo images for the Hotel Signals dashboard."""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Create output directory
output_dir = Path(__file__).resolve().parents[1] / "assets"

def create_logo(size, filename):
    """Create a simple Hotel Signals logo."""
    # Create image with dark background
    img = Image.new('RGB', size, color='#001a35')
    draw = ImageDraw.Draw(img)
    
    # Add text
    text = "🏨 Hotel Signals"
    
    # Try to use a nice font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", int(size[0] * 0.15))
    except:
        font = ImageFont.load_default()
    
    # Calculate text position (center)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    # Draw text
    draw.text((x, y), text, fill='#00BCD4', font=font)
    
    # Save image
    img.save(output_dir / filename)
    print(f"✅ Created {filename} ({size[0]}x{size[1]})")

# Create logo images
create_logo((200, 80), "hotel_signals_logo.png")  # Sidebar logo
create_logo((32, 32), "hotel_signals_favicon.png")  # Favicon

print("\n📝 Note: These are placeholder logos.")
print("Replace with your actual Hotel Signals logo images:")
print(f"  • hotel_signals_logo.png (200x80 or similar for sidebar)")
print(f"  • hotel_signals_favicon.png (32x32 for favicon)")
