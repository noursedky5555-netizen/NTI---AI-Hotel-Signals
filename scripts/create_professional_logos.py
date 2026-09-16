"""Create professional Hotel Signals logos matching the brand design."""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import math

output_dir = Path(__file__).resolve().parents[1] / "assets"

def create_professional_logo():
    """Create a professional Hotel Signals logo with proper branding."""
    
    # Large sidebar logo (200x80)
    print("Creating sidebar logo (200x80)...")
    logo_size = (200, 80)
    img = Image.new('RGBA', logo_size, color=(255, 255, 255, 0))  # Transparent
    draw = ImageDraw.Draw(img)
    
    # Draw background rounded rectangle (dark navy)
    bg_color = (0, 26, 53)  # Dark navy from your brand
    margin = 5
    draw.rounded_rectangle(
        [(margin, margin), (logo_size[0]-margin, logo_size[1]-margin)],
        radius=10,
        fill=bg_color
    )
    
    # Add text "Hotel Signals" with proper styling
    try:
        font_large = ImageFont.truetype("arial.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 8)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Main text
    text_main = "Hotel Signals"
    
    # Get text bounding box to center it
    bbox = draw.textbbox((0, 0), text_main, font=font_large)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (logo_size[0] - text_width) // 2
    y = (logo_size[1] - text_height) // 2 - 2
    
    # Draw text in white
    draw.text((x, y), text_main, fill=(255, 255, 255), font=font_large)
    
    # Save high quality
    img.save(output_dir / "hotel_signals_logo.png", "PNG", quality=95)
    print(f"✅ Sidebar logo saved (200x80)")
    
    # Create favicon (32x32)
    print("Creating favicon (32x32)...")
    favicon_size = (32, 32)
    favicon = Image.new('RGBA', favicon_size, color=(255, 255, 255, 0))
    draw_ico = ImageDraw.Draw(favicon)
    
    # Draw background rounded rectangle
    margin_ico = 2
    draw_ico.rounded_rectangle(
        [(margin_ico, margin_ico), (favicon_size[0]-margin_ico, favicon_size[1]-margin_ico)],
        radius=4,
        fill=bg_color
    )
    
    # Add simple house icon in white
    # Draw simplified house shape
    padding = 6
    x_start = padding
    y_start = padding
    width = favicon_size[0] - (padding * 2)
    height = favicon_size[1] - (padding * 2)
    
    # Roof (triangle)
    roof_points = [
        (x_start + width//2, y_start),  # top
        (x_start, y_start + height//3),  # left
        (x_start + width, y_start + height//3)  # right
    ]
    draw_ico.polygon(roof_points, fill=(255, 255, 255))
    
    # House body (rectangle)
    draw_ico.rectangle(
        [(x_start, y_start + height//3), (x_start + width, y_start + height)],
        fill=(0, 188, 212)  # Cyan color for contrast
    )
    
    favicon.save(output_dir / "hotel_signals_favicon.png", "PNG", quality=95)
    print(f"✅ Favicon saved (32x32)")
    
    # Create a larger header logo (400x160) for potential use
    print("Creating header logo (400x160)...")
    header_size = (400, 160)
    header = Image.new('RGBA', header_size, color=(255, 255, 255, 0))
    draw_header = ImageDraw.Draw(header)
    
    # Draw background
    draw_header.rounded_rectangle(
        [(5, 5), (header_size[0]-5, header_size[1]-5)],
        radius=15,
        fill=bg_color
    )
    
    try:
        font_header = ImageFont.truetype("arial.ttf", 36)
        font_sub = ImageFont.truetype("arial.ttf", 10)
    except:
        font_header = ImageFont.load_default()
        font_sub = ImageFont.load_default()
    
    # Main title
    title = "Hotel Signals"
    bbox_title = draw_header.textbbox((0, 0), title, font=font_header)
    title_width = bbox_title[2] - bbox_title[0]
    title_x = (header_size[0] - title_width) // 2
    title_y = (header_size[1] - (bbox_title[3] - bbox_title[1])) // 2 - 10
    
    draw_header.text((title_x, title_y), title, fill=(255, 255, 255), font=font_header)
    
    # Subtitle
    subtitle = "Booking Intelligence"
    bbox_sub = draw_header.textbbox((0, 0), subtitle, font=font_sub)
    sub_width = bbox_sub[2] - bbox_sub[0]
    sub_x = (header_size[0] - sub_width) // 2
    sub_y = title_y + (bbox_title[3] - bbox_title[1]) + 5
    
    draw_header.text((sub_x, sub_y), subtitle, fill=(0, 188, 212), font=font_sub)
    
    header.save(output_dir / "hotel_signals_header.png", "PNG", quality=95)
    print(f"✅ Header logo saved (400x160)")
    
    print("\n✨ All logos created successfully!")
    print("📁 Files created:")
    print("  • hotel_signals_logo.png (200x80) - For sidebar")
    print("  • hotel_signals_favicon.png (32x32) - For browser tab")
    print("  • hotel_signals_header.png (400x160) - Optional header use")
    print("\n💡 These are professional quality PNG files with transparent backgrounds.")
    print("   They will display cleanly on both light and dark backgrounds.")

if __name__ == "__main__":
    create_professional_logo()
