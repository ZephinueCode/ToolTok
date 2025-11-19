# src/tools/visual_utils.py

from PIL import Image, ImageDraw, ImageFont

def draw_cursor(image: Image.Image, x: int, y: int, color: str = "red", radius: int = 10) -> Image.Image:
    """
    Draws a visual cursor on the image at coordinates (x, y) with label 'cursor'.
    Operates directly on the original image size.
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    w, h = img_copy.size
    
    # Clamp coordinates to current image dimensions
    x = max(0, min(x, w-1))
    y = max(0, min(y, h-1))
    
    # 1. Crosshair lines
    line_len = radius * 1.5
    draw.line([(x - line_len, y), (x + line_len, y)], fill=color, width=4)
    draw.line([(x, y - line_len), (x, y + line_len)], fill=color, width=4)
    
    # 2. Circle
    draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], outline=color, width=4)
    
    # 3. Text Label
    try:
        # Try to use a proportional font size based on image width
        font_size = max(15, int(w * 0.02)) 
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
        
    label = "cursor"
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    
    # Position text: Try top-right of cursor, fallback if out of bounds
    text_x = x + radius + 5
    text_y = y - radius - text_h
    
    if text_x + text_w > w: text_x = x - radius - text_w - 5
    if text_y < 0: text_y = y + radius + 5

    # Draw background and text
    draw.rectangle([text_x - 2, text_y - 2, text_x + text_w + 2, text_y + text_h + 2], fill="black")
    draw.text((text_x, text_y), label, font=font, fill="white")
    
    return img_copy