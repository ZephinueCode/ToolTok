# src/tools/visual_utils.py

from PIL import Image, ImageDraw, ImageFont
import textwrap # [NEW] For wrapping long instructions

def draw_cursor(image: Image.Image, x: int, y: int, color: str = "red", radius: int = 40) -> Image.Image:
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
    draw.line([(x - line_len, y), (x + line_len, y)], fill=color, width=10)
    draw.line([(x, y - line_len), (x, y + line_len)], fill=color, width=10)
    
    # 2. Circle
    draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], outline=color, width=10)
    
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

def visualize_trajectory(
    base_image: Image.Image, 
    cursor_path: list, 
    actions: list, 
    gt_bbox: list, 
    success: bool,
    instruction: str = None # [NEW] Optional instruction text
) -> Image.Image:
    """
    Draws the full execution path on a single image for Evaluation visualization.
    - Green Box: Ground Truth Target
    - Cyan Lines: Movement Path
    - Red X: End Point
    - Top Right: Instruction Text Overlay
    - Image Border: Green (Success) / Red (Fail)
    """
    # 1. Create a canvas (Dim original image slightly to make path pop)
    canvas = base_image.convert("RGBA")
    overlay = Image.new('RGBA', canvas.size, (0, 0, 0, 60)) # Darken slightly (alpha 60)
    canvas = Image.alpha_composite(canvas, overlay).convert("RGB")
    
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    
    # 2. Draw Ground Truth BBox (Green)
    if gt_bbox:
        # ScreenSpot format: [x1, y1, x2, y2]
        draw.rectangle(gt_bbox, outline="#00FF00", width=5)
        # Label GT
        try:
            font_gt = ImageFont.truetype("arial.ttf", 20)
        except:
            font_gt = ImageFont.load_default()
            
        # Draw label background
        label_x = gt_bbox[0]
        label_y = max(0, gt_bbox[1] - 25)
        draw.rectangle([label_x, label_y, label_x+80, label_y+25], fill="#00FF00")
        draw.text((label_x+5, label_y+2), "TARGET", fill="black", font=font_gt)

    # 3. Draw Path (Cyan Lines)
    if len(cursor_path) > 1:
        draw.line(cursor_path, fill="cyan", width=4)

    # 4. Draw Key Points
    if cursor_path:
        # Start Point (White Circle)
        sx, sy = cursor_path[0]
        draw.ellipse([sx-8, sy-8, sx+8, sy+8], fill="white", outline="black", width=2)

        # End Point (Red Crosshair)
        ex, ey = cursor_path[-1]
        r = 40
        draw.line([ex-r, ey-r, ex+r, ey+r], fill="red", width=8)
        draw.line([ex-r, ey+r, ex+r, ey-r], fill="red", width=8)
        draw.ellipse([ex-r, ey-r, ex+r, ey+r], outline="red", width=8)

    # 5. [NEW] Draw Instruction Text (Top Right)
    if instruction:
        try:
            font_text = ImageFont.truetype("arial.ttf", 24)
        except:
            font_text = ImageFont.load_default()
            
        # Wrap text to fit in 40% of screen width
        max_chars = int((w * 0.4) / 12) # Approx char width
        lines = textwrap.wrap(f"Instr: {instruction}", width=max_chars)
        
        # Calculate box size
        line_height = 30
        box_w = w * 0.45
        box_h = len(lines) * line_height + 20
        
        box_x = w - box_w - 20
        box_y = 20
        
        # Draw Semi-transparent Box
        overlay_box = Image.new('RGBA', canvas.size, (0,0,0,0))
        draw_box = ImageDraw.Draw(overlay_box)
        draw_box.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill=(0, 0, 0, 180))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay_box).convert("RGB")
        
        # Draw Text
        draw = ImageDraw.Draw(canvas) # Re-init draw on new canvas
        for i, line in enumerate(lines):
            draw.text((box_x + 10, box_y + 10 + i*line_height), line, fill="white", font=font_text)

    # 6. Add Border based on Success/Fail
    border_color = "#00FF00" if success else "#FF0000"
    border_width = 10
    draw.rectangle([0, 0, w-1, h-1], outline=border_color, width=border_width)
    
    return canvas