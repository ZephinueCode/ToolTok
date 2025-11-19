from PIL import Image, ImageDraw, ImageFont
import os

def draw_cursor(image: Image.Image, x: int, y: int, color: str = "red", radius: int = 10) -> Image.Image:
    """
    Draws a visual cursor on the image at coordinates (x, y) with label 'cursor'.
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    w, h = img_copy.size
    
    # Clamp coordinates within image bounds
    x = max(0, min(x, w-1))
    y = max(0, min(y, h-1))
    
    # 1. Draw crosshair lines
    line_len = radius * 1.5
    draw.line([(x - line_len, y), (x + line_len, y)], fill=color, width=4)
    draw.line([(x, y - line_len), (x, y + line_len)], fill=color, width=4)
    
    # 2. Draw circle
    draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], outline=color, width=4)
    
    # 3. Draw text label
    try:
        # Dynamic font size based on image width
        font_size = max(12, int(w * 0.03)) 
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            try:
                # Common paths for Linux/Mac
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except:
                font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
        
    label = "cursor"
    
    # Get text dimensions (using robust anchor method)
    # anchor='lt' means left-top, ensuring bbox matches default drawing behavior
    text_bbox = draw.textbbox((0, 0), label, font=font, anchor='lt')
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    
    # Calculate text position: default to top-right of cursor
    # Added extra offset to text_h to ensure it doesn't overlap the circle
    text_x = x + radius + 8 
    text_y = y - radius - text_h - 5 
    
    # Boundary checks: keep text within image
    if text_x + text_w > w: 
        text_x = x - radius - text_w - 8 # Move to left side
    if text_y < 0: 
        text_y = y + radius + 5 # Move to bottom side

    # --- Text Rendering ---
    # 1. No background rectangle used.
    # 2. Font color set to "red".
    # 3. Added stroke (outline) to ensure visibility on any background.
    draw.text(
        (text_x, text_y), 
        label, 
        font=font, 
        fill="red",          # Red text
        stroke_width=2,      # 2px outline
        stroke_fill="white"  # White outline
    )
    
    return img_copy

if __name__ == "__main__":
    # ==========================================
    # Test Code
    # ==========================================
    input_filename = "image.png"
    output_filename = "drawn.png"

    # 1. Generate dummy data if image.png doesn't exist
    if not os.path.exists(input_filename):
        print(f"Warning: {input_filename} not found. Generating dummy...")
        # Create a dark gray background to test text contrast
        dummy_img = Image.new('RGB', (800, 600), color=(100, 100, 100)) 
        dummy_img.save(input_filename)

    try:
        # 2. Load image
        print(f"Loading {input_filename}...")
        original_image = Image.open(input_filename).convert("RGB")

        # 3. Define test coordinates (e.g., center of image)
        target_x = original_image.width // 2
        target_y = original_image.height // 2
        
        print(f"Drawing cursor at ({target_x}, {target_y})...")

        # 4. Call draw function
        result_image = draw_cursor(original_image, target_x, target_y, color="red", radius=20)

        # 5. Save result
        result_image.save(output_filename)
        print(f"Success! Output saved to {output_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")