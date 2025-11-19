# src/utils/generate_data.py

import json
import random
import os
from .parameters import HYPERPARAMS as HP

# ================= CONFIGURATION =================
OUTPUT_FILE = HP.SFT_DATA_PATH
SAMPLES_PER_TOKEN = HP.SFT_SAMPLES_PER_ACTION

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ================= VOCABULARY & TEMPLATES =================
VOCAB = {
    # UI Elements
    "ui_element": [
        "button", "icon", "link", "menu", "image", "field", "card", "tab", "toggle", "slider",
        "checkbox", "dropdown", "thumbnail", "avatar", "banner", "navigation bar", "search bar",
        "widget", "modal", "popup"
    ],
    "container": [
        "page", "screen", "window", "list", "panel", "view", "content area", "sidebar", 
        "dashboard", "interface", "canvas", "grid"
    ],
    
    # Verbs (Expanded)
    "move_verb": [
        "Move", "Shift", "Slide", "Position the cursor", "Glide", 
        "Nudge", "Relocate", "Adjust", "Drag", "Sweep", 
        "Translate", "Hover", "Bring", "Guide", "Point"
    ],
    "click_verb": [
        "Click", "Tap", "Select", "Hit", "Press", 
        "Choose", "Activate", "Trigger", "Interact with", "Pick"
    ],
    "long_click_verb": [
        "Long press", "Hold down", "Press and hold", "Click and hold", 
        "Keep pressing", "Sustain a click on", "Long-tap"
    ],
    "scroll_verb": [
        "Scroll", "Swipe", "Pan", "Roll", "Navigate", "Browse"
    ],
    "nav_verb": [
        "Go", "Navigate", "Return", "Jump", "Switch", "Head"
    ],
    "type_verb": [
        "Type", "Enter", "Input", "Write", "Fill in", "Submit"
    ],
    
    # Adverbs/Adjectives for Distance (Expanded)
    "far_desc": [
        "significantly", "a lot", "a long distance", "far", "way", "by 200 pixels", "a large jump",
        "substantially", "across the screen", "a big leap", "drastically", "a huge amount"
    ],
    "mid_desc": [
        "moderately", "a bit", "somewhat", "medium distance", "by 30 pixels", "a normal step",
        "noticeably", "standard amount", "an average distance", "halfway"
    ],
    "clo_desc": [
        "slightly", "a tiny bit", "just a nudge", "barely", "by 5 pixels", "pixel-perfect adjustment",
        "minimally", "a hair", "a smidge", "very carefully", "precisely", "micro-adjustment"
    ],
    
    # Directions (Expanded)
    "up": [
        "up", "upwards", "to the top", "north", "higher", 
        "vertically up", "towards the ceiling", "ascend"
    ],
    "down": [
        "down", "downwards", "to the bottom", "south", "lower", 
        "vertically down", "towards the floor", "descend"
    ],
    "left": [
        "left", "leftwards", "to the west", "to the left side", 
        "horizontally left", "backwards (direction)"
    ],
    "right": [
        "right", "rightwards", "to the east", "to the right side", 
        "horizontally right", "forwards (direction)"
    ],

    # Text Contexts (Specific strings to type)
    "text_content": [
        "Hello world", "search query", "user@example.com", "password123", 
        "Python tutorial", "New York City", "The quick brown fox", 
        "Meeting notes", "TODO list", "123-456-7890", 
        "Buy milk", "Address: 123 Main St", "Confirm", "John Doe",
        "react native", "chmod +x script.sh", "Funny cat videos"
    ],
    
    # Termination
    "end_action": [
        "Stop task", "Finish interaction", "Task complete", "End the session", "Done",
        "Terminate", "Quit", "Exit", "Halt operation", "Wrap up"
    ],
}

# Templates to wrap the core instruction
TEMPLATES = [
    "[Action] Perform a step for the following action: {instruction}",
    "[Action] Perform a step for the following action: {instruction}",
    "[Action] Perform a step for the following action: {instruction}",
    "[Action] Perform a step for the following action: {instruction}",
    "[Action] Perform a step for the following action: {instruction}", # Weighting higher
    "[Action] Please {instruction_lower}",
    "[Action] Action: {instruction}",
    "[Action] Execute: {instruction}",
    "[Action] I need you to {instruction_lower}",
    "[Action] Task: {instruction}",
]

# ================= TOKEN DEFINITIONS & METADATA =================
# Note: For TEXT_INPUT_SEQUENCE, the key is a placeholder for generation logic,
# not the literal token output.
TOKEN_META = {
    # --- Movement ---
    "<MOVE_UP_FAR>":    {"type": "move", "dir": "up", "dist": "far"},
    "<MOVE_UP_MID>":    {"type": "move", "dir": "up", "dist": "mid"},
    "<MOVE_UP_CLO>":    {"type": "move", "dir": "up", "dist": "clo"},
    
    "<MOVE_DOWN_FAR>":  {"type": "move", "dir": "down", "dist": "far"},
    "<MOVE_DOWN_MID>":  {"type": "move", "dir": "down", "dist": "mid"},
    "<MOVE_DOWN_CLO>":  {"type": "move", "dir": "down", "dist": "clo"},
    
    "<MOVE_LEFT_FAR>":  {"type": "move", "dir": "left", "dist": "far"},
    "<MOVE_LEFT_MID>":  {"type": "move", "dir": "left", "dist": "mid"},
    "<MOVE_LEFT_CLO>":  {"type": "move", "dir": "left", "dist": "clo"},
    
    "<MOVE_RIGHT_FAR>": {"type": "move", "dir": "right", "dist": "far"},
    "<MOVE_RIGHT_MID>": {"type": "move", "dir": "right", "dist": "mid"},
    "<MOVE_RIGHT_CLO>": {"type": "move", "dir": "right", "dist": "clo"},

    # --- Interaction ---
    "<CLICK_SHORT>":    {"type": "click", "mode": "short"},
    "<CLICK_LONG>":     {"type": "click", "mode": "long"},

    # --- Navigation ---
    "<GO_BACK>":        {"type": "nav", "mode": "back"},
    "<GO_HOME>":        {"type": "nav", "mode": "home"},

    # --- Scrolling ---
    "<SCROLL_UP>":      {"type": "scroll", "dir": "up"},
    "<SCROLL_DOWN>":    {"type": "scroll", "dir": "down"},
    "<SCROLL_LEFT>":    {"type": "scroll", "dir": "left"},
    "<SCROLL_RIGHT>":   {"type": "scroll", "dir": "right"},

    # --- Text (Combined) ---
    "TEXT_INPUT_SEQUENCE": {"type": "text_seq"},

    # --- Termination ---
    "<END_ACTION>":     {"type": "end"},
}

# ================= GENERATION LOGIC =================

def generate_sample(token_key, meta):
    """
    Returns tuple: (instruction_text, assistant_response_string)
    """
    t_type = meta["type"]
    
    if t_type == "move":
        verb = random.choice(VOCAB["move_verb"])
        direction = random.choice(VOCAB[meta["dir"]])
        distance_desc = random.choice(VOCAB[f"{meta['dist']}_desc"])
        
        case = random.randint(1, 5)
        if case == 1:
            instr = f"{verb} {direction} {distance_desc}"
        elif case == 2:
            instr = f"{verb} the cursor {distance_desc} {direction}"
        elif case == 3:
            instr = f"{distance_desc} {verb.lower()} {direction}"
        elif case == 4:
            instr = f"Make a {verb.lower()} {direction} {distance_desc}"
        else:
            instr = f"{verb} {direction}, make it {distance_desc}"
        
        return instr, token_key

    elif t_type == "click":
        target = random.choice(VOCAB["ui_element"])
        if meta["mode"] == "short":
            verb = random.choice(VOCAB["click_verb"])
            instr = f"{verb} the {target}"
        else:
            verb = random.choice(VOCAB["long_click_verb"])
            instr = f"{verb} on the {target}"
        return instr, token_key

    elif t_type == "scroll":
        verb = random.choice(VOCAB["scroll_verb"])
        direction = random.choice(VOCAB[meta["dir"]])
        target = random.choice(VOCAB["container"])
        instr = f"{verb} {direction} the {target}"
        return instr, token_key

    elif t_type == "nav":
        if meta["mode"] == "back":
            instr = random.choice(["Go back", "Return to previous page", "Navigate back", "Click the back button", "Retreat to last screen"])
        else:
            instr = random.choice(["Go to home", "Return to dashboard", "Navigate to the main menu", "Press the home button", "Jump to start"])
        return instr, token_key

    elif t_type == "text_seq":
        # Generate content
        content = random.choice(VOCAB["text_content"])
        verb = random.choice(VOCAB["type_verb"])
        target = random.choice(VOCAB["ui_element"])
        
        # Instruction variants
        case = random.randint(1, 3)
        if case == 1:
            instr = f"{verb} '{content}'"
        elif case == 2:
            instr = f"{verb} '{content}' into the {target}"
        else:
            instr = f"Input the text '{content}'"
            
        # Response: <TEXT_START> content <TEXT_END>
        response = f"<TEXT_START> {content} <TEXT_END>"
        return instr, response

    elif t_type == "end":
        instr = random.choice(VOCAB["end_action"])
        return instr, token_key
    
    return "Perform action", token_key

def apply_template(instruction):
    template = random.choice(TEMPLATES)
    if "{instruction_lower}" in template:
        return template.format(instruction_lower=instruction.lower())
    return template.format(instruction=instruction)

# ================= MAIN EXECUTION =================

def main():
    print(f"Generating {SAMPLES_PER_TOKEN} samples per token logic...")
    
    count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # Iterate through each defined token configuration
        for token_key, meta in TOKEN_META.items():
            for _ in range(SAMPLES_PER_TOKEN):
                
                # 1. Generate Core Instruction & Response
                core_instruction, assistant_response = generate_sample(token_key, meta)
                
                # 2. Apply formatting template to instruction
                user_content = apply_template(core_instruction)
                
                # 3. Create JSON structure
                record = {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_response}
                    ]
                }
                
                f.write(json.dumps(record) + "\n")
                count += 1
    
    print(f"Successfully generated {count} records.")
    print(f"Saved to: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()