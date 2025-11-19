# src/utils/generate_data.py

import json
import random
import os
from .parameters import HYPERPARAMS as HP

# ================= CONFIGURATION =================
OUTPUT_FILE = HP.SFT_1_DATA_PATH
SAMPLES_PER_TOKEN = HP.SFT_1_SAMPLES_PER_ACTION

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

    # Text Contexts
    "text_start": [
        "Start typing", "Begin input", "Activate the text field", "Focus on the input box", "Ready to write",
        "Initiate text entry", "Open the keyboard", "Click to type", "Enter text mode"
    ],
    "text_end": [
        "Finish typing", "Complete input", "Stop writing", "Confirm text", "Exit text mode",
        "Submit the text", "Close the keyboard", "Done editing", "Finalize entry"
    ],
    
    # Termination
    "end_action": [
        "Stop task", "Finish interaction", "Task complete", "End the session", "Done",
        "Terminate", "Quit", "Exit", "Halt operation", "Wrap up"
    ],
}

# Templates to wrap the core instruction
TEMPLATES = [
    "Perform a step for the following action: {instruction}",
    "Perform a step for the following action: {instruction}",
    "Perform a step for the following action: {instruction}", # Weighting higher
    "Please {instruction_lower}",
    "Action: {instruction}",
    "{instruction}",
    "Execute: {instruction}",
    "I need you to {instruction_lower}",
    "Task: {instruction}",
]

# ================= TOKEN DEFINITIONS & METADATA =================
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

    # --- Text ---
    "<TEXT_START>":     {"type": "text", "mode": "start"},
    "<TEXT_END>":       {"type": "text", "mode": "end"},

    # --- Termination ---
    "<END_ACTION>":     {"type": "end"},
}

# ================= GENERATION LOGIC =================

def generate_instruction(meta):
    t_type = meta["type"]
    
    if t_type == "move":
        verb = random.choice(VOCAB["move_verb"])
        direction = random.choice(VOCAB[meta["dir"]])
        distance_desc = random.choice(VOCAB[f"{meta['dist']}_desc"])
        
        # Expanded Structures:
        case = random.randint(1, 5)
        if case == 1:
            return f"{verb} {direction} {distance_desc}"
        elif case == 2:
            return f"{verb} the cursor {distance_desc} {direction}"
        elif case == 3:
            return f"{distance_desc} {verb.lower()} {direction}"
        elif case == 4:
            return f"Make a {verb.lower()} {direction} {distance_desc}"
        else:
            return f"{verb} {direction}, make it {distance_desc}"

    elif t_type == "click":
        target = random.choice(VOCAB["ui_element"])
        if meta["mode"] == "short":
            verb = random.choice(VOCAB["click_verb"])
            return f"{verb} the {target}"
        else:
            verb = random.choice(VOCAB["long_click_verb"])
            return f"{verb} on the {target}"

    elif t_type == "scroll":
        verb = random.choice(VOCAB["scroll_verb"])
        direction = random.choice(VOCAB[meta["dir"]])
        target = random.choice(VOCAB["container"])
        return f"{verb} {direction} the {target}"

    elif t_type == "nav":
        if meta["mode"] == "back":
            return random.choice(["Go back", "Return to previous page", "Navigate back", "Click the back button", "Retreat to last screen"])
        else:
            return random.choice(["Go to home", "Return to dashboard", "Navigate to the main menu", "Press the home button", "Jump to start"])

    elif t_type == "text":
        if meta["mode"] == "start":
            return random.choice(VOCAB["text_start"])
        else:
            return random.choice(VOCAB["text_end"])

    elif t_type == "end":
        return random.choice(VOCAB["end_action"])
    
    return "Perform action"

def apply_template(instruction):
    template = random.choice(TEMPLATES)
    # Handle case capitalization based on template position
    if "{instruction_lower}" in template:
        return template.format(instruction_lower=instruction.lower())
    return template.format(instruction=instruction)

# ================= MAIN EXECUTION =================

def main():
    print(f"Generating {SAMPLES_PER_TOKEN} samples per token for {len(TOKEN_META)} tokens...")
    
    count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # Iterate through each defined token
        for token, meta in TOKEN_META.items():
            for _ in range(SAMPLES_PER_TOKEN):
                
                # 1. Generate the core natural language instruction
                core_instruction = generate_instruction(meta)
                
                # 2. Apply formatting template
                user_content = apply_template(core_instruction)
                
                # 3. Create JSON structure
                record = {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": token}
                    ]
                }
                
                f.write(json.dumps(record) + "\n")
                count += 1
    
    print(f"Successfully generated {count} records.")
    print(f"Saved to: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()