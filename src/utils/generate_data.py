import json
import random
import os
from .parameters import HYPERPARAMS as HP

# ================= CONFIGURATION =================
OUTPUT_FILE = HP.SFT_DATA_PATH
SAMPLES_PER_ACTION = HP.SFT_SAMPLES_PER_ACTION # e.g., 50
# REDUCE explanation ratio to avoid overfitting definitions
SAMPLES_PER_EXPLANATION = int(HP.SFT_SAMPLES_PER_ACTION / 4) # Reduced from /2 to /4

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ================= VOCABULARY & TEMPLATES (ACTION) =================
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
    
    # Verbs (Action Generation)
    "move_verb": ["Move", "Shift", "Slide", "Position the cursor", "Glide", "Nudge", "Relocate", "Adjust", "Drag"],
    "click_verb": ["Click", "Tap", "Select", "Hit", "Press", "Choose", "Activate", "Trigger"],
    "long_click_verb": ["Long press", "Hold down", "Press and hold", "Click and hold"],
    "scroll_verb": ["Scroll", "Swipe", "Pan", "Roll", "Navigate", "Browse"],
    "nav_verb": ["Go", "Navigate", "Return", "Jump", "Switch", "Head"],
    "type_verb": ["Type", "Enter", "Input", "Write", "Fill in", "Submit"],
    
    # Adverbs
    "far_desc": ["significantly", "a lot", "a long distance", "far", "way", "by 200 pixels", "a large jump"],
    "mid_desc": ["moderately", "a bit", "somewhat", "medium distance", "by 30 pixels", "a normal step"],
    "clo_desc": ["slightly", "a tiny bit", "just a nudge", "barely", "by 5 pixels", "pixel-perfect adjustment"],
    
    # Directions
    "up": ["up", "upwards", "to the top", "north"],
    "down": ["down", "downwards", "to the bottom", "south"],
    "left": ["left", "leftwards", "to the west"],
    "right": ["right", "rightwards", "to the east"],

    # Text Contexts
    "text_content": [
        "Hello world", "search query", "user@example.com", "password123", 
        "Python tutorial", "New York City", "The quick brown fox", 
        "Meeting notes", "TODO list", "123-456-7890", 
        "Buy milk", "Address: 123 Main St", "Confirm", "John Doe"
    ],
    
    # Termination
    "end_action": ["Stop task", "Finish interaction", "Task complete", "End the session", "Done"]
}

# Templates for Action Execution (Text -> Token)
# Added "[Action]" prefix to strongly condition the model
ACTION_TEMPLATES = [
    "[Action] Perform a step for the following action: {instruction}",
    "[Action] Perform a step for the following action: {instruction}", 
    "[Action] Please {instruction_lower}",
    "[Action] Action: {instruction}",
    "[Action] Execute: {instruction}",
    "[Action] I need you to {instruction_lower}",
    "[Action] Task: {instruction}",
]

# ================= DEFINITIONS & TEMPLATES (EXPLANATION) =================

# Natural Language Definitions for Tokens
TOKEN_DEFINITIONS = {
    # Movement
    "<MOVE_UP_FAR>":    "move the cursor upwards by a large distance, specifically 200 pixels",
    "<MOVE_UP_MID>":    "move the cursor upwards by a medium distance, specifically 30 pixels",
    "<MOVE_UP_CLO>":    "move the cursor upwards by a tiny distance, specifically 5 pixels",
    
    "<MOVE_DOWN_FAR>":  "move the cursor downwards by a large distance, specifically 200 pixels",
    "<MOVE_DOWN_MID>":  "move the cursor downwards by a medium distance, specifically 30 pixels",
    "<MOVE_DOWN_CLO>":  "move the cursor downwards by a tiny distance, specifically 5 pixels",
    
    "<MOVE_LEFT_FAR>":  "move the cursor to the left by a large distance, specifically 200 pixels",
    "<MOVE_LEFT_MID>":  "move the cursor to the left by a medium distance, specifically 30 pixels",
    "<MOVE_LEFT_CLO>":  "move the cursor to the left by a tiny distance, specifically 5 pixels",
    
    "<MOVE_RIGHT_FAR>": "move the cursor to the right by a large distance, specifically 200 pixels",
    "<MOVE_RIGHT_MID>": "move the cursor to the right by a medium distance, specifically 30 pixels",
    "<MOVE_RIGHT_CLO>": "move the cursor to the right by a tiny distance, specifically 5 pixels",

    # Interaction
    "<CLICK_SHORT>":    "perform a single, short click at the current cursor location",
    "<CLICK_LONG>":     "perform a long press (press and hold) at the current cursor location",

    # Navigation
    "<GO_BACK>":        "navigate back to the previous page or screen in the system history",
    "<GO_HOME>":        "navigate directly to the home screen or main dashboard",

    # Scrolling
    "<SCROLL_UP>":      "scroll the page content upwards (swiping down physically)",
    "<SCROLL_DOWN>":    "scroll the page content downwards (swiping up physically)",
    "<SCROLL_LEFT>":    "scroll the content horizontally to the left",
    "<SCROLL_RIGHT>":   "scroll the content horizontally to the right",

    # Text
    "<TEXT_START>":     "mark the beginning of a text input sequence",
    "<TEXT_END>":       "mark the end of a text input sequence",

    # Termination
    "<END_ACTION>":     "terminate the current task sequence indicating completion"
}

# Question Templates for Explanation (Token -> Definition)
# Added "[Define]" prefix to distinguish from action execution
QA_QUESTIONS = [
    "[Define] What does the token {token} mean?",
    "[Define] Explain the function of {token}.",
    "[Define] What action does {token} perform?",
    "[Define] Define {token}.",
    "[Define] Please describe the semantic meaning of {token}.",
    "[Define] In the GUI agent, what is {token}?",
]

QA_ANSWERS = [
    "The token {token} is used to {definition}.",
    "{token} represents an action to {definition}.",
    "It means to {definition}.",
    "This token triggers the agent to {definition}.",
    "Function: {definition}.",
]

# ================= TOKEN META =================
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
    "TEXT_INPUT_SEQUENCE": {"type": "text_seq"},

    # --- Termination ---
    "<END_ACTION>":     {"type": "end"},
}

# ================= LOGIC =================

def generate_action_sample(token_key, meta):
    """Generates (Instruction, ActionToken) pair."""
    t_type = meta["type"]
    
    if t_type == "move":
        verb = random.choice(VOCAB["move_verb"])
        direction = random.choice(VOCAB[meta["dir"]])
        distance_desc = random.choice(VOCAB[f"{meta['dist']}_desc"])
        
        case = random.randint(1, 5)
        if case == 1: instr = f"{verb} {direction} {distance_desc}"
        elif case == 2: instr = f"{verb} the cursor {distance_desc} {direction}"
        elif case == 3: instr = f"{distance_desc} {verb.lower()} {direction}"
        elif case == 4: instr = f"Make a {verb.lower()} {direction} {distance_desc}"
        else: instr = f"{verb} {direction}, make it {distance_desc}"
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
            instr = random.choice(["Go back", "Return to previous page", "Navigate back", "Click the back button"])
        else:
            instr = random.choice(["Go to home", "Return to dashboard", "Navigate to the main menu", "Press the home button"])
        return instr, token_key

    elif t_type == "text_seq":
        content = random.choice(VOCAB["text_content"])
        verb = random.choice(VOCAB["type_verb"])
        target = random.choice(VOCAB["ui_element"])
        case = random.randint(1, 3)
        if case == 1: instr = f"{verb} '{content}'"
        elif case == 2: instr = f"{verb} '{content}' into the {target}"
        else: instr = f"Input the text '{content}'"
        response = f"<TEXT_START> {content} <TEXT_END>"
        return instr, response

    elif t_type == "end":
        instr = random.choice(VOCAB["end_action"])
        return instr, token_key
    
    return "Perform action", token_key

def apply_action_template(instruction):
    template = random.choice(ACTION_TEMPLATES)
    if "{instruction_lower}" in template:
        return template.format(instruction_lower=instruction.lower())
    return template.format(instruction=instruction)

def generate_explanation_sample(token, definition):
    """Generates (Question about token, Definition) pair."""
    q_temp = random.choice(QA_QUESTIONS)
    a_temp = random.choice(QA_ANSWERS)
    
    question = q_temp.format(token=token)
    answer = a_temp.format(token=token, definition=definition)
    
    return question, answer

# ================= MAIN =================

def main():
    print(f"Generating Data...")
    print(f"1. Action Samples: {SAMPLES_PER_ACTION} per token logic (Requires Image)")
    print(f"2. Explanation Samples: {SAMPLES_PER_EXPLANATION} per token (Text Only)")
    
    all_records = []
    
    count_action = 0
    count_explain = 0
    
    # LOOP 1: Action Execution Data (Instruction -> Token)
    # We add a flag "type": "action" so the Dataset loader knows to inject an Image
    for token_key, meta in TOKEN_META.items():
        for _ in range(SAMPLES_PER_ACTION):
            instr, resp = generate_action_sample(token_key, meta)
            user_content = apply_action_template(instr)
            
            record = {
                "data_type": "action", # Flag for Dataset loader
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": resp}
                ]
            }
            all_records.append(record)
            count_action += 1
            
    # LOOP 2: Semantic Explanation Data (Token -> Definition)
    # We add a flag "type": "explanation" so the Dataset loader knows NOT to use an Image
    # or to use a specific "Text Only" mode if supported by the model
    for token, definition in TOKEN_DEFINITIONS.items():
        for _ in range(SAMPLES_PER_EXPLANATION):
            q, a = generate_explanation_sample(token, definition)
            
            record = {
                "data_type": "explanation", # Flag for Dataset loader
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ]
            }
            all_records.append(record)
            count_explain += 1
    
    # SHUFFLE THE DATA
    print("Shuffling data...")
    random.shuffle(all_records)
    
    # WRITE TO FILE
    print(f"Writing {len(all_records)} records to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")
            
    print(f"\nSuccessfully generated {count_action + count_explain} records.")
    print(f" - Action Execution: {count_action}")
    print(f" - Semantic Explanation: {count_explain}")
    print(f"Saved to: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()