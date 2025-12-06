# src/utils/action.py

"""
GUI Agent Action Space Definition & Semantic Anchors.

This dictionary maps specific action tokens to a list of natural language words.
These words are used to initialize the embeddings of the action tokens, 
giving them a "Semantic Anchor" in the vector space.
"""

ACTION_BASE_EMBEDDING = {
    # ===========================
    # General Action Head
    # ===========================
    "<ACTION>": ["action", "perform", "execute", "interact", "interface", "operate"],
    
    # ===========================
    # Movement (Cursor Control)
    # ===========================

    # Up
    "<MOVE_UP_FAR>":   ["move", "cursor", "up", "top", "far", "jump", "leap"],
    "<MOVE_UP_MID>":   ["move", "cursor", "up", "medium", "shift"],
    "<MOVE_UP_CLO>":   ["move", "cursor", "up", "near", "tiny", "nudge", "slight"],

    # Down
    "<MOVE_DOWN_FAR>": ["move", "cursor", "down", "bottom", "far", "jump", "leap"],
    "<MOVE_DOWN_MID>": ["move", "cursor", "down", "medium", "shift"],
    "<MOVE_DOWN_CLO>": ["move", "cursor", "down", "near", "tiny", "nudge", "slight"],

    # Left
    "<MOVE_LEFT_FAR>": ["move", "cursor", "left", "west", "far", "jump", "leap"],
    "<MOVE_LEFT_MID>": ["move", "cursor", "left", "medium", "shift"],
    "<MOVE_LEFT_CLO>": ["move", "cursor", "left", "near", "tiny", "nudge", "slight"],

    # Right
    "<MOVE_RIGHT_FAR>": ["move", "cursor", "right", "east", "far", "jump", "leap"],
    "<MOVE_RIGHT_MID>": ["move", "cursor", "right", "medium", "shift"],
    "<MOVE_RIGHT_CLO>": ["move", "cursor", "right", "near", "tiny", "nudge", "slight"],

    # ===========================
    # Interaction (Clicking)
    # ===========================
    "<CLICK_SHORT>": ["click", "tap", "select", "press", "touch", "brief", "mouse"],
    "<CLICK_LONG>":  ["click", "hold", "press", "long", "keep", "sustain"],

    # ===========================
    # Navigation (System)
    # ===========================
    "<GO_BACK>": ["navigate", "back", "return", "previous", "history", "reverse", "phone", "mobile"],
    "<GO_HOME>": ["navigate", "home", "main", "dashboard", "desktop", "launcher", "phone", "mobile"],

    # ===========================
    # Scrolling
    # ===========================
    "<SCROLL_UP>":    ["scroll", "swipe", "pan", "up", "top", "content"],
    "<SCROLL_DOWN>":  ["scroll", "swipe", "pan", "down", "bottom", "content"],
    "<SCROLL_LEFT>":  ["scroll", "swipe", "pan", "left", "west", "content"],
    "<SCROLL_RIGHT>": ["scroll", "swipe", "pan", "right", "east", "content"],

    # ===========================
    # Text Input
    # ===========================
    "<TEXT_START>": ["type", "write", "input", "keyboard", "start", "focus", "activate"],
    "<TEXT_END>":   ["type", "finish", "enter", "submit", "confirm", "return"],
    
    # ===========================
    # Termination
    # ===========================
    "<END_ACTION>": ["stop", "finish", "complete", "task", "done", "halt", "cease"],
}

# Helper list for easy access to keys
ACTION_TOKENS = list(ACTION_BASE_EMBEDDING.keys())