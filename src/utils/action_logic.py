# src/utils/action_logic.py

# Coordinate deltas for movement tokens
MOVE_DELTAS = {
    "<MOVE_UP_FAR>":    (0, -500),
    "<MOVE_UP_MID>":    (0, -150),
    "<MOVE_UP_CLO>":    (0, -30),
    
    "<MOVE_DOWN_FAR>":  (0, 500),
    "<MOVE_DOWN_MID>":  (0, 150),
    "<MOVE_DOWN_CLO>":  (0, 30),
    
    "<MOVE_LEFT_FAR>":  (-500, 0),
    "<MOVE_LEFT_MID>":  (-150, 0),
    "<MOVE_LEFT_CLO>":  (-30, 0),
    
    "<MOVE_RIGHT_FAR>": (500, 0),
    "<MOVE_RIGHT_MID>": (150, 0),
    "<MOVE_RIGHT_CLO>": (30, 0),
}

def get_action_type(token: str) -> str:
    """
    Classifies tokens into specific action categories.
    """
    if "MOVE" in token:
        return "move"
    
    elif "CLICK" in token:
        return "click"
        
    elif "SCROLL" in token:
        return "scroll"
        
    elif "GO_" in token:
        return "nav"
        
    elif "TEXT" in token:
        return "text"
        
    elif "END_ACTION" in token:
        return "end"
        
    else:
        return "unknown"

def is_interaction(action_type: str) -> bool:
    """Helper to check if an action interacts with the UI (vs just moving cursor)."""
    return action_type in ["click", "scroll", "nav", "text"]