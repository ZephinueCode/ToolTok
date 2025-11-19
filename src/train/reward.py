# src/train/reward.py

from typing import List
# Assuming AgentTrajectory is imported or we treat it as a duck-typed object
# from ..tools.runner import AgentTrajectory 

def compute_reward(agent_traj) -> float:
    """
    Computes scalar reward based on AgentTrajectory status and step count.
    
    Status Codes (agent_traj.failed):
    - 0: Success (All GT steps completed, END_ACTION outputted)
    - 1: Invalid Token (Hallucination)
    - 2: Wrong Type (e.g., Moving when should click - logic dependent)
    - 3: Wrong Position (Clicked outside target bbox)
    - 4: No Stop / Timeout (Max steps reached or stopped early)
    
    Scoring Rules:
    - Success: 100 - steps
    - Invalid: -100 + (Progress% * 100)
    - Wrong Type: -50 + (Progress% * 100)
    - Wrong Pos: -30 + (Progress% * 100)
    - No Stop: 50 - steps (Positive reward for partial correctness, heavily penalized by steps)
    """
    
    # Safeguard against division by zero if total_gt_steps is not set properly
    total_gt = getattr(agent_traj, 'total_gt_steps', 1)
    passed_gt = getattr(agent_traj, 'gt_steps_passed', 0)
    step_count = getattr(agent_traj, 'step_count', 0)
    
    # Calculate Progress Bonus (0 to 100)
    if total_gt > 0:
        progress_bonus = (passed_gt / total_gt) * 100.0
    else:
        progress_bonus = 0.0

    # Step Penalty (Encourage efficiency)
    step_penalty = step_count * 1.0
    
    status = agent_traj.failed
    
    # 1. Success
    if status == 0:
        return 100.0 - step_penalty
        
    # 2. Invalid Token
    elif status == 1:
        return -100.0 + progress_bonus
        
    # 3. Wrong Tool Type
    elif status == 2:
        return -50.0 + progress_bonus
        
    # 4. Wrong Position (Missed Target)
    elif status == 3:
        return -30.0 + progress_bonus
        
    # 5. No Stop / Timeout / Incomplete
    # (User specified positive base 50 here)
    elif status == 4:
        return 50.0 - step_penalty
    
    # Fallback
    return 0.0

def batch_compute_rewards(trajectories: List[object], **kwargs) -> List[float]:
    """
    Batch wrapper for reward computation.
    """
    return [compute_reward(t) for t in trajectories]