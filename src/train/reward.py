# src/train/reward.py

import math
from typing import List, Dict, Any

def compute_euclidean_distance(pos1, pos2):
    return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])

def compute_reward(agent_traj) -> float:
    """
    Computes scalar reward based on:
    1. Progress Credit: (Completed Steps / Total GT Steps) * 100
    2. Status & Efficiency: Penalties for Invalid actions, Wrong Types, Timeouts.
    3. Spatial Accuracy (Dynamic): 
       - Close Miss (<50px): Better than Timeout (Encourage rational confidence).
       - Far Miss (>50px): Worse than Timeout (Discourage blind guessing).
    4. Shaping: Distance improvement and absolute proximity.
    
    Status Codes (agent_traj.failed):
    - 0: Success
    - 1: Invalid Token
    - 2: Wrong Action Type
    - 3: Wrong Position (Miss)
    - 4: Timeout
    """
    
    status = agent_traj.failed
    steps = agent_traj.step_count
    
    # ============================================================
    # 0. Context Extraction
    # ============================================================
    # Get total steps from the trajectory object
    total_gt_steps = getattr(agent_traj, 'total_gt_steps', 1)
    total_gt_steps = max(1, total_gt_steps) # Prevent div by zero
    
    gt_steps_passed = agent_traj.gt_steps_passed
    
    # Calculate final distance to the CURRENT active target
    final_dist = float('inf')
    if hasattr(agent_traj, 'target_bbox') and agent_traj.target_bbox:
        x1, y1, x2, y2 = agent_traj.target_bbox
        tx, ty = (x1 + x2) // 2, (y1 + y2) // 2
        
        if hasattr(agent_traj, 'cursor_path') and agent_traj.cursor_path:
            end_pos = agent_traj.cursor_path[-1]
            final_dist = compute_euclidean_distance(end_pos, (tx, ty))

    # ============================================================
    # 1. Progress Reward (Base Score)
    # ============================================================
    # Award points proportional to the completed portion of the task.
    # e.g., 2/4 steps done = 50 points.
    if status == 0:
        base_score = 100.0
    else:
        base_score = (gt_steps_passed / total_gt_steps) * 100.0

    # ============================================================
    # 2. Status & Dynamic Penalty Logic
    # ============================================================
    status_score = 0.0
    
    if status == 0:
        # Bonus for full completion
        status_score = +10.0
        
    elif status == 1: 
        # Invalid Token: Severe penalty to fix format
        status_score = -100.0
        
    elif status == 2: 
        # Wrong Action Type (e.g. Click instead of Scroll)
        status_score = -50.0
        
    elif status == 4: 
        # Timeout:
        # Represents "Exploring but ran out of time". 
        # We use this as a baseline (-15).
        status_score = -15.0 
        
    elif status == 3: 
        # Miss (Clicking at wrong location):
        # LOGIC: Close Miss > Timeout > Far Miss
        
        NEAR_MISS_THRESHOLD = 50.0 # pixels
        
        if final_dist < NEAR_MISS_THRESHOLD:
            # "Rational Confidence": The agent was very close. 
            # Penalty (-5) is smaller (better) than Timeout (-15).
            status_score = -5.0 
        else:
            # "Blind Guess": The agent clicked far away.
            # Penalty (-60) is much larger (worse) than Timeout (-15).
            # This encourages the agent to keep moving rather than guessing.
            status_score = -60.0

    # ============================================================
    # 3. Efficiency Penalty
    # ============================================================
    # Small cost per step to encourage taking the shortest path.
    step_penalty = steps * 0.5
    
    # ============================================================
    # 4. Spatial Shaping (Dense Reward)
    # ============================================================
    spatial_reward = 0.0
    
    # Only calculate if we have a valid path and didn't crash (status 1)
    if status != 1 and hasattr(agent_traj, 'cursor_path') and agent_traj.cursor_path:
        # A. Relative Improvement
        # Reward for getting closer compared to the starting point
        start_pos = agent_traj.cursor_path[0]
        if hasattr(agent_traj, 'target_bbox') and agent_traj.target_bbox:
             # Re-calculate target center (tx, ty already defined above)
             initial_dist = compute_euclidean_distance(start_pos, (tx, ty))
             dist_improvement = initial_dist - final_dist
             spatial_reward += dist_improvement * 0.1
        
        # B. Absolute Proximity
        # Reward simply for ending up near the target, regardless of start.
        max_screen_dist = 1414.0 # approx diagonal of 1000x1000
        if final_dist < max_screen_dist:
            # Linearly scale from 0 to +20 points based on closeness
            proximity_score = 20.0 * (1.0 - (final_dist / max_screen_dist))
            spatial_reward += proximity_score
    
    # ============================================================
    # 5. Total Sum
    # ============================================================
    total_reward = base_score + status_score - step_penalty + spatial_reward
    
    return total_reward

def batch_compute_rewards(trajectories: List[Any], **kwargs) -> List[float]:
    """
    Batch wrapper for reward computation.
    """
    return [compute_reward(t) for t in trajectories]