# src/eval/baseline_vlm.py

import torch
import base64
import re
import os  # [FIX] Added missing import
from io import BytesIO
from PIL import Image
from typing import Tuple, List
from openai import OpenAI

from ..utils.parameters import HYPERPARAMS as HP
from ..utils.prompts import BASELINE_API_PROMPT
from ..utils.action_logic import MOVE_DELTAS, get_action_type
from ..utils.action import ACTION_TOKENS
from ..tools.runner import Runner, ToolData, AgentTrajectory, GTTrajectory

def encode_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

class BaselineAPIRunner(Runner):
    """
    Runner that uses an external VLM API (e.g., Qwen-Max/235B) to control the agent.
    Supports Chain-of-Thought (CoT) reasoning via Prompt Engineering.
    """
    def __init__(self):
        # No super().__init__ needed if we don't use local toolbox models
        print(f"[BaselineAPIRunner] Initializing API Client...")
        print(f" - Base URL: {HP.VLM_BASE_URL}")
        print(f" - Model: {HP.BASELINE_MODEL_NAME}")
        
        self.client = OpenAI(
            api_key=HP.VLM_API_KEY,
            base_url=HP.VLM_BASE_URL
        )

    def parse_api_response(self, content: str) -> Tuple[str, str]:
        """
        Parses the CoT response to separate Reasoning from Action.
        Returns: (reasoning_text, action_token)
        """
        content = content.strip()
        reasoning = ""
        action_token = None

        # 1. Try to split by "Action:" marker
        if "Action:" in content:
            parts = content.split("Action:")
            reasoning = parts[0].replace("Reasoning:", "").strip()
            potential_action = parts[1].strip()
        else:
            # Fallback: Treat whole content as potential mix, no clear reasoning split
            reasoning = "No explicit reasoning section found."
            potential_action = content

        # 2. Extract the valid token from the Action part
        # Look for tokens like <MOVE_...> or <TEXT_START>...
        # We use a regex that captures the defined token format
        
        # Priority 1: Exact match from ACTION_TOKENS list in the 'Action' section
        for token in ACTION_TOKENS:
            if token in potential_action:
                action_token = token
                break
        
        # Priority 2: If not found, try regex for general <TOKEN> pattern
        if not action_token:
            match = re.search(r"(<[A-Z_]+(?: .*?)?>)", potential_action)
            if match:
                action_token = match.group(1)

        return reasoning, action_token

    def run_trajectory(
        self,
        input_text: str,
        ground_truth_data: List[dict],
        max_steps: int = 10
    ) -> Tuple[AgentTrajectory, None, None]:
        
        # 1. Init State
        gt_traj = GTTrajectory(ground_truth_data)
        agent_traj = AgentTrajectory(input_text, gt_traj.get_step(0))
        
        # [FIX] Setup Debug Directory
        debug_dir = "debug_visuals"
        os.makedirs(debug_dir, exist_ok=True)
        
        print(f"\n=== API TASK: {input_text} ===")

        for step in range(max_steps):
            agent_traj.step_count += 1
            current_gt = gt_traj.get_step(agent_traj.current_gt_step_idx)
            
            # 2. Prepare Image
            current_view = agent_traj.get_current_view()
            img_w, img_h = current_view.size
            
            base64_img = encode_image_to_base64(current_view)
            
            # 3. Build Messages
            # [FIX] Added Image Size explicitly to the prompt
            prompt_text = (
                f"Instruction: {agent_traj.global_question}\n"
                f"Provide your Reasoning and Action:"
            )
            
            messages = [
                {"role": "system", "content": BASELINE_API_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            
            # 4. Call API
            try:
                completion = self.client.chat.completions.create(
                    model=HP.BASELINE_MODEL_NAME,
                    messages=messages,
                    temperature=0.0, # Greedy for deterministic eval
                    max_tokens=512   # Increased for CoT
                )
                response_text = completion.choices[0].message.content
            except Exception as e:
                print(f"[API Error] Step {step}: {e}")
                agent_traj.failed = 1 # Execution Error
                break
            
            # 5. Parse Token and Reasoning
            reasoning, token_text = self.parse_api_response(response_text)
            
            # [LOGGING] Show the reasoning in console
            print(f"\n[Step {step+1}]")
            print(f"  Ref: {reasoning}")
            print(f"  Act: {token_text}")
            
            if not token_text:
                print(" -> [Fail] No valid token found.")
                agent_traj.failed = 1 # Invalid
                agent_traj.tools.append(f"INVALID: {response_text[:20]}...")
                break
            
            agent_traj.tools.append(token_text)

            # 6. Execute Logic (Same as Standard Runner)
            action_type = get_action_type(token_text)
            
            # --- Move ---
            if action_type == "move":
                if token_text in MOVE_DELTAS:
                    dx, dy = MOVE_DELTAS[token_text]
                    agent_traj.move_cursor(dx, dy)
                else:
                    agent_traj.failed = 1
                    break
            
            # --- Interact ---
            elif action_type in ["click", "scroll", "text", "nav"]:
                # Check Type Consistency
                if action_type != current_gt.action_type and current_gt.action_type == "click":
                      print(f" -> [Fail] Wrong Type (Expected {current_gt.action_type})")
                      agent_traj.failed = 2
                      break

                # Check Position Hit
                if self.check_hit(agent_traj.cursor_pos, current_gt.bbox):
                    print(" -> HIT! Target Reached.")
                    if agent_traj.current_gt_step_idx < gt_traj.total_steps - 1:
                        next_step = gt_traj.get_step(agent_traj.current_gt_step_idx + 1)
                        agent_traj.advance_to_next_gt_step(next_step)
                    else:
                        agent_traj.current_gt_step_idx += 1
                        agent_traj.gt_steps_passed += 1 
                else:
                    print(" -> MISS! Clicked outside.")
                    agent_traj.failed = 3
                    break
            
            # --- End ---
            elif action_type == "end":
                if agent_traj.current_gt_step_idx >= gt_traj.total_steps:
                    agent_traj.failed = 0 # Success
                else:
                    print(" -> [Fail] Ended too early.")
                    agent_traj.failed = 4
                break
            
            else:
                agent_traj.failed = 1
                break

        # Timeout
        if agent_traj.failed == 0 and agent_traj.step_count >= max_steps:
             if agent_traj.tools and "END_ACTION" not in agent_traj.tools[-1]:
                 agent_traj.failed = 4

        return agent_traj, None, None