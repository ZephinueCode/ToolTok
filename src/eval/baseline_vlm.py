# src/eval/baseline_vlm.py

import torch
import base64
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
    It relies on Prompt Engineering to force the model to output our specific Action Tokens.
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

    def parse_api_response(self, content: str) -> str:
        """Extracts the first valid token from the API response."""
        content = content.strip()
        # Priority: Exact Match
        if content in ACTION_TOKENS:
            return content
        
        # Fallback: Search for token in string (e.g., "I will use <CLICK_SHORT>")
        for token in ACTION_TOKENS:
            if token in content:
                return token
                
        return None

    def run_trajectory(
        self,
        input_text: str,
        ground_truth_data: List[dict],
        max_steps: int = 10
    ) -> Tuple[AgentTrajectory, None, None]:
        
        # 1. Init State
        gt_traj = GTTrajectory(ground_truth_data)
        agent_traj = AgentTrajectory(input_text, gt_traj.get_step(0))
        
        print(f"\n=== API TASK: {input_text} ===")

        for step in range(max_steps):
            agent_traj.step_count += 1
            current_gt = gt_traj.get_step(agent_traj.current_gt_step_idx)
            
            # 2. Prepare Image
            base64_img = encode_image_to_base64(agent_traj.get_current_view())
            
            # 3. Build Messages
            messages = [
                {"role": "system", "content": BASELINE_API_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}},
                        {"type": "text", "text": f"Instruction: {agent_traj.global_question}\nCurrent Cursor: {agent_traj.cursor_pos}\nNext Action Token:"}
                    ]
                }
            ]
            
            # 4. Call API
            try:
                completion = self.client.chat.completions.create(
                    model=HP.BASELINE_MODEL_NAME,
                    messages=messages,
                    temperature=0.0, # Greedy for deterministic eval
                    max_tokens=20
                )
                response_text = completion.choices[0].message.content
            except Exception as e:
                print(f"[API Error] Step {step}: {e}")
                agent_traj.failed = 1 # Execution Error
                break
            
            # 5. Parse Token
            token_text = self.parse_api_response(response_text)
            print(f"Step {step+1}: API said '{response_text}' -> Parsed: {token_text}")
            
            if not token_text:
                print(" -> [Fail] No valid token found.")
                agent_traj.failed = 1 # Invalid
                agent_traj.tools.append(f"INVALID: {response_text}")
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
                # Check Type Consistency (ScreenSpot is Click only)
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

        # Note: Returns None for IDs/Logprobs as API doesn't provide them easily
        return agent_traj, None, None