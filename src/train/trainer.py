import torch
import numpy as np
from typing import Optional, Union, List, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.trainer_utils import EvalLoopOutput
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from tqdm import tqdm

from ..tools.runner import Runner, ToolData, AgentTrajectory
from .reward import batch_compute_rewards
from ..utils.action_logic import MOVE_DELTAS, get_action_type
from ..utils.parameters import HYPERPARAMS as HP

class ToolGRPOTrainer(GRPOTrainer):
    """
    GRPO Trainer tailored for GUI Agents.
    Includes specialized generation logic and custom evaluation loop.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        args: GRPOConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        runner: Optional[Runner] = None,
        max_tool_steps: int = 10,
        beta: float = 0.1,
        **kwargs
    ):
        super().__init__(
            model=model,
            reward_funcs=self._dummy_reward,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            **kwargs
        )
        
        self.max_tool_steps = max_tool_steps
        self.runner = runner or Runner()
        self.beta = beta
        
        # --- Embedding Freeze & Untie Logic ---
        print(f"\n[ToolGRPOTrainer] Configuring Model Layers...")
        if hasattr(self.model, "get_input_embeddings"):
            embeddings = self.model.get_input_embeddings()
            if embeddings is not None:
                embeddings.requires_grad_(False)
                print(" -> Input embeddings FROZEN.")

        if self.model.config.tie_word_embeddings:
            print(" -> Weight Tying detected. Untying LM Head...")
            if hasattr(self.model, "get_output_embeddings"):
                lm_head = self.model.get_output_embeddings()
                if lm_head.weight is embeddings.weight:
                    new_weight = embeddings.weight.clone().detach()
                    lm_head.weight = torch.nn.Parameter(new_weight)
                    self.model.config.tie_word_embeddings = False
                lm_head.requires_grad_(True)
                print(" -> LM Head UNTIED and set to TRAINABLE.")
        else:
            if hasattr(self.model, "get_output_embeddings"):
                self.model.get_output_embeddings().requires_grad_(True)

    def _dummy_reward(self, prompts, completions, **kwargs):
        return [0.0] * len(prompts)

    def _prepare_inputs(self, inputs: Union[dict, list]) -> dict:
        """
        Intercept batch processing to run generation logic (Training Phase).
        """
        if isinstance(inputs, dict):
            if "question" in inputs and isinstance(inputs["question"], list):
                batch_size = len(inputs["question"])
                inputs_list = [
                    {
                        "question": inputs["question"][i],
                        "image": inputs["image"][i],
                        "ground_truth_traj": inputs["ground_truth_traj"][i],
                    }
                    for i in range(batch_size)
                ]
            else:
                if "per_token_logps" in inputs: return inputs
                raise ValueError("Input dict must contain lists for batching.")
        else:
            inputs_list = inputs
            
        return self._generate_and_score_completions(inputs_list)

    def _generate_and_score_completions(self, inputs: list[dict]) -> dict:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        batch_size = len(inputs)
        
        # 1. Online Generation (No Gradients)
        all_trajectories = []
        all_token_ids = []
        all_old_logprobs = []
        
        for sample in inputs:
            tool_input = ToolData(image=sample['image'], text=sample['question'])
            
            traj, token_ids, logprobs = self.runner.run_trajectory(
                model=self.model,
                processor=self.processing_class,
                input_data=tool_input,
                ground_truth_bbox=sample['ground_truth_traj'], # Pass the list
                max_steps=self.max_tool_steps,
                temperature=self.args.temperature if mode == "train" else 0.0
            )
            all_trajectories.append(traj)
            all_token_ids.append(token_ids)
            all_old_logprobs.append(logprobs)

        # 2. Compute Rewards
        rewards = batch_compute_rewards(all_trajectories)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        
        # 3. Advantage Norm (GRPO)
        all_rewards_gathered = self.accelerator.gather(rewards_tensor)
        advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        
        # 4. Recompute Gradients
        all_current_logprobs = self._recompute_logprobs(all_trajectories, all_token_ids)
        
        # 5. Padding & Collation
        max_len = max(len(ids) for ids in all_token_ids) if all_token_ids else 1
        
        pad_ids, pad_mask, pad_curr, pad_old = [], [], [], []
        pad_token_id = self.processing_class.tokenizer.pad_token_id
        
        for i in range(batch_size):
            length = len(all_token_ids[i])
            diff = max_len - length
            
            ids = all_token_ids[i] + [pad_token_id] * diff
            mask = [1] * length + [0] * diff
            pad_ids.append(torch.tensor(ids, device=device, dtype=torch.long))
            pad_mask.append(torch.tensor(mask, device=device, dtype=torch.long))
            
            curr = all_current_logprobs[i]
            old = torch.tensor(all_old_logprobs[i], device=device, dtype=torch.float32)
            
            if diff > 0:
                curr = torch.cat([curr, torch.zeros(diff, device=device, dtype=curr.dtype)])
                old = torch.cat([old, torch.zeros(diff, device=device, dtype=old.dtype)])
            
            pad_curr.append(curr)
            pad_old.append(old)
            
        return {
            "completion_mask": torch.stack(pad_mask),
            "advantages": advantages.detach(),
            "per_token_logps": torch.stack(pad_curr),
            "old_token_logps": torch.stack(pad_old),
            "completion_ids": torch.stack(pad_ids)
        }

    def _recompute_logprobs(self, trajectories: List[AgentTrajectory], token_ids_list: List[List[int]]):
        """Re-run forward pass for gradient calculation."""
        all_logprobs = []
        
        for traj, token_ids in zip(trajectories, token_ids_list):
            if not token_ids:
                all_logprobs.append(torch.tensor([], device=self.model.device))
                continue
            
            traj_logprobs = []
            history_tokens = []
            
            init_img = traj.images[0]
            curr_cursor = [init_img.width // 2, init_img.height // 2]
            
            for step_i, target_id in enumerate(token_ids):
                step_image = traj.images[step_i]
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": step_image},
                            {"type": "text", "text": f"[Action] Perform a step for the following action: {traj.global_question}\nCurrent Cursor: {curr_cursor}"}
                        ]
                    }
                ]
                
                if history_tokens:
                    hist_str = "\nAction History:\n" + "\n".join([f"- {t}" for t in history_tokens])
                    messages[1]["content"].append({"type": "text", "text": hist_str})

                text = self.processing_class.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processing_class(
                    text=[text], 
                    images=[step_image], 
                    return_tensors="pt", 
                    padding=True
                ).to(self.model.device)
                
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]
                lp = torch.log_softmax(logits, dim=-1)[target_id]
                traj_logprobs.append(lp)
                
                token_text = traj.tools[step_i]
                history_tokens.append(token_text)
                
                # Sync Cursor Logic
                from ..utils.action_logic import MOVE_DELTAS, get_action_type
                atype = get_action_type(token_text)
                
                if atype == "move" and token_text in MOVE_DELTAS:
                    dx, dy = MOVE_DELTAS[token_text]
                    w, h = step_image.size
                    nx = max(0, min(curr_cursor[0] + dx, w))
                    ny = max(0, min(curr_cursor[1] + dy, h))
                    curr_cursor = [nx, ny]
                
                elif atype == "interact":
                    if step_i + 1 < len(traj.images):
                        next_img = traj.images[step_i + 1]
                        if next_img.size != step_image.size or next_img is not step_image:
                            curr_cursor = [next_img.width // 2, next_img.height // 2]

            all_logprobs.append(torch.stack(traj_logprobs))
            
        return all_logprobs

    def compute_loss(self, model, inputs, return_outputs=False):
        if return_outputs: raise ValueError("Not supported")
        
        per_token_logps = inputs["per_token_logps"]
        old_token_logps = inputs["old_token_logps"]
        mask = inputs["completion_mask"]
        advantages = inputs["advantages"]
        
        traj_logps = (per_token_logps * mask).sum(dim=1)
        traj_old_logps = (old_token_logps * mask).sum(dim=1)
        
        pg_loss = -(advantages * traj_logps).mean()
        kl_penalty = (traj_logps - traj_old_logps).pow(2).mean()
        
        loss = pg_loss + self.beta * kl_penalty
        
        if self.model.training:
            self.log({
                "loss/total": loss.item(),
                "loss/pg": pg_loss.item(),
                "loss/kl": kl_penalty.item()
            })
            
        return loss

    # =========================================================================
    # CUSTOM EVALUATION LOOP (Metric-Centric)
    # =========================================================================
    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Run evaluation on the held-out set using Greedy Decoding (temperature=0).
        Computes REAL metrics: Average Reward, Success Rate, Average Steps.
        Ignores loss calculation entirely.
        """
        # 1. Prepare Model
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()
        
        print(f"\n[{metric_key_prefix}] Starting Evaluation Loop on {len(dataloader.dataset)} samples...")
        print(f"[{metric_key_prefix}] Strategy: Greedy Decoding (Temp=0.0)")

        # 2. Metrics Container
        all_rewards = []
        success_count = 0
        total_steps = 0
        failed_counts = {1:0, 2:0, 3:0, 4:0} # Invalid, Type, Pos, NoStop

        # 3. Iterate
        # Note: We iterate manually to handle batching logic specifically for Runner
        for step, batch_inputs in tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating"):
            
            # Handle DataCollator vs raw DataLoader differences
            # Standard HF DataLoader yields dict of stacked tensors/lists
            # Our custom Dataset returns dicts.
            
            # Convert batch dict to list of samples
            if isinstance(batch_inputs, dict):
                bs = len(batch_inputs["question"])
                inputs_list = [
                    {
                        "question": batch_inputs["question"][i],
                        "image": batch_inputs["image"][i],
                        "ground_truth_traj": batch_inputs["ground_truth_traj"][i],
                    }
                    for i in range(bs)
                ]
            else:
                # Fallback
                inputs_list = batch_inputs

            # Run Generation (No Gradients)
            for sample in inputs_list:
                tool_input = ToolData(image=sample['image'], text=sample['question'])
                
                # Run with Temperature = 0.0 for deterministic eval
                traj, _, _ = self.runner.run_trajectory(
                    model=model,
                    processor=self.processing_class,
                    input_data=tool_input,
                    ground_truth_bbox=sample['ground_truth_traj'],
                    max_steps=self.max_tool_steps,
                    temperature=0.0 
                )
                
                # Score
                reward = batch_compute_rewards([traj])[0]
                all_rewards.append(reward)
                
                # Stats
                total_steps += traj.step_count
                if traj.failed == 0:
                    success_count += 1
                else:
                    if traj.failed in failed_counts:
                        failed_counts[traj.failed] += 1

        # 4. Aggregation
        num_samples = len(all_rewards)
        if num_samples == 0: return EvalLoopOutput(predictions=None, label_ids=None, metrics={}, num_samples=0)

        avg_reward = sum(all_rewards) / num_samples
        success_rate = success_count / num_samples
        avg_steps = total_steps / num_samples
        
        metrics = {
            f"{metric_key_prefix}_reward": avg_reward,
            f"{metric_key_prefix}_success_rate": success_rate,
            f"{metric_key_prefix}_avg_steps": avg_steps,
            f"{metric_key_prefix}_fail_invalid": failed_counts[1] / num_samples,
            f"{metric_key_prefix}_fail_miss": failed_counts[3] / num_samples, # Wrong Position
            f"{metric_key_prefix}_fail_timeout": failed_counts[4] / num_samples,
        }
        
        print(f"\n{'='*40}")
        print(f"EVALUATION RESULTS ({num_samples} samples)")
        print(f"Avg Reward:   {avg_reward:.2f}")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Avg Steps:    {avg_steps:.2f}")
        print(f"{'='*40}\n")

        # 5. Return standardized Output
        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=num_samples
        )