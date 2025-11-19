# src/utils/tokenizer.py

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from .action import ACTION_BASE_EMBEDDING

def add_new_tokens(model, processor, base_embedding=ACTION_BASE_EMBEDDING):
    """
    Add custom GUI action tokens to the model, RESIZE embeddings, and UNTIE weights.
    
    Process:
    1. Add tokens to tokenizer.
    2. Resize model embeddings (Input & Output).
    3. Untie LM Head from Input Embeddings (Permanent architectural change).
    4. Smart Initialize BOTH Input and Output layers using semantic anchors.
    """
    
    tokenizer = processor.tokenizer
    
    # Ensure we are operating on the text backbone
    if hasattr(model, 'model'):
        text_model = model.model
    else:
        text_model = model
    
    # 1. Add tokens to tokenizer
    new_tokens = list(base_embedding.keys())
    existing_tokens = set(tokenizer.get_vocab().keys())
    tokens_to_add = [t for t in new_tokens if t not in existing_tokens]
    
    if tokens_to_add:
        tokenizer.add_special_tokens({'additional_special_tokens': tokens_to_add})
        print(f"\n[Tokenizer] Added {len(tokens_to_add)} new tokens. Total vocab size: {len(tokenizer)}")
    
    # 2. Resize model embeddings
    # This temporarily resizes tied weights if they are currently tied
    model.resize_token_embeddings(len(tokenizer))
    
    # 3. Untie LM Head from Input Embeddings
    # We do this IMMEDIATELY after resize to ensure independent training later.
    input_embeddings = text_model.get_input_embeddings()
    output_embeddings = text_model.get_output_embeddings()
    
    if output_embeddings is not None and output_embeddings.weight is input_embeddings.weight:
        print("[Tokenizer] Weight Tying detected. Untying LM Head from Input Embeddings...")
        # Clone the weights to create independent parameters
        new_lm_head_weight = input_embeddings.weight.clone().detach()
        output_embeddings.weight = torch.nn.Parameter(new_lm_head_weight)
        model.config.tie_word_embeddings = False
        print("[Tokenizer] Untying complete. Weights are now independent.")
    
    # 4. Smart Initialization (Semantic Anchoring)
    print(f"[Tokenizer] Initializing {len(base_embedding)} tokens (Input & Output layers)...")
    
    input_embeddings = text_model.get_input_embeddings()
    output_embeddings = text_model.get_output_embeddings()
    
    with torch.no_grad():
        for new_token, base_words in base_embedding.items():
            new_token_id = tokenizer.convert_tokens_to_ids(new_token)
            
            # Calculate Semantic Anchor (Mean of base words)
            anchor_vals = []
            for word in base_words:
                word_ids = tokenizer.encode(word, add_special_tokens=False)
                if len(word_ids) > 0:
                    # Use input embeddings to calculate the semantic mean
                    # (Input embeddings are the "definitions" of words)
                    word_emb = input_embeddings.weight[word_ids].mean(dim=0)
                    anchor_vals.append(word_emb)
            
            if anchor_vals:
                smart_init_emb = torch.stack(anchor_vals).mean(dim=0)
                
                # Initialize Input Embedding (The Definition)
                input_embeddings.weight[new_token_id] = smart_init_emb
                
                # Initialize Output Embedding (The Prediction)
                # Since we untied them, we must initialize output explicitely too
                output_embeddings.weight[new_token_id] = smart_init_emb
            else:
                print(f"Warning: No anchors found for {new_token}")
    
    print(f"[Tokenizer] All custom tokens initialized and layers untied!\n")
    
    return model, processor


def save_model(model, processor, output_dir):
    """Save model and processor."""
    print(f"[Saver] Saving model with additional tokens to {output_dir}...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"[Saver] Model successfully saved!")