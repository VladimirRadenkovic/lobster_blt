#!/usr/bin/env python3
"""
Test script for ByteLatentTransformer model using hydra configuration.
Tests model initialization, parameter counting, and forward/backward passes.
"""

import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import OmegaConf
import traceback
import sys

# Add the src directory to path so we can import lobster modules
sys.path.append('src')

from lobster.model.lm_base._latent_transformer import ByteLatentTransformer
from lobster.data import AmplifyLightningDataModule

def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def test_forward_backward(model: nn.Module, batch: dict, optimizer=None):
    """Test forward and backward passes."""
    print("\n=== Testing Forward/Backward Passes ===")
    try:
        model.train()
        
        print("Testing forward pass...")
        outputs = model(
            tokens=batch['input_ids'].squeeze(1).to('cuda'),
            patch_lengths=batch['patch_lengths'].squeeze(1).to('cuda'),
            attn_mask=batch['attention_mask'].to('cuda'),
            attn_patch_mask=batch['attention_mask_patch'].to('cuda'),
        )
        logits = outputs
        print(f"✓ Forward pass successful. Logits shape: {logits.shape}")
        
        print("Computing cross entropy loss...")
        batch_size, seq_len, vocab_size = logits.shape
        batch['labels'] = batch['labels'].to('cuda')
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(
            logits.view(-1, vocab_size),  
            batch['labels'].view(-1)      
        )
        
        print(f"✓ Loss computed: {loss.item():.4f}")
        
        print("Testing backward pass...")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        print("✓ Backward pass successful")

        gradient_check_passed = True
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                print(f"✗ No gradient for parameter: {name}")
                gradient_check_passed = False
            elif param.requires_grad and torch.isnan(param.grad).any():
                print(f"✗ NaN gradient for parameter: {name}")
                gradient_check_passed = False
        
        if gradient_check_passed:
            print("✓ All gradients are valid")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward/backward test failed: {e}")
        traceback.print_exc()
        return False

def test_forward_backward2(model: nn.Module, batch: dict):
    success = True
    for i in range(4):
        
        batch['input_ids'] = batch['input_ids'][i]
        mask = batch['input_ids'] != 1
        batch['input_ids'] = batch['input_ids'][mask].unsqueeze(0).unsqueeze(1)
        batch['labels'] = batch['labels'][i][mask].unsqueeze(0)

        success = success and test_forward_backward(model, batch)
    
    return success

def main():
    """Main test function."""
    print("=" * 60)
    print("ByteLatentTransformer Model Test")
    print("=" * 60)
    
    # Load configuration using OmegaConf (this is how Hydra works)
    config_path = "src/lobster/hydra_config/model/rlt_mlm_model.yaml"
    print(f"Loading configuration from: {config_path}")
    config = OmegaConf.load(config_path)
    print("✓ Configuration loaded successfully")
    

    print(f"Config keys: {list(config.keys())}")
    
    print("\n=== Initializing Model ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    

    model = ByteLatentTransformer(config)
    #model.init_weights()
    #model = model.to(device)
    
    print("✓ Model initialized successfully")

    total_params, trainable_params = count_parameters(model)
    print(f"\n=== Model Statistics ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (assuming float32)")
    
    encoder_params = sum(p.numel() for p in model.local_encoder.parameters())
    decoder_params = sum(p.numel() for p in model.local_decoder.parameters())
    global_params = sum(p.numel() for p in model.global_transformer.parameters())
    
    print(f"\nParameter breakdown:")
    print(f"  Local Encoder: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
    print(f"  Global Transformer: {global_params:,} ({global_params/total_params*100:.1f}%)")
    print(f"  Local Decoder: {decoder_params:,} ({decoder_params/total_params*100:.1f}%)")
    
    data_module = AmplifyLightningDataModule(
        tokenizer_max_length=2048,  # Set a reasonable sequence length
        root=Path("/rds/user/vr375/hpc-work/lobster/data"),  # Change this to an appropriate dataset directory
        seed=42,
        batch_size=4,
        num_workers=0,  # Set to >0 for parallel data loading
        pin_memory=False,
        shuffle_buffer_size=10000,
        model_type = config.model_type
    )

    print("\n==> Setting up the datasets")
    data_module.setup()
    train_loader = data_module.train_dataloader()

    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.98),
            eps=1e-12,
        )
    i = 0
    for batch in train_loader:
        i += 1
        if i > 10:
            break
        print(batch)
        batch['input_ids'] = batch['input_ids'].squeeze(1)
        batch['patch_lengths'] = batch['patch_lengths'].squeeze(1)
        batch['labels'] = batch['labels'].squeeze(1)

        success = test_forward_backward(model, batch, optimizer)
    
        if success:
            print("\n" + "=" * 60)
            print("✓ ALL TESTS PASSED! Model is working correctly.")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("✗ SOME TESTS FAILED!")
            print("=" * 60)


if __name__ == "__main__":
    main() 