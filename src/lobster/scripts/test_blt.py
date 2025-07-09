import torch
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from lobster.data import AmplifyLightningDataModule
from lobster.model.lm_base._latent_transformer import ByteLatentTransformer
from lobster.config import BLTConfig

@hydra.main(version_base=None, config_path="../configs", config_name="model/blt")
def main(cfg: DictConfig):
    # Convert config to BLTConfig
    model_config = BLTConfig(**cfg.model)
    
    # Initialize model
    model = ByteLatentTransformer(model_config)
    model = model.cuda()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Initialize data module
    data_module = AmplifyLightningDataModule(
        tokenizer_max_length=cfg.data.tokenizer_max_length,
        root=Path(cfg.data.root),
        seed=cfg.data.seed,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        shuffle_buffer_size=cfg.data.shuffle_buffer_size,
    )

    # Setup data module
    data_module.setup()
    train_loader = data_module.train_dataloader()

    # Get a batch
    batch = next(iter(train_loader))
    tokens = batch["input_ids"].squeeze(1).cuda()
    patch_lengths = batch["patch_lengths"].squeeze(1).cuda()

    # Forward pass
    with torch.cuda.amp.autocast():
        output = model(
            tokens=tokens,
            patch_lengths=patch_lengths,
        )

    print(f"\nInput shape: {tokens.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Patch lengths shape: {patch_lengths.shape}")

if __name__ == "__main__":
    main() 