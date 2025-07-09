from pathlib import Path

from lightning import LightningDataModule
from torch import Generator
from torch.utils.data import DataLoader, Dataset

from lobster.datasets import AMPLIFYIterableDataset

from lobster.tokenization import AminoAcidTokenizerFast
from transformers import AutoTokenizer
from lobster.transforms import TokenizerTransform
from lobster.transforms._patching_transform import PatchifierTransform


class AmplifyLightningDataModule(LightningDataModule):
    def __init__(self,
        max_length: int,
        *,
        root: Path | str | None = None,
        seed: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = True,
        shuffle_buffer_size: int = 10000,
        max_num_patches: int = 512,
        model_type: str = "mlm",
    ) -> None:
        """Initialize an AmplifyLightningDataModule.

        Parameters
        ----------
        tokenizer_max_length : int
            Maximum length of the tokenized input. Should match the model's maximum input length.
        root : Path | str | None, optional
            Root directory where the datasets are stored. If None, the default directory will be used.
        seed : int, optional
            Random seed for reproducibility.
        batch_size : int, optional
            Batch size.
        num_workers : int, optional
            Number of workers for data loading.
        pin_memory : bool, optional
            Whether to pin memory for faster GPU transfer.
        shuffle_buffer_size : int, optional
            Size of the shuffle buffer for training datasets. Is for shuffling iterable datasets.


        """
        super().__init__()

        self._root = root
        self._max_length = max_length
        self._generator = Generator().manual_seed(seed)
        self._seed = seed
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._shuffle_buffer_size = shuffle_buffer_size
        self._max_num_patches = max_num_patches
        self._model_type = model_type

        # Initialize tokenizer transform for AMPLIFY dataset
        self._tokenizer_transform = TokenizerTransform(
            AminoAcidTokenizerFast(),
            padding="max_length",
            truncation=True,
            max_length=self._max_length
        )
        self._patchifier_transform = PatchifierTransform(
            AminoAcidTokenizerFast(),
            max_num_patches=max_num_patches,
            truncation=True,
            padding="max_length",
            random_truncation=True,
            max_seq_length=self._max_length,
            model_type=model_type,
        )


    def _get_dataset(self, split: str) -> Dataset:
        """Get a dataset instance with appropriate tokenizer transform."""

        return AMPLIFYIterableDataset(
            root=self._root,
            #transform=self._tokenizer_transform,
            transform=self._patchifier_transform,
            split=split,
            shuffle=(split == split),
            shuffle_buffer_size=self._shuffle_buffer_size,
            download=False,
        )

    
    def setup(self, stage: str | None = None) -> None:


        self._train_dataset = self._get_dataset(split="train")
        self._train_size = 448_000_000

        self._val_dataset = self._get_dataset(split="test")
        self._val_size = 40_000  


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader | None:
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=False,
        )
