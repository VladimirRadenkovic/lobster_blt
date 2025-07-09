
from pathlib import Path
from typing import Callable, ClassVar, Literal

from lobster.datasets._huggingface_iterable_dataset import HuggingFaceIterableDataset
from lobster.transforms import Transform

from lobster.datasets import AMPLIFYIterableDataset


dataset = AMPLIFYIterableDataset(
        root="/rds/user/vr375/hpc-work/lobster/data",
        download=True  # Download entire dataset first
    )

for i, sample in enumerate(dataset):
        if i >= 3:
            break
        print(f"Sample {i+1}: {sample[:50]}...")


