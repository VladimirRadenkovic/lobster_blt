import importlib.resources

from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from ._make_pretrained_tokenizer_fast import make_pretrained_tokenizer_fast

AA_VOCAB = {
    "<bos>": 0,
    "<pad>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "<mask>": 4,
    "|": 5,
    "L": 6,
    "A": 7,
    "G": 8,
    "V": 9,
    "S": 10,
    "E": 11,
    "R": 12,
    "T": 13,
    "I": 14,
    "D": 15,
    "P": 16,
    "K": 17,
    "Q": 18,
    "N": 19,
    "F": 20,
    "Y": 21,
    "M": 22,
    "H": 23,
    "W": 24,
    "C": 25,
    "B": 26,
}

PRETRAINED_TOKENIZER_PATH = importlib.resources.files("lobster") / "assets" / "amino_acid_tokenizer"


def _make_amino_acid_tokenizer() -> PreTrainedTokenizerFast:
    """Create a `PreTrainedTokenizerFast` object for tokenization of protein sequences.

    To create the tokenizer config stored under lobster/assets/amino_acid_tokenizer we run

    ```
    tokenizer = _make_amino_acid_tokenizer()
    tokenizer.save_pretrained("src/lobster/assets/amino_acid_tokenizer")
    ```

    This can now be loaded using
    `PreTrainedTokenizerFast.from_pretrained("src/lobster/assets/amino_acid_tokenizer")`
    """

    # BPE with no merges => just use input vocab
    tokenizer_model = BPE(AA_VOCAB, merges=[], unk_token="<unk>", ignore_merges=True)

    # bert style post processing
    post_processor = TemplateProcessing(
        single="<cls> $A <eos>",
        pair="<cls> $A <eos> $B:1 <eos>:1",
        special_tokens=[("<cls>", 0), ("<eos>", 2)],  # NOTE must match ids from AA_VOCAB
    )

    return make_pretrained_tokenizer_fast(
        tokenizer_model=tokenizer_model,
        post_processor=post_processor,
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<cls>",
        mask_token="<mask>",
    )


class AminoAcidTokenizerFast(PreTrainedTokenizerFast):
    padding_side = "right"
    truncation_side = "right"
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self):
        super().__init__(
            tokenizer_file=str(PRETRAINED_TOKENIZER_PATH / "tokenizer_amplify.json"),
            bos_token="<bos>",
            eos_token="<eos>",
            unk_token="<unk>",
            sep_token=None,
            pad_token="<pad>",
            cls_token=None,
            mask_token="<mask>",
        )
