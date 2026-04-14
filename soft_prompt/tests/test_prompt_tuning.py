import torch
from transformers import T5Config, T5ForConditionalGeneration

from soft_prompt_repro.prompt_tuning import SoftPromptT5


def _build_tiny_t5():
    config = T5Config(
        vocab_size=128,
        d_model=32,
        d_ff=64,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    return T5ForConditionalGeneration(config)


def test_soft_prompt_prepends_virtual_tokens():
    prompt_model = SoftPromptT5(model=_build_tiny_t5(), num_virtual_tokens=5)
    input_ids = torch.tensor([[3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 1]])

    embeds, prompted_mask = prompt_model.build_prompted_inputs(input_ids, attention_mask)

    assert embeds.shape == (1, 8, 32)
    assert prompted_mask.tolist() == [[1, 1, 1, 1, 1, 1, 1, 1]]


def test_only_prompt_is_trainable():
    prompt_model = SoftPromptT5(model=_build_tiny_t5(), num_virtual_tokens=4)
    trainable_names = [name for name, param in prompt_model.named_parameters() if param.requires_grad]

    assert trainable_names == ["prompt_embeddings"]