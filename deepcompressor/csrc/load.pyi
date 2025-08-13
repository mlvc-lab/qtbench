import torch

class _C:
    """Deepcompressor C++ extension."""

    @staticmethod
    def round_to_nearest_in_codebook_cuda(
        tensor: torch.Tensor, codebook: torch.Tensor, inplace: bool = False, bnb: bool = False
    ) -> torch.Tensor:
        """Round tensor to nearest codebook value.

        Args:
            tensor (`torch.Tensor`):
                Tensor to be rounded.
            codebook (`torch.Tensor`):
                Codebook tensor.
            inplace (`bool`, *optional*, defaults to `False`):
                Whether to update the tensor in place.
            bnb (`bool`, *optional*, defaults to `False`):
                Whether to use bitsandbytes rounding.

        Returns:
            `torch.Tensor`:
                Rounded tensor.
        """
        pass
