import torch
import torch.nn as nn
from mvid_omega.models.aggregator import Aggregator
from mvid_omega.models.heads import CameraHead
from mvid_omega.models.heads.mvid_heads import MVIDDenseHead, SHReadoutHead

class MVIDModel(nn.Module):

    def __init__(self, patch_size: int=16, embed_dim: int=1024, enable_camera: bool=True, enable_sh: bool=True) -> None:
        super().__init__()
        self.aggregator = Aggregator(patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.dense_head = MVIDDenseHead(dim_in=2 * embed_dim, patch_size=patch_size)
        self.sh_head = SHReadoutHead(dim_in=2 * embed_dim) if enable_sh else None

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            aggregated_tokens_list, patch_token_start = self.aggregator(images)
        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError('Aggregator did not cache the final layer, which MVIDModel needs.')
        predictions: dict[str, torch.Tensor] = {'camera_and_register_tokens': final_tokens[:, :, :patch_token_start].contiguous()}
        with torch.autocast(device_type='cuda', enabled=False):
            if self.camera_head is not None:
                predictions['pose_enc'] = self.camera_head(aggregated_tokens_list, patch_token_start=patch_token_start)
            predictions.update(self.dense_head(aggregated_tokens_list, images=images, patch_token_start=patch_token_start))
            if self.sh_head is not None:
                predictions.update(self.sh_head(aggregated_tokens_list, patch_token_start=patch_token_start))
        if not self.training:
            predictions['images'] = images
        return predictions
