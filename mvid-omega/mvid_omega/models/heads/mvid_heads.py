import torch
import torch.nn as nn
import torch.nn.functional as F
from mvid_omega.models.layers import SelfAttentionBlock
from .dense_head import DenseHead, _init_small_conf_prediction_head, _make_prediction_head

class MVIDDenseHead(DenseHead):

    def __init__(self, dim_in: int=2048, patch_size: int=16, features: int=256, out_channels: list[int]=[256, 512, 1024, 1024], intermediate_layer_idx: list[int]=[4, 11, 17, 23], albedo_min: float=0.03, albedo_max: float=0.95) -> None:
        super().__init__(dim_in=dim_in, patch_size=patch_size, features=features, out_channels=out_channels, intermediate_layer_idx=intermediate_layer_idx)
        self.albedo_min = albedo_min
        self.albedo_range = albedo_max - albedo_min
        subpix = self.final_shuffle_factor ** 2
        self.up_features = features // 2
        self.pre_up = nn.Conv2d(features, self.up_features, 3, padding=1)

        def full_scale_head(out_ch: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(self.up_features, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, out_ch, 1))
        self.proj_albedo = full_scale_head(3)
        self.proj_normal = _make_prediction_head(features, 3 * subpix)
        self.proj_albedo_conf = _make_prediction_head(features, subpix)
        self.proj_normal_conf = _make_prediction_head(features, subpix)
        _init_small_conf_prediction_head(self.proj_albedo_conf)
        _init_small_conf_prediction_head(self.proj_normal_conf)
        self.proj_vis = _make_prediction_head(features, subpix)
        nn.init.zeros_(self.proj_vis.weight)
        nn.init.constant_(self.proj_vis.bias, 2.0)
        self.proj_shading = full_scale_head(3)
        nn.init.zeros_(self.proj_shading[-1].weight)
        nn.init.zeros_(self.proj_shading[-1].bias)
        self.proj_residual = full_scale_head(3)
        nn.init.zeros_(self.proj_residual[-1].weight)
        nn.init.constant_(self.proj_residual[-1].bias, -4.0)

    def forward(self, aggregated_tokens_list: list[torch.Tensor | None], images: torch.Tensor, patch_token_start: int, frames_chunk_size: int | None=8) -> dict[str, torch.Tensor]:
        if patch_token_start is None:
            raise ValueError('patch_token_start is required for MVIDDenseHead')
        _, num_frames, _, _, _ = images.shape
        if frames_chunk_size is None or frames_chunk_size >= num_frames:
            return self._forward_impl_mvid(aggregated_tokens_list, images, patch_token_start)
        assert frames_chunk_size > 0
        chunks: list[dict[str, torch.Tensor]] = []
        for start in range(0, num_frames, frames_chunk_size):
            end = min(start + frames_chunk_size, num_frames)
            chunks.append(self._forward_impl_mvid(aggregated_tokens_list, images, patch_token_start, start, end))
        return {k: torch.cat([c[k] for c in chunks], dim=1) for k in chunks[0]}

    def _forward_impl_mvid(self, aggregated_tokens_list: list[torch.Tensor | None], images: torch.Tensor, patch_token_start: int, frames_start_idx: int | None=None, frames_end_idx: int | None=None) -> dict[str, torch.Tensor]:
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()
        batch_size, num_frames, _, height, width = images.shape
        patch_h, patch_w = (height // self.patch_size, width // self.patch_size)
        multi_scale_features = []
        for feature_idx, layer_idx in enumerate(self.intermediate_layer_idx):
            x = aggregated_tokens_list[layer_idx]
            if x is None:
                raise ValueError(f'Aggregator did not cache layer {layer_idx}, which MVIDDenseHead needs.')
            x = x[:, :, patch_token_start:]
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]
            if x.dtype != torch.float32:
                x = x.float()
            x = x.reshape(batch_size * num_frames, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[feature_idx](x)
            x = self._apply_pos_embed(x, width, height)
            x = self.resize_layers[feature_idx](x)
            multi_scale_features.append(x)
        fused = self.scratch_forward(multi_scale_features)
        fused = self._apply_pos_embed(fused, width, height)

        def shuffle(logits: torch.Tensor) -> torch.Tensor:
            out = F.pixel_shuffle(logits, self.final_shuffle_factor)
            return out.permute(0, 2, 3, 1)
        up = F.interpolate(self.pre_up(fused), size=(height, width), mode='bilinear', align_corners=True)

        def at_full(logits: torch.Tensor) -> torch.Tensor:
            return logits.permute(0, 2, 3, 1)
        depth = torch.exp(shuffle(self.proj(fused)))
        depth_conf = 1.0 + torch.exp(shuffle(self.proj_conf(fused)).squeeze(-1))
        albedo = self.albedo_min + self.albedo_range * torch.sigmoid(at_full(self.proj_albedo(up)))
        albedo_conf = 1.0 + torch.exp(shuffle(self.proj_albedo_conf(fused)).squeeze(-1))
        normal = F.normalize(shuffle(self.proj_normal(fused)), dim=-1, eps=1e-06)
        normal_conf = 1.0 + torch.exp(shuffle(self.proj_normal_conf(fused)).squeeze(-1))
        visibility = torch.sigmoid(shuffle(self.proj_vis(fused)).squeeze(-1))
        shading = torch.exp(at_full(self.proj_shading(up)))
        residual = torch.exp(at_full(self.proj_residual(up)))

        def per_frame(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch_size, num_frames, *t.shape[1:])
        out = {'depth': per_frame(depth), 'depth_conf': per_frame(depth_conf), 'albedo': per_frame(albedo), 'albedo_conf': per_frame(albedo_conf), 'normal': per_frame(normal), 'normal_conf': per_frame(normal_conf), 'visibility': per_frame(visibility), 'shading': per_frame(shading), 'residual': per_frame(residual)}
        for k, v in out.items():
            if v.dtype != torch.float32:
                raise TypeError(f'MVIDDenseHead output {k} must be fp32, got {v.dtype}')
        return out

class SHReadoutHead(nn.Module):

    def __init__(self, dim_in: int=2048, num_sh: int=9, num_channels: int=3) -> None:
        super().__init__()
        self.num_sh = num_sh
        self.num_channels = num_channels
        self.token_norm = nn.LayerNorm(dim_in, eps=1e-05)
        self.lighting_token = nn.Parameter(torch.zeros(1, 1, dim_in))
        nn.init.trunc_normal_(self.lighting_token, std=0.02)
        self.readout_blocks = nn.ModuleList([SelfAttentionBlock(dim=dim_in, num_heads=16, ffn_ratio=4.0, qkv_bias=True, proj_bias=True, ffn_bias=True, init_values=1e-05, use_qk_norm=False, mask_k_bias=False) for _ in range(4)])
        self.lighting_token_norm = nn.LayerNorm(dim_in, eps=1e-05)
        self.embedding_projector = nn.Sequential(nn.Linear(dim_in, dim_in // 2, bias=True), nn.GELU(), nn.LayerNorm(dim_in // 2, eps=1e-05), nn.Linear(dim_in // 2, dim_in, bias=True))
        self.sh_decoder = nn.Linear(dim_in, num_sh * num_channels, bias=True)
        nn.init.zeros_(self.sh_decoder.weight)
        with torch.no_grad():
            bias = self.sh_decoder.bias.view(num_sh, num_channels)
            bias.zero_()
            bias[0].fill_(0.5)

    def forward(self, aggregated_tokens_list: list[torch.Tensor | None], patch_token_start: int) -> dict[str, torch.Tensor]:
        tokens = aggregated_tokens_list[-1]
        if tokens is None:
            raise ValueError('Aggregator did not cache the final layer, which SHReadoutHead needs.')
        if patch_token_start is None:
            raise ValueError('patch_token_start is required for SHReadoutHead')
        if tokens.dtype != torch.float32:
            tokens = tokens.float()
        batch_size, num_frames, _, _ = tokens.shape
        ctx = tokens[:, :, :patch_token_start]
        ctx = self.token_norm(ctx)
        ctx = ctx.reshape(batch_size, num_frames * patch_token_start, -1)
        lighting_token = self.lighting_token.expand(batch_size, -1, -1)
        readout = torch.cat([lighting_token, ctx], dim=1)
        for block in self.readout_blocks:
            readout = block(readout, None)
        lighting_embedding = self.embedding_projector(self.lighting_token_norm(readout[:, 0]))
        raw = self.sh_decoder(lighting_embedding)
        raw = raw.view(batch_size, self.num_sh, self.num_channels)
        sh = torch.cat([F.softplus(raw[:, :1]), raw[:, 1:]], dim=1)
        return {'sh_coeffs': sh, 'lighting_embedding': lighting_embedding}
