# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, Optional

from torch import Tensor

from mmdet.registry import MODELS
from .dino import DINO


@MODELS.register_module()
class OCRDINO(DINO):
    """DINO extended for OCR."""

    def forward_decoder(
        self,
        query: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        dn_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict:
        """Forward DINO decoder and preserve encoder information for OCR."""

        # Run the original official DINO decoder unchanged.
        decoder_outputs_dict = super().forward_decoder(
            query=query,
            memory=memory,
            memory_mask=memory_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            dn_mask=dn_mask,
            **kwargs,
        )

        # Preserve the visual memory required by the future OCR decoder.
        decoder_outputs_dict.update(
            ocr_memory=memory,
            ocr_memory_mask=memory_mask,
            ocr_spatial_shapes=spatial_shapes,
            ocr_level_start_index=level_start_index,
            ocr_valid_ratios=valid_ratios,
        )

        return decoder_outputs_dict