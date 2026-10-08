# Copyright (c) OpenMMLab. All rights reserved.

from typing import List

from mmdet.registry import DATASETS
from .base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class OCRDinoDataset(BaseDetDataset):
    """Generic dataset for OCR-DINO unified annotations.

    Expected annotation format:

    {
        "metainfo": {
            "format_version": "ocrdino_v1",
            "classes": [...]
        },
        "data_list": [
            {
                "img_path": "...",
                "height": ...,
                "width": ...,
                "instances": [
                    {
                        "bbox": [x1, y1, x2, y2],
                        "bbox_label": ...,
                        "ignore_flag": 0,
                        "text": "...",
                        "text_type": "text",
                        "poly": [...],
                        "order": ...
                    }
                ]
            }
        ]
    }
    """

    def load_data_list(self) -> List[dict]:
        data_list = super().load_data_list()

        format_version = self.metainfo.get(
            'format_version',
            None,
        )

        if format_version != 'ocrdino_v1':
            raise ValueError(
                'Unsupported OCR-DINO annotation format: '
                f'{format_version}. Expected "ocrdino_v1".'
            )

        return data_list