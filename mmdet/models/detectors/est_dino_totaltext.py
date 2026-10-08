from typing import List

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy

from .est_dino import ESTDINO


@MODELS.register_module()
class ESTDINOTotalText(ESTDINO):

    @staticmethod
    def _decode_rec_tokens(tokens: Tensor) -> List[str]:
        texts = []

        for row in tokens.detach().cpu().tolist():
            chars = []

            for token in row:
                token = int(token)

                if token == 96:
                    break

                if 0 <= token <= 94:
                    chars.append(chr(token + 32))
                elif token == 95:
                    chars.append('\ufffd')
                else:
                    raise RuntimeError(
                        f'Invalid recognition token: {token}'
                    )

            texts.append(
                ''.join(chars)
            )

        return texts

    def predict(
        self,
        batch_inputs: Tensor,
        batch_data_samples: SampleList,
        rescale: bool = True,
    ) -> SampleList:

        img_feats = self.extract_feat(
            batch_inputs
        )

        (
            head_inputs_dict,
            text_outputs_dict,
        ) = self._forward_transformer_impl(
            img_feats=img_feats,
            batch_data_samples=batch_data_samples,
        )

        hidden_states = (
            head_inputs_dict[
                'hidden_states'
            ]
        )

        references = (
            head_inputs_dict[
                'references'
            ]
        )

        recognition_logits = (
            text_outputs_dict[
                'recognition_logits'
            ]
        )

        (
            all_cls_scores,
            all_bbox_preds,
        ) = self.bbox_head(
            hidden_states,
            references,
        )

        cls_scores = (
            all_cls_scores[-1]
        )

        bbox_preds = (
            all_bbox_preds[-1]
        )

        rec_logits = (
            recognition_logits[-1]
        )

        if cls_scores.ndim != 3:
            raise RuntimeError(
                'Unexpected classification '
                f'shape: {tuple(cls_scores.shape)}'
            )

        if bbox_preds.ndim != 3:
            raise RuntimeError(
                'Unexpected bbox '
                f'shape: {tuple(bbox_preds.shape)}'
            )

        if rec_logits.ndim != 4:
            raise RuntimeError(
                'Unexpected recognition '
                f'shape: {tuple(rec_logits.shape)}'
            )

        batch_size = (
            cls_scores.shape[0]
        )

        if (
            bbox_preds.shape[0]
            != batch_size
            or rec_logits.shape[0]
            != batch_size
        ):
            raise RuntimeError(
                'Prediction batch dimensions '
                'are not aligned.'
            )

        if (
            bbox_preds.shape[1]
            != cls_scores.shape[1]
            or rec_logits.shape[1]
            != cls_scores.shape[1]
        ):
            raise RuntimeError(
                'Detection and recognition '
                'query dimensions are not aligned.'
            )

        result_list = []

        num_classes = (
            cls_scores.shape[-1]
        )

        max_per_img = int(
            self.test_cfg.get(
                'max_per_img',
                self.num_queries,
            )
        )

        for image_index in range(
            batch_size
        ):
            cls_score = (
                cls_scores[
                    image_index
                ]
            )

            bbox_pred = (
                bbox_preds[
                    image_index
                ]
            )

            image_rec_logits = (
                rec_logits[
                    image_index
                ]
            )

            if self.bbox_head.loss_cls.use_sigmoid:
                scores = (
                    cls_score.sigmoid()
                )

                flattened_scores = (
                    scores.reshape(-1)
                )

                num_topk = min(
                    max_per_img,
                    flattened_scores.numel(),
                )

                (
                    det_scores,
                    topk_indices,
                ) = flattened_scores.topk(
                    num_topk
                )

                det_labels = (
                    topk_indices
                    % num_classes
                )

                query_indices = (
                    topk_indices
                    // num_classes
                )

            else:
                scores = (
                    cls_score.softmax(-1)[
                        ...,
                        :-1
                    ]
                )

                flattened_scores = (
                    scores.reshape(-1)
                )

                num_topk = min(
                    max_per_img,
                    flattened_scores.numel(),
                )

                (
                    det_scores,
                    topk_indices,
                ) = flattened_scores.topk(
                    num_topk
                )

                det_labels = (
                    topk_indices
                    % num_classes
                )

                query_indices = (
                    topk_indices
                    // num_classes
                )

            selected_bbox_pred = (
                bbox_pred[
                    query_indices
                ]
            )

            selected_rec_logits = (
                image_rec_logits[
                    query_indices
                ]
            )

            det_bboxes = (
                bbox_cxcywh_to_xyxy(
                    selected_bbox_pred
                )
            )

            img_meta = (
                batch_data_samples[
                    image_index
                ].metainfo
            )

            img_shape = (
                img_meta[
                    'img_shape'
                ]
            )

            img_h = float(
                img_shape[0]
            )

            img_w = float(
                img_shape[1]
            )

            det_bboxes[
                :,
                0::2
            ] *= img_w

            det_bboxes[
                :,
                1::2
            ] *= img_h

            det_bboxes[
                :,
                0::2
            ].clamp_(
                min=0.0,
                max=img_w,
            )

            det_bboxes[
                :,
                1::2
            ].clamp_(
                min=0.0,
                max=img_h,
            )

            if rescale:
                scale_factor = (
                    img_meta.get(
                        'scale_factor',
                        None,
                    )
                )

                if scale_factor is not None:
                    scale_factor = (
                        det_bboxes.new_tensor(
                            scale_factor
                        )
                    )

                    if (
                        scale_factor.numel()
                        == 2
                    ):
                        scale_factor = (
                            scale_factor.repeat(2)
                        )

                    det_bboxes = (
                        det_bboxes
                        / scale_factor
                    )

            rec_tokens = (
                selected_rec_logits.argmax(
                    dim=-1
                )
            )

            rec_texts = (
                self._decode_rec_tokens(
                    rec_tokens
                )
            )


            from mmengine.structures import InstanceData

            pred_instances = InstanceData()

            pred_instances.bboxes = (
                det_bboxes
            )

            pred_instances.scores = (
                det_scores
            )

            pred_instances.labels = (
                det_labels
            )

            pred_instances.rec_tokens = (
                rec_tokens
            )

            pred_instances.rec_texts = (
                rec_texts
            )

            result_list.append(
                pred_instances
            )

        return self.add_pred_to_datasample(
            batch_data_samples,
            result_list,
        )