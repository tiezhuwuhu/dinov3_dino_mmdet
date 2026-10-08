# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmdet.utils import InstanceList

from ..layers.transformer.ocr_dino_layers import (
    BlockTextDecoder,
    BlockVisualExtractor,
)
from .dino_head import DINOHead
from mmdet.utils.ocr_tokenizer import OCRDinoCharTokenizer

@MODELS.register_module()
class OCRDINOHead(DINOHead):
    """DINO head extended with OCR recognition."""

    def __init__(
        self,
        *args,
        ocr_vocab_size: int = 5000,
        ocr_num_visual_queries: int = 64,
        ocr_max_seq_len: int = 1024,
        ocr_visual_num_layers: int = 2,
        ocr_text_num_layers: int = 2,
        ocr_num_heads: int = 8,
        ocr_ffn_channels: int = 1024,
        ocr_dropout: float = 0.1,
        ocr_loss_weight: float = 1.0,
        ocr_vocab_file: Optional[str] = None,
        ocr_class_ids: Optional[List[int]] = None,  
        ocr_score_thr: float = 0.3,
        ocr_decode_max_len: Optional[int] = None,
        ocr_chunk_size: int = 16,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.ocr_vocab_size = ocr_vocab_size
        self.ocr_num_visual_queries = ocr_num_visual_queries
        self.ocr_max_seq_len = ocr_max_seq_len
        self.ocr_loss_weight = ocr_loss_weight

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.ocr_score_thr = float(ocr_score_thr)
        
        self.ocr_class_ids = (
            None
            if ocr_class_ids is None
            else tuple(int(x) for x in ocr_class_ids)
        )
        
        self.ocr_decode_max_len = (
            ocr_max_seq_len
            if ocr_decode_max_len is None
            else min(
                int(ocr_decode_max_len),
                ocr_max_seq_len,
            )
        )
        
        self.ocr_chunk_size = max(
            1,
            int(ocr_chunk_size),
        )
        
        self.ocr_tokenizer = None
        
        if ocr_vocab_file is not None:
            self.ocr_tokenizer = OCRDinoCharTokenizer(
                ocr_vocab_file
            )
        
            if (
                self.ocr_tokenizer.vocab_size
                != self.ocr_vocab_size
            ):
                raise ValueError(
                    'Tokenizer vocabulary size '
                    f'{self.ocr_tokenizer.vocab_size} '
                    'does not match '
                    f'ocr_vocab_size={self.ocr_vocab_size}.'
                )



        self.block_visual_extractor = BlockVisualExtractor(
            embed_dims=self.embed_dims,
            num_visual_queries=ocr_num_visual_queries,
            num_layers=ocr_visual_num_layers,
            num_heads=ocr_num_heads,
            num_feature_levels=4,
            num_points=4,
            ffn_channels=ocr_ffn_channels,
            dropout=0.0,
        )

        self.block_text_decoder = BlockTextDecoder(
            vocab_size=ocr_vocab_size,
            embed_dims=self.embed_dims,
            max_seq_len=ocr_max_seq_len,
            num_layers=ocr_text_num_layers,
            num_heads=ocr_num_heads,
            ffn_channels=ocr_ffn_channels,
            dropout=ocr_dropout,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )

    def forward(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        **kwargs,
    ):
        return super().forward(
            hidden_states,
            references,
        )

    def loss(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        enc_outputs_class: Tensor,
        enc_outputs_coord: Tensor,
        batch_data_samples: SampleList,
        dn_meta: Dict[str, int],
        ocr_memory: Optional[Tensor] = None,
        ocr_memory_mask: Optional[Tensor] = None,
        ocr_spatial_shapes: Optional[Tensor] = None,
        ocr_level_start_index: Optional[Tensor] = None,
        ocr_valid_ratios: Optional[Tensor] = None,
        **kwargs,
    ) -> dict:
        """Calculate detection and OCR losses."""

        losses = super().loss(
            hidden_states=hidden_states,
            references=references,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord=enc_outputs_coord,
            batch_data_samples=batch_data_samples,
            dn_meta=dn_meta,
        )

        required_ocr_inputs = (
            ocr_memory,
            ocr_spatial_shapes,
            ocr_level_start_index,
            ocr_valid_ratios,
        )

        if any(
            value is None
            for value in required_ocr_inputs
        ):
            losses['loss_ocr'] = (
                hidden_states.sum() * 0.0
            )
            return losses

        loss_ocr = self.loss_ocr(
            hidden_states=hidden_states,
            references=references,
            batch_data_samples=batch_data_samples,
            dn_meta=dn_meta,
            ocr_memory=ocr_memory,
            ocr_memory_mask=ocr_memory_mask,
            ocr_spatial_shapes=ocr_spatial_shapes,
            ocr_level_start_index=ocr_level_start_index,
            ocr_valid_ratios=ocr_valid_ratios,
        )
        
        

        losses['loss_ocr'] = loss_ocr

        return losses


    def predict(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        batch_data_samples: SampleList,
        rescale: bool = True,
        ocr_memory: Optional[Tensor] = None,
        ocr_memory_mask: Optional[Tensor] = None,
        ocr_spatial_shapes: Optional[Tensor] = None,
        ocr_level_start_index: Optional[Tensor] = None,
        ocr_valid_ratios: Optional[Tensor] = None,
        **kwargs,
    ) -> InstanceList:
        """Predict detection and OCR results."""
    
        (
            matching_hidden_states,
            matching_cls_scores,
            matching_bbox_preds,
        ) = self._get_matching_outputs(
            hidden_states=hidden_states,
            references=references,
            dn_meta=None,
        )
    
        result_list = []
    
        batch_size = matching_hidden_states.shape[0]
    
        for image_index in range(batch_size):
            img_meta = batch_data_samples[
                image_index
            ].metainfo
    
            cls_score = matching_cls_scores[
                image_index
            ]
    
            bbox_pred = matching_bbox_preds[
                image_index
            ]
    
            if self.loss_cls.use_sigmoid:
                probabilities = cls_score.sigmoid()
    
                query_scores, query_labels = (
                    probabilities.max(dim=-1)
                )
            else:
                probabilities = F.softmax(
                    cls_score,
                    dim=-1,
                )[..., :self.num_classes]
    
                query_scores, query_labels = (
                    probabilities.max(dim=-1)
                )
    
            max_per_img = self.test_cfg.get(
                'max_per_img',
                len(query_scores),
            )
    
            max_per_img = min(
                max_per_img,
                len(query_scores),
            )
    
            scores, query_indices = torch.topk(
                query_scores,
                k=max_per_img,
            )
    
            labels = query_labels[
                query_indices
            ]
    
            selected_bbox_preds = bbox_pred[
                query_indices
            ]
    
            score_thr = self.test_cfg.get(
                'score_thr',
                0.0,
            )
    
            keep = scores >= score_thr
    
            scores = scores[keep]
            labels = labels[keep]
            query_indices = query_indices[keep]
            selected_bbox_preds = (
                selected_bbox_preds[keep]
            )
    
            img_h, img_w = img_meta[
                'img_shape'
            ][:2]
    
            det_bboxes = bbox_cxcywh_to_xyxy(
                selected_bbox_preds
            )
    
            det_bboxes[:, 0::2] *= img_w
            det_bboxes[:, 1::2] *= img_h
    
            det_bboxes[:, 0::2].clamp_(
                min=0,
                max=img_w,
            )
    
            det_bboxes[:, 1::2].clamp_(
                min=0,
                max=img_h,
            )
    
            if rescale and len(det_bboxes) > 0:
                scale_factor = img_meta.get(
                    'scale_factor'
                )
    
                if scale_factor is not None:
                    det_bboxes /= (
                        det_bboxes.new_tensor(
                            scale_factor
                        ).repeat((1, 2))
                    )
    
            results = InstanceData()
    
            results.bboxes = det_bboxes
            results.scores = scores
            results.labels = labels
            results.query_indices = query_indices
    
            num_results = len(scores)
    
            texts = [
                ''
                for _ in range(num_results)
            ]
    
            token_results = [
                []
                for _ in range(num_results)
            ]
    
            if (
                num_results > 0
                and ocr_memory is not None
                and ocr_spatial_shapes is not None
                and ocr_level_start_index is not None
                and ocr_valid_ratios is not None
            ):
                ocr_mask = (
                    scores >= self.ocr_score_thr
                )
    
                if self.ocr_class_ids is not None:
                    class_mask = torch.zeros_like(
                        ocr_mask,
                        dtype=torch.bool,
                    )
    
                    for class_id in self.ocr_class_ids:
                        class_mask |= labels.eq(
                            class_id
                        )
    
                    ocr_mask &= class_mask
    
                ocr_positions = torch.nonzero(
                    ocr_mask,
                    as_tuple=False,
                ).squeeze(-1)
    
                for start in range(
                    0,
                    ocr_positions.numel(),
                    self.ocr_chunk_size,
                ):
                    positions = ocr_positions[
                        start:
                        start + self.ocr_chunk_size
                    ]
    
                    chunk_query_indices = (
                        query_indices[
                            positions
                        ]
                    )
    
                    selected_queries = (
                        matching_hidden_states[
                            image_index:
                            image_index + 1,
                            chunk_query_indices,
                            :,
                        ]
                    )
    
                    selected_boxes = (
                        matching_bbox_preds[
                            image_index:
                            image_index + 1,
                            chunk_query_indices,
                            :,
                        ]
                    )
    
                    image_memory = ocr_memory[
                        image_index:
                        image_index + 1
                    ]
    
                    if ocr_memory_mask is None:
                        image_memory_mask = None
                    else:
                        image_memory_mask = (
                            ocr_memory_mask[
                                image_index:
                                image_index + 1
                            ]
                        )
    
                    image_valid_ratios = (
                        ocr_valid_ratios[
                            image_index:
                            image_index + 1
                        ]
                    )
    
                    block_visual_features = (
                        self.block_visual_extractor(
                            object_queries=(
                                selected_queries
                            ),
                            object_boxes=(
                                selected_boxes
                            ),
                            memory=image_memory,
                            memory_mask=(
                                image_memory_mask
                            ),
                            spatial_shapes=(
                                ocr_spatial_shapes
                            ),
                            level_start_index=(
                                ocr_level_start_index
                            ),
                            valid_ratios=(
                                image_valid_ratios
                            ),
                        )
                    )
    
                    generated = (
                        self._greedy_decode_ocr(
                            block_visual_features=(
                                block_visual_features
                            ),
                            block_queries=(
                                selected_queries
                            ),
                        )
                    )
    
                    for local_index, result_index in enumerate(
                        positions.tolist()
                    ):
                        token_ids = generated[
                            local_index
                        ]
    
                        token_results[
                            result_index
                        ] = token_ids
    
                        if self.ocr_tokenizer is not None:
                            texts[
                                result_index
                            ] = (
                                self.ocr_tokenizer.decode(
                                    token_ids
                                )
                            )
    
            results.texts = texts
            results.ocr_token_ids = token_results
    
            result_list.append(results)
    
        return result_list

    def _greedy_decode_ocr(
        self,
        block_visual_features: Tensor,
        block_queries: Tensor,
    ) -> List[List[int]]:
        """Autoregressively decode OCR tokens."""
    
        num_blocks = block_visual_features.shape[1]
    
        if num_blocks == 0:
            return []
    
        device = block_visual_features.device
    
        generated = torch.full(
            (
                1,
                num_blocks,
                1,
            ),
            fill_value=self.bos_token_id,
            dtype=torch.long,
            device=device,
        )
    
        finished = torch.zeros(
            (
                1,
                num_blocks,
            ),
            dtype=torch.bool,
            device=device,
        )
    
        for _ in range(
            self.ocr_decode_max_len - 1
        ):
            logits = self.block_text_decoder(
                block_visual_features=(
                    block_visual_features
                ),
                input_ids=generated,
                block_queries=block_queries,
            )
    
            next_token = logits[
                :,
                :,
                -1,
                :,
            ].argmax(dim=-1)
    
            next_token = torch.where(
                finished,
                torch.full_like(
                    next_token,
                    self.pad_token_id,
                ),
                next_token,
            )
    
            generated = torch.cat(
                [
                    generated,
                    next_token.unsqueeze(-1),
                ],
                dim=-1,
            )
    
            finished = (
                finished
                | next_token.eq(
                    self.eos_token_id
                )
            )
    
            if bool(finished.all()):
                break
    
        return [
            generated[
                0,
                index,
            ].tolist()
            for index in range(num_blocks)
        ]


    def _gather_queries(
        self,
        tensor: Tensor,
        query_indices: Tensor,
    ) -> Tensor:
        """Gather selected queries from a batched tensor."""

        if tensor.dim() != 3:
            raise ValueError(
                'tensor must have shape [B, Q, C].'
            )

        if query_indices.dim() != 2:
            raise ValueError(
                'query_indices must have shape [B, K].'
            )

        batch_size, _, channels = tensor.shape

        if query_indices.shape[0] != batch_size:
            raise ValueError(
                'Batch size mismatch in query_indices.'
            )

        gather_index = query_indices.unsqueeze(-1).expand(
            -1,
            -1,
            channels,
        )

        return torch.gather(
            tensor,
            dim=1,
            index=gather_index,
        )

    def _get_matching_outputs(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        dn_meta: Optional[Dict[str, int]],
    ):
        """Get final outputs belonging only to matching queries."""

        all_cls_scores, all_bbox_preds = super().forward(
            hidden_states,
            references,
        )

        num_dn_queries = 0

        if dn_meta is not None:
            num_dn_queries = int(
                dn_meta.get(
                    'num_denoising_queries',
                    0,
                )
            )

        final_hidden_states = hidden_states[
            -1,
            :,
            num_dn_queries:,
            :,
        ]

        final_cls_scores = all_cls_scores[
            -1,
            :,
            num_dn_queries:,
            :,
        ]

        final_bbox_preds = all_bbox_preds[
            -1,
            :,
            num_dn_queries:,
            :,
        ]

        return (
            final_hidden_states,
            final_cls_scores,
            final_bbox_preds,
        )

    def _assign_single_image(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        gt_instances: InstanceData,
        img_meta: dict,
    ):
        """Run the same Hungarian assignment used by DETR/DINO."""

        img_h, img_w = img_meta['img_shape'][:2]

        factor = bbox_pred.new_tensor(
            [
                img_w,
                img_h,
                img_w,
                img_h,
            ]
        ).unsqueeze(0)

        pred_bboxes = bbox_cxcywh_to_xyxy(
            bbox_pred
        )

        pred_bboxes = pred_bboxes * factor

        pred_instances = InstanceData(
            scores=cls_score,
            bboxes=pred_bboxes,
        )

        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta,
        )

        pos_query_inds = torch.nonzero(
            assign_result.gt_inds > 0,
            as_tuple=False,
        ).squeeze(-1)

        assigned_gt_inds = (
            assign_result.gt_inds[
                pos_query_inds
            ] - 1
        )

        return (
            pos_query_inds,
            assigned_gt_inds.long(),
        )

    def _filter_ocr_matches(
        self,
        query_inds: Tensor,
        gt_inds: Tensor,
        gt_instances: InstanceData,
    ):
        """Keep only GT instances that have valid OCR supervision."""

        valid_query_inds = []
        valid_gt_inds = []

        ocr_token_ids = gt_instances.ocr_token_ids
        ocr_trainable = gt_instances.ocr_trainable

        for query_idx, gt_idx in zip(
            query_inds.tolist(),
            gt_inds.tolist(),
        ):
            trainable = bool(
                ocr_trainable[gt_idx]
            )

            if not trainable:
                continue

            token_ids = ocr_token_ids[gt_idx]

            if not token_ids:
                continue

            input_length = len(token_ids) - 1

            if input_length <= 0:
                continue

            if input_length > self.ocr_max_seq_len:
                continue

            valid_query_inds.append(
                query_idx
            )

            valid_gt_inds.append(
                gt_idx
            )

        if not valid_query_inds:
            device = query_inds.device

            return (
                torch.empty(
                    0,
                    dtype=torch.long,
                    device=device,
                ),
                torch.empty(
                    0,
                    dtype=torch.long,
                    device=device,
                ),
            )

        return (
            query_inds.new_tensor(
                valid_query_inds,
                dtype=torch.long,
            ),
            gt_inds.new_tensor(
                valid_gt_inds,
                dtype=torch.long,
            ),
        )

    def _build_teacher_forcing_batch(
        self,
        gt_instances: InstanceData,
        gt_inds: Tensor,
        device,
    ):
        """Build dynamically padded OCR input and target tensors."""

        sequences = [
            gt_instances.ocr_token_ids[
                int(gt_idx)
            ]
            for gt_idx in gt_inds.tolist()
        ]

        max_input_len = max(
            len(sequence) - 1
            for sequence in sequences
        )

        num_sequences = len(sequences)

        input_ids = torch.full(
            (
                1,
                num_sequences,
                max_input_len,
            ),
            fill_value=self.pad_token_id,
            dtype=torch.long,
            device=device,
        )

        target_ids = torch.full(
            (
                1,
                num_sequences,
                max_input_len,
            ),
            fill_value=self.pad_token_id,
            dtype=torch.long,
            device=device,
        )

        for index, sequence in enumerate(sequences):
            sequence_tensor = torch.tensor(
                sequence,
                dtype=torch.long,
                device=device,
            )

            decoder_input = sequence_tensor[:-1]
            decoder_target = sequence_tensor[1:]

            length = decoder_input.numel()

            input_ids[
                0,
                index,
                :length,
            ] = decoder_input

            target_ids[
                0,
                index,
                :length,
            ] = decoder_target

        return input_ids, target_ids

    def loss_ocr(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        batch_data_samples: SampleList,
        dn_meta: Optional[Dict[str, int]],
        ocr_memory: Tensor,
        ocr_memory_mask: Optional[Tensor],
        ocr_spatial_shapes: Tensor,
        ocr_level_start_index: Tensor,
        ocr_valid_ratios: Tensor,
    ) -> Tensor:
        """Calculate OCR recognition loss."""
    
        (
            matching_hidden_states,
            matching_cls_scores,
            matching_bbox_preds,
        ) = self._get_matching_outputs(
            hidden_states=hidden_states,
            references=references,
            dn_meta=dn_meta,
        )
    
        loss_sum = (
            matching_hidden_states.sum()
            * 0.0
        )
    
        total_tokens = 0
    
        batch_size = (
            matching_hidden_states.shape[0]
        )
    
        for image_index in range(batch_size):
            data_sample = batch_data_samples[
                image_index
            ]
    
            gt_instances = (
                data_sample.gt_instances
            )
    
            if len(gt_instances) == 0:
                continue
    
            if not hasattr(
                gt_instances,
                'ocr_token_ids',
            ):
                continue
    
            (
                query_inds,
                gt_inds,
            ) = self._assign_single_image(
                cls_score=matching_cls_scores[
                    image_index
                ],
                bbox_pred=matching_bbox_preds[
                    image_index
                ],
                gt_instances=gt_instances,
                img_meta=data_sample.metainfo,
            )
    
            if query_inds.numel() == 0:
                continue
    
            (
                query_inds,
                gt_inds,
            ) = self._filter_ocr_matches(
                query_inds=query_inds,
                gt_inds=gt_inds,
                gt_instances=gt_instances,
            )
    
            if query_inds.numel() == 0:
                continue
    
            image_memory = ocr_memory[
                image_index:
                image_index + 1
            ]
    
            if ocr_memory_mask is None:
                image_memory_mask = None
            else:
                image_memory_mask = (
                    ocr_memory_mask[
                        image_index:
                        image_index + 1
                    ]
                )
    
            image_valid_ratios = (
                ocr_valid_ratios[
                    image_index:
                    image_index + 1
                ]
            )
    
            num_matches = query_inds.numel()
    
            for start in range(
                0,
                num_matches,
                self.ocr_chunk_size,
            ):
                end = min(
                    start + self.ocr_chunk_size,
                    num_matches,
                )
    
                chunk_query_inds = query_inds[
                    start:end
                ]
    
                chunk_gt_inds = gt_inds[
                    start:end
                ]
    
                selected_queries = (
                    matching_hidden_states[
                        image_index:
                        image_index + 1,
                        chunk_query_inds,
                        :,
                    ]
                )
    
                selected_boxes = (
                    matching_bbox_preds[
                        image_index:
                        image_index + 1,
                        chunk_query_inds,
                        :,
                    ]
                )
    
                (
                    input_ids,
                    target_ids,
                ) = self._build_teacher_forcing_batch(
                    gt_instances=gt_instances,
                    gt_inds=chunk_gt_inds,
                    device=hidden_states.device,
                )
    
                block_visual_features = (
                    self.block_visual_extractor(
                        object_queries=(
                            selected_queries
                        ),
                        object_boxes=(
                            selected_boxes
                        ),
                        memory=image_memory,
                        memory_mask=(
                            image_memory_mask
                        ),
                        spatial_shapes=(
                            ocr_spatial_shapes
                        ),
                        level_start_index=(
                            ocr_level_start_index
                        ),
                        valid_ratios=(
                            image_valid_ratios
                        ),
                    )
                )
    
                logits = self.block_text_decoder(
                    block_visual_features=(
                        block_visual_features
                    ),
                    input_ids=input_ids,
                    block_queries=selected_queries,
                )
    
                chunk_loss = F.cross_entropy(
                    logits.reshape(
                        -1,
                        self.ocr_vocab_size,
                    ),
                    target_ids.reshape(-1),
                    ignore_index=self.pad_token_id,
                    reduction='sum',
                )
    
                valid_token_count = int(
                    (
                        target_ids
                        != self.pad_token_id
                    ).sum().item()
                )
    
                loss_sum = (
                    loss_sum
                    + chunk_loss
                )
    
                total_tokens += (
                    valid_token_count
                )
    
        if total_tokens == 0:
            return loss_sum
    
        loss_ocr = (
            loss_sum
            / float(total_tokens)
        )
    
        return (
            loss_ocr
            * self.ocr_loss_weight
        )

    def forward_ocr(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        ocr_memory: Tensor,
        ocr_memory_mask: Optional[Tensor],
        ocr_spatial_shapes: Tensor,
        ocr_level_start_index: Tensor,
        ocr_valid_ratios: Tensor,
        query_indices: Tensor,
        input_ids: Tensor,
        dn_meta: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Tensor]:
        """Forward OCR branch for selected DINO queries."""

        (
            matching_hidden_states,
            _,
            matching_bbox_preds,
        ) = self._get_matching_outputs(
            hidden_states=hidden_states,
            references=references,
            dn_meta=dn_meta,
        )

        selected_queries = self._gather_queries(
            matching_hidden_states,
            query_indices,
        )

        selected_boxes = self._gather_queries(
            matching_bbox_preds,
            query_indices,
        )

        block_visual_features = (
            self.block_visual_extractor(
                object_queries=selected_queries,
                object_boxes=selected_boxes,
                memory=ocr_memory,
                memory_mask=ocr_memory_mask,
                spatial_shapes=ocr_spatial_shapes,
                level_start_index=(
                    ocr_level_start_index
                ),
                valid_ratios=ocr_valid_ratios,
            )
        )

        ocr_logits = self.block_text_decoder(
            block_visual_features=(
                block_visual_features
            ),
            input_ids=input_ids,
            block_queries=selected_queries,
        )

        return dict(
            selected_queries=selected_queries,
            selected_boxes=selected_boxes,
            block_visual_features=(
                block_visual_features
            ),
            ocr_logits=ocr_logits,
        )