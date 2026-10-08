from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures.bbox import (
    bbox_cxcywh_to_xyxy,
)
from mmdet.structures import SampleList
from mmdet.utils import (
    InstanceList,
    reduce_mean,
)

from .dino_head import DINOHead


@MODELS.register_module()
class ESTDINOHead(DINOHead):
    """DINO head with text-recognition supervision.

    The recognition branch supports:

        1. matching-query recognition targets generated from
           Hungarian assignment;

        2. denoising-query recognition targets generated directly
           from DINO DN group layout;

        3. character-level cross entropy;

        4. auxiliary recognition losses from every decoder layer.

    Recognition target convention:

        gt_instances.rec:
            [num_gt, text_length]

        valid character IDs:
            [0, num_rec_classes - 1]

        ignored character:
            rec_ignore_index, default -100
    """

    def __init__(
        self,
        *args,
        rec_loss_weight: float = 1.0,
        dn_rec_loss_weight: float = 1.0,
        rec_ignore_index: int = -100,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )

        self.rec_loss_weight = float(
            rec_loss_weight
        )

        self.dn_rec_loss_weight = float(
            dn_rec_loss_weight
        )

        self.rec_ignore_index = int(
            rec_ignore_index
        )

        if self.rec_loss_weight < 0:
            raise ValueError(
                'rec_loss_weight must be '
                'non-negative.'
            )

        if self.dn_rec_loss_weight < 0:
            raise ValueError(
                'dn_rec_loss_weight must be '
                'non-negative.'
            )

    # ========================================================
    # Complete detection + recognition training loss
    # ========================================================

    def loss(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        enc_outputs_class: Tensor,
        enc_outputs_coord: Tensor,
        batch_data_samples: SampleList,
        dn_meta: Dict[str, int],
        recognition_logits: Tensor,
    ) -> Dict[str, Tensor]:
        """Calculate complete OCRDINO training losses.

        This method preserves the original DINO detection loss path
        and additionally attaches recognition losses.

        Args:
            hidden_states:
                [L, B, N_total, C]

            references:
                Initial + intermediate detection references.

            enc_outputs_class:
                Encoder classification predictions.

            enc_outputs_coord:
                Encoder bbox predictions.

            batch_data_samples:
                Training data samples containing:

                    gt_instances.bboxes
                    gt_instances.labels
                    gt_instances.rec

            dn_meta:
                Native DINO denoising metadata.

            recognition_logits:
                [L, B, N_total, T, num_rec_classes]

        Returns:
            Complete loss dictionary containing both:

                DINO detection losses
                OCR recognition losses
        """

        if recognition_logits is None:
            raise ValueError(
                'recognition_logits are required '
                'during OCRDINO training.'
            )

        if recognition_logits.ndim != 5:
            raise ValueError(
                'recognition_logits must have shape '
                '[L, B, N, T, C].'
            )

        if (
            recognition_logits.shape[0]
            != hidden_states.shape[0]
        ):
            raise ValueError(
                'Recognition and detection decoder '
                'layer counts do not match.'
            )

        if (
            recognition_logits.shape[1]
            != hidden_states.shape[1]
        ):
            raise ValueError(
                'Recognition and detection batch '
                'sizes do not match.'
            )

        if (
            recognition_logits.shape[2]
            != hidden_states.shape[2]
        ):
            raise ValueError(
                'Recognition and detection query '
                'counts do not match.'
            )

        # ----------------------------------------------------
        # Extract GT / metadata exactly as stock DINOHead.loss.
        # ----------------------------------------------------

        batch_gt_instances = []
        batch_img_metas = []

        for data_sample in batch_data_samples:

            batch_img_metas.append(
                data_sample.metainfo
            )

            batch_gt_instances.append(
                data_sample.gt_instances
            )

        # ----------------------------------------------------
        # Run detection prediction heads ONCE.
        #
        # These same cls / bbox predictions are used both by:
        #
        #   1. stock DINO detection losses
        #   2. recognition Hungarian target assignment
        #
        # Therefore recognition and detection always use
        # exactly the same per-layer predictions.
        # ----------------------------------------------------

        (
            all_layers_cls_scores,
            all_layers_bbox_preds,
        ) = self(
            hidden_states,
            references,
        )

        # ----------------------------------------------------
        # Original DINO losses.
        # ----------------------------------------------------

        losses = self.loss_by_feat(
            all_layers_cls_scores=(
                all_layers_cls_scores
            ),
            all_layers_bbox_preds=(
                all_layers_bbox_preds
            ),
            enc_cls_scores=(
                enc_outputs_class
            ),
            enc_bbox_preds=(
                enc_outputs_coord
            ),
            batch_gt_instances=(
                batch_gt_instances
            ),
            batch_img_metas=(
                batch_img_metas
            ),
            dn_meta=dn_meta,
        )

        # ----------------------------------------------------
        # OCR recognition losses.
        #
        # matching:
        #   each layer performs its own Hungarian assignment.
        #
        # DN:
        #   all layers share native DN -> GT correspondence.
        # ----------------------------------------------------

        recognition_losses = (
            self.loss_recognition_by_feat(
                all_layers_recognition_logits=(
                    recognition_logits
                ),
                all_layers_cls_scores=(
                    all_layers_cls_scores
                ),
                all_layers_bbox_preds=(
                    all_layers_bbox_preds
                ),
                batch_gt_instances=(
                    batch_gt_instances
                ),
                batch_img_metas=(
                    batch_img_metas
                ),
                dn_meta=dn_meta,
            )
        )

        # Prevent accidental silent key collision.
        duplicated_keys = (
            set(losses.keys())
            & set(
                recognition_losses.keys()
            )
        )

        if duplicated_keys:
            raise RuntimeError(
                'Recognition loss keys collide '
                'with detection loss keys: '
                f'{sorted(duplicated_keys)}'
            )

        losses.update(
            recognition_losses
        )

        return losses

    # ========================================================
    # GT recognition validation
    # ========================================================

    @staticmethod
    def _get_gt_rec(
        gt_instances: InstanceData,
    ) -> Tensor:
        """Get recognition annotations.

        Expected shape:

            [num_gt, text_length]
        """

        if 'rec' not in gt_instances:
            raise KeyError(
                'gt_instances must contain '
                '`rec` for recognition.'
            )

        gt_rec = (
            gt_instances.rec
        )

        if not isinstance(
            gt_rec,
            Tensor,
        ):
            raise TypeError(
                'gt_instances.rec must '
                'be a Tensor.'
            )

        if gt_rec.ndim != 2:
            raise ValueError(
                'gt_instances.rec must have '
                'shape [num_gt, text_length].'
            )

        if (
            gt_rec.shape[0]
            != len(gt_instances)
        ):
            raise ValueError(
                'gt_instances.rec first '
                'dimension must equal '
                'the number of GT instances.'
            )

        if (
            gt_rec.dtype
            != torch.long
        ):
            raise TypeError(
                'gt_instances.rec must '
                'use torch.long.'
            )

        return gt_rec

    # ========================================================
    # DN recognition targets
    # ========================================================

    def _get_dn_recognition_targets_single(
        self,
        gt_instances: InstanceData,
        dn_meta: Dict[str, int],
    ) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        """Build DN recognition targets for one image.

        Only positive DN queries receive recognition supervision.

        Returns:
            rec_targets:
                [num_dn, T]

            rec_valid_mask:
                [num_dn]

            pos_inds:
                positive DN query indices.

            pos_assigned_gt_inds:
                GT index for every positive DN query.
        """

        gt_rec = self._get_gt_rec(
            gt_instances
        )

        num_gt = (
            gt_rec.shape[0]
        )

        num_groups = (
            dn_meta[
                'num_denoising_groups'
            ]
        )

        num_dn = (
            dn_meta[
                'num_denoising_queries'
            ]
        )

        if num_groups <= 0:
            raise ValueError(
                'num_denoising_groups '
                'must be positive.'
            )

        if (
            num_dn
            % num_groups
            != 0
        ):
            raise ValueError(
                'num_denoising_queries '
                'must be divisible by '
                'num_denoising_groups.'
            )

        num_queries_each_group = (
            num_dn
            // num_groups
        )

        text_length = (
            gt_rec.shape[1]
        )

        rec_targets = (
            gt_rec.new_full(
                (
                    num_dn,
                    text_length,
                ),
                self.rec_ignore_index,
            )
        )

        rec_valid_mask = (
            torch.zeros(
                num_dn,
                dtype=torch.bool,
                device=gt_rec.device,
            )
        )

        if num_gt == 0:

            empty = torch.empty(
                0,
                dtype=torch.long,
                device=gt_rec.device,
            )

            return (
                rec_targets,
                rec_valid_mask,
                empty,
                empty,
            )

        # ----------------------------------------------------
        # Same positive indexing as MMDetection DINOHead.
        #
        # Example:
        #
        # num_gt = 2
        # queries_per_group = 4
        #
        # each group:
        #
        #   0 -> positive GT 0
        #   1 -> positive GT 1
        #   2 -> negative
        #   3 -> negative
        # ----------------------------------------------------

        gt_indices = torch.arange(
            num_gt,
            dtype=torch.long,
            device=gt_rec.device,
        )

        repeated_gt_indices = (
            gt_indices
            .unsqueeze(0)
            .repeat(
                num_groups,
                1,
            )
        )

        pos_assigned_gt_inds = (
            repeated_gt_indices.flatten()
        )

        group_offsets = torch.arange(
            num_groups,
            dtype=torch.long,
            device=gt_rec.device,
        )

        pos_inds = (
            group_offsets.unsqueeze(1)
            * num_queries_each_group
            + repeated_gt_indices
        )

        pos_inds = (
            pos_inds.flatten()
        )

        if (
            pos_inds.numel() > 0
            and pos_inds.max().item()
            >= num_dn
        ):
            raise RuntimeError(
                'Computed DN positive '
                'index exceeds '
                'num_denoising_queries.'
            )

        rec_targets[
            pos_inds
        ] = (
            gt_rec[
                pos_assigned_gt_inds
            ]
        )

        rec_valid_mask[
            pos_inds
        ] = True

        return (
            rec_targets,
            rec_valid_mask,
            pos_inds,
            pos_assigned_gt_inds,
        )

    def get_dn_recognition_targets(
        self,
        batch_gt_instances: InstanceList,
        dn_meta: Dict[str, int],
    ) -> Tuple[
        Tensor,
        Tensor,
        List[Tensor],
        List[Tensor],
    ]:
        """Build DN recognition targets for a batch."""

        target_list = []
        valid_mask_list = []

        pos_inds_list = []
        assigned_gt_inds_list = []

        for gt_instances in (
            batch_gt_instances
        ):

            (
                targets,
                valid_mask,
                pos_inds,
                assigned_gt_inds,
            ) = (
                self
                ._get_dn_recognition_targets_single(
                    gt_instances=(
                        gt_instances
                    ),
                    dn_meta=dn_meta,
                )
            )

            target_list.append(
                targets
            )

            valid_mask_list.append(
                valid_mask
            )

            pos_inds_list.append(
                pos_inds
            )

            assigned_gt_inds_list.append(
                assigned_gt_inds
            )

        return (
            torch.stack(
                target_list,
                dim=0,
            ),
            torch.stack(
                valid_mask_list,
                dim=0,
            ),
            pos_inds_list,
            assigned_gt_inds_list,
        )

    # ========================================================
    # Matching recognition targets
    # ========================================================

    def _get_matching_recognition_targets_single(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        gt_instances: InstanceData,
        img_meta: dict,
    ) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        """Build matching recognition target for one image.

        The assignment is performed using exactly the detection
        Hungarian assigner of the current decoder layer.
        """

        gt_rec = self._get_gt_rec(
            gt_instances
        )

        if cls_score.ndim != 2:
            raise ValueError(
                'cls_score must have '
                'shape [Q, C].'
            )

        if (
            bbox_pred.ndim != 2
            or bbox_pred.shape[-1]
            != 4
        ):
            raise ValueError(
                'bbox_pred must have '
                'shape [Q, 4].'
            )

        if (
            cls_score.shape[0]
            != bbox_pred.shape[0]
        ):
            raise ValueError(
                'cls_score and bbox_pred '
                'query counts must match.'
            )

        num_queries = (
            bbox_pred.shape[0]
        )

        text_length = (
            gt_rec.shape[1]
        )

        rec_targets = (
            gt_rec.new_full(
                (
                    num_queries,
                    text_length,
                ),
                self.rec_ignore_index,
            )
        )

        rec_valid_mask = (
            torch.zeros(
                num_queries,
                dtype=torch.bool,
                device=bbox_pred.device,
            )
        )

        img_h, img_w = (
            img_meta[
                'img_shape'
            ][:2]
        )

        factor = (
            bbox_pred.new_tensor(
                [
                    img_w,
                    img_h,
                    img_w,
                    img_h,
                ]
            )
            .unsqueeze(0)
        )

        pred_bboxes = (
            bbox_cxcywh_to_xyxy(
                bbox_pred
            )
            * factor
        )

        pred_instances = (
            InstanceData(
                scores=cls_score,
                bboxes=pred_bboxes,
            )
        )

        assign_result = (
            self.assigner.assign(
                pred_instances=(
                    pred_instances
                ),
                gt_instances=(
                    gt_instances
                ),
                img_meta=img_meta,
            )
        )

        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0,
            as_tuple=False,
        )

        pos_inds = (
            pos_inds
            .squeeze(-1)
            .unique()
        )

        pos_assigned_gt_inds = (
            assign_result.gt_inds[
                pos_inds
            ]
            - 1
        )

        if (
            pos_inds.numel()
            > 0
        ):

            rec_targets[
                pos_inds
            ] = (
                gt_rec[
                    pos_assigned_gt_inds.long()
                ]
            )

            rec_valid_mask[
                pos_inds
            ] = True

        return (
            rec_targets,
            rec_valid_mask,
            pos_inds,
            pos_assigned_gt_inds,
        )

    def get_matching_recognition_targets(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
    ) -> Tuple[
        Tensor,
        Tensor,
        List[Tensor],
        List[Tensor],
    ]:
        """Build matching recognition targets for one decoder layer.

        Args:
            cls_scores:
                [B, Q, C]

            bbox_preds:
                [B, Q, 4]
        """

        if cls_scores.ndim != 3:
            raise ValueError(
                'cls_scores must have '
                'shape [B, Q, C].'
            )

        if (
            bbox_preds.ndim != 3
            or bbox_preds.shape[-1]
            != 4
        ):
            raise ValueError(
                'bbox_preds must have '
                'shape [B, Q, 4].'
            )

        batch_size = (
            cls_scores.shape[0]
        )

        if (
            bbox_preds.shape[0]
            != batch_size
        ):
            raise ValueError(
                'cls_scores and bbox_preds '
                'batch sizes must match.'
            )

        if (
            len(batch_gt_instances)
            != batch_size
        ):
            raise ValueError(
                'batch_gt_instances '
                'size mismatch.'
            )

        if (
            len(batch_img_metas)
            != batch_size
        ):
            raise ValueError(
                'batch_img_metas '
                'size mismatch.'
            )

        target_list = []
        valid_mask_list = []

        pos_inds_list = []
        assigned_gt_inds_list = []

        for image_index in range(
            batch_size
        ):

            (
                targets,
                valid_mask,
                pos_inds,
                assigned_gt_inds,
            ) = (
                self
                ._get_matching_recognition_targets_single(
                    cls_score=(
                        cls_scores[
                            image_index
                        ]
                    ),
                    bbox_pred=(
                        bbox_preds[
                            image_index
                        ]
                    ),
                    gt_instances=(
                        batch_gt_instances[
                            image_index
                        ]
                    ),
                    img_meta=(
                        batch_img_metas[
                            image_index
                        ]
                    ),
                )
            )

            target_list.append(
                targets
            )

            valid_mask_list.append(
                valid_mask
            )

            pos_inds_list.append(
                pos_inds
            )

            assigned_gt_inds_list.append(
                assigned_gt_inds
            )

        return (
            torch.stack(
                target_list,
                dim=0,
            ),
            torch.stack(
                valid_mask_list,
                dim=0,
            ),
            pos_inds_list,
            assigned_gt_inds_list,
        )

    # ========================================================
    # Character-level recognition CE
    # ========================================================

    def _recognition_ce_single(
        self,
        recognition_logits: Tensor,
        rec_targets: Tensor,
        rec_valid_mask: Tensor,
    ) -> Tensor:
        """Character CE for one decoder layer.

        Args:
            recognition_logits:
                [B, Q, T, C]

            rec_targets:
                [B, Q, T]

            rec_valid_mask:
                [B, Q]

        Invalid query slots are fully ignored.

        Character positions equal to rec_ignore_index are also
        ignored independently.
        """

        if recognition_logits.ndim != 4:
            raise ValueError(
                'recognition_logits must have '
                'shape [B, Q, T, C].'
            )

        batch_size = (
            recognition_logits.shape[0]
        )

        num_queries = (
            recognition_logits.shape[1]
        )

        text_length = (
            recognition_logits.shape[2]
        )

        num_classes = (
            recognition_logits.shape[3]
        )

        expected_target_shape = (
            batch_size,
            num_queries,
            text_length,
        )

        expected_mask_shape = (
            batch_size,
            num_queries,
        )

        if (
            rec_targets.shape
            != expected_target_shape
        ):
            raise ValueError(
                'Unexpected recognition '
                'target shape: '
                f'{tuple(rec_targets.shape)}, '
                f'expected '
                f'{expected_target_shape}.'
            )

        if (
            rec_valid_mask.shape
            != expected_mask_shape
        ):
            raise ValueError(
                'Unexpected recognition '
                'valid mask shape: '
                f'{tuple(rec_valid_mask.shape)}, '
                f'expected '
                f'{expected_mask_shape}.'
            )

        if (
            rec_targets.dtype
            != torch.long
        ):
            raise TypeError(
                'Recognition targets must '
                'use torch.long.'
            )

        if (
            rec_valid_mask.dtype
            != torch.bool
        ):
            raise TypeError(
                'Recognition valid mask must '
                'use torch.bool.'
            )

        # ----------------------------------------------------
        # Mask entire invalid query slots.
        # ----------------------------------------------------

        effective_targets = (
            rec_targets.clone()
        )

        effective_targets = (
            effective_targets.masked_fill(
                ~rec_valid_mask.unsqueeze(-1),
                self.rec_ignore_index,
            )
        )

        valid_char_mask = (
            effective_targets
            != self.rec_ignore_index
        )

        valid_targets = (
            effective_targets[
                valid_char_mask
            ]
        )

        # ----------------------------------------------------
        # Validate all supervised characters.
        # ----------------------------------------------------

        if valid_targets.numel() > 0:

            if (
                valid_targets.min().item()
                < 0
            ):
                raise ValueError(
                    'Recognition target contains '
                    'a negative class index other '
                    'than rec_ignore_index.'
                )

            if (
                valid_targets.max().item()
                >= num_classes
            ):
                raise ValueError(
                    'Recognition target class '
                    'index exceeds recognition '
                    'classifier output size.'
                )

        flat_logits = (
            recognition_logits.reshape(
                -1,
                num_classes,
            )
        )

        flat_targets = (
            effective_targets.reshape(
                -1
            )
        )

        loss_sum = F.cross_entropy(
            flat_logits,
            flat_targets,
            ignore_index=(
                self.rec_ignore_index
            ),
            reduction='sum',
        )

        num_valid_chars = (
            valid_char_mask
            .sum()
            .to(
                dtype=loss_sum.dtype
            )
        )

        # ----------------------------------------------------
        # Same distributed-normalization philosophy as DETR:
        #
        # local summed loss /
        # globally averaged normalization factor.
        # ----------------------------------------------------

        avg_factor = reduce_mean(
            num_valid_chars.reshape(1)
        )

        avg_factor = (
            torch.clamp(
                avg_factor,
                min=1.0,
            )
            .item()
        )

        loss = (
            loss_sum
            / avg_factor
        )

        return loss

    # ========================================================
    # Six-layer recognition losses
    # ========================================================

    def loss_recognition_by_feat(
        self,
        all_layers_recognition_logits: Tensor,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
    ) -> Dict[str, Tensor]:
        """Calculate matching + DN recognition loss.

        Args:
            all_layers_recognition_logits:
                [L, B, N_total, T, C]

            all_layers_cls_scores:
                [L, B, N_total, det_classes]

            all_layers_bbox_preds:
                [L, B, N_total, 4]

        Every matching decoder layer performs its own Hungarian
        assignment using detection predictions from the same layer.

        DN recognition targets are shared across layers because their
        query-to-GT correspondence is determined by DN group layout.
        """

        if (
            all_layers_recognition_logits.ndim
            != 5
        ):
            raise ValueError(
                'all_layers_recognition_logits '
                'must have shape '
                '[L, B, N, T, C].'
            )

        num_layers = (
            all_layers_recognition_logits
            .shape[0]
        )

        if (
            all_layers_cls_scores.shape[0]
            != num_layers
        ):
            raise ValueError(
                'Recognition and detection '
                'classification layer counts '
                'must match.'
            )

        if (
            all_layers_bbox_preds.shape[0]
            != num_layers
        ):
            raise ValueError(
                'Recognition and detection '
                'bbox layer counts must match.'
            )

        num_dn = (
            dn_meta[
                'num_denoising_queries'
            ]
        )

        num_total = (
            all_layers_recognition_logits
            .shape[2]
        )

        if (
            num_dn < 0
            or num_dn > num_total
        ):
            raise ValueError(
                'Invalid '
                'num_denoising_queries.'
            )

        # ----------------------------------------------------
        # Split recognition output at exact same DINO boundary.
        # ----------------------------------------------------

        all_layers_dn_recognition = (
            all_layers_recognition_logits[
                :,
                :,
                :num_dn,
            ]
        )

        all_layers_matching_recognition = (
            all_layers_recognition_logits[
                :,
                :,
                num_dn:,
            ]
        )

        # ----------------------------------------------------
        # Split detection outputs using stock DINO semantics.
        # ----------------------------------------------------

        (
            all_layers_matching_cls,
            all_layers_matching_bbox,
            all_layers_dn_cls,
            all_layers_dn_bbox,
        ) = self.split_outputs(
            all_layers_cls_scores,
            all_layers_bbox_preds,
            dn_meta,
        )

        del all_layers_dn_cls
        del all_layers_dn_bbox

        if (
            all_layers_matching_recognition
            .shape[2]
            != all_layers_matching_cls.shape[2]
        ):
            raise RuntimeError(
                'Matching recognition and '
                'detection query counts '
                'do not match.'
            )

        # ----------------------------------------------------
        # DN target mapping is identical for every layer.
        # ----------------------------------------------------

        (
            dn_targets,
            dn_valid_mask,
            _,
            _,
        ) = self.get_dn_recognition_targets(
            batch_gt_instances=(
                batch_gt_instances
            ),
            dn_meta=dn_meta,
        )

        matching_losses = []
        dn_losses = []

        # ----------------------------------------------------
        # Each decoder layer receives its OWN Hungarian mapping.
        # ----------------------------------------------------

        for layer_index in range(
            num_layers
        ):

            (
                matching_targets,
                matching_valid_mask,
                _,
                _,
            ) = (
                self
                .get_matching_recognition_targets(
                    cls_scores=(
                        all_layers_matching_cls[
                            layer_index
                        ]
                    ),
                    bbox_preds=(
                        all_layers_matching_bbox[
                            layer_index
                        ]
                    ),
                    batch_gt_instances=(
                        batch_gt_instances
                    ),
                    batch_img_metas=(
                        batch_img_metas
                    ),
                )
            )

            matching_loss = (
                self._recognition_ce_single(
                    recognition_logits=(
                        all_layers_matching_recognition[
                            layer_index
                        ]
                    ),
                    rec_targets=(
                        matching_targets
                    ),
                    rec_valid_mask=(
                        matching_valid_mask
                    ),
                )
                * self.rec_loss_weight
            )

            matching_losses.append(
                matching_loss
            )

            if num_dn > 0:

                dn_loss = (
                    self._recognition_ce_single(
                        recognition_logits=(
                            all_layers_dn_recognition[
                                layer_index
                            ]
                        ),
                        rec_targets=(
                            dn_targets
                        ),
                        rec_valid_mask=(
                            dn_valid_mask
                        ),
                    )
                    * self.dn_rec_loss_weight
                )

            else:

                dn_loss = (
                    all_layers_recognition_logits[
                        layer_index
                    ].sum()
                    * 0.0
                )

            dn_losses.append(
                dn_loss
            )

        loss_dict = {}

        # ----------------------------------------------------
        # Final decoder layer.
        # ----------------------------------------------------

        loss_dict[
            'loss_rec'
        ] = (
            matching_losses[-1]
        )

        loss_dict[
            'dn_loss_rec'
        ] = (
            dn_losses[-1]
        )

        # ----------------------------------------------------
        # Auxiliary decoder layers.
        #
        # For six layers:
        #
        # d0 ... d4 = auxiliary
        # final      = layer 5
        # ----------------------------------------------------

        for layer_index in range(
            num_layers - 1
        ):

            loss_dict[
                f'd{layer_index}.loss_rec'
            ] = (
                matching_losses[
                    layer_index
                ]
            )

            loss_dict[
                f'd{layer_index}.dn_loss_rec'
            ] = (
                dn_losses[
                    layer_index
                ]
            )

        return loss_dict