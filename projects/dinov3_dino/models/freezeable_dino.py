from __future__ import annotations

from mmdet.models.detectors.dino import DINO
from mmdet.registry import MODELS


@MODELS.register_module()
class FreezeableDINO(DINO):
    """DINO supporting neck-only training.

    train_neck_only=True:
        freeze backbone + DINO detector,
        only neck is trainable.

    train_neck_only=False:
        normal DINO training.
        Backbone freezing is controlled by backbone.frozen.
    """

    def __init__(
        self,
        train_neck_only: bool = False,
        **kwargs,
    ):
        self.train_neck_only = bool(train_neck_only)

        super().__init__(**kwargs)

        if self.train_neck_only:
            self._freeze_except_neck()

    def _freeze_except_neck(self):

        # Freeze complete model.
        for param in self.parameters():
            param.requires_grad = False

        # Re-enable neck.
        for param in self.neck.parameters():
            param.requires_grad = True