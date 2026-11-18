# SPDX-License-Identifier: Apache-2.0
# Adapted from the tested reference integration; model weights are not included.
from .primary import LearnedRepair as Base
from .projection import islands, project
import numpy as np

class LearnedRepair(Base):

    def repair(self, rgb, alpha, boxes, method=None):
        aa, ff, _ = super().repair(rgb, alpha, boxes, 'contrast_guarded')
        aa, ff, _ = project(rgb, aa, ff, boxes, 'bound')
        return (aa, ff, int(np.count_nonzero(alpha != aa)))
