

from pytorch_trainer.training.extensions.linear_shift import LinearShift


class ObjectLinearShift(LinearShift):
    def __init__(self, attr, value_range, time_range, target):
        super().__init__(attr, value_range, time_range, optimizer=target)

    def _update_value(self, target, value):
        setattr(target, self._attr, value)
        self._last_value = value
