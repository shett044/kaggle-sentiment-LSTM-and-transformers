from typing import Any, Callable

from torch.optim.lr_scheduler import ReduceLROnPlateau


class CustomSchedulerLRPlateau(ReduceLROnPlateau):

    def step(self, custom_step_func: Callable, metrics: Any, *args, **kwargs) -> None:
        super(CustomSchedulerLRPlateau, self).step(metrics, *args, **kwargs)
        if self.best and self.best == metrics:
            print(f"Saving the best metric: {metrics}")
            custom_step_func(0)
