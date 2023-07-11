from torchmetrics import Metric, Accuracy
import torch


class MultitaskAccuracy(Metric):
    def __init__(self, tasks, ignore_index=-1):
        super(MultitaskAccuracy, self).__init__()
        self.accs = torch.nn.ModuleDict()
        for task, n_classes in tasks.items():
            self.accs[task] = Accuracy(task="multiclass", num_classes=n_classes, ignore_index=ignore_index)

    def update(self, pred, target):
        for task in self.accs.keys():
            self.accs[task].update(pred[task], target[task])

    def compute(self):
        return {task: self.accs[task].compute() for task in self.accs.keys()}
