import torch.distributed as dist
import torch
import os

from sklearn.metrics import classification_report


def reduce_tensor(tensor, n=None):
    if n is None:
        n = dist.get_world_size()
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / n
    return rt


def gather_tensor(tensor_list, tensor):
    gt = tensor.clone()
    dist.all_gather(tensor_list, gt)

    return tensor_list
    


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def sync(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        val = torch.tensor(self.val).cuda()
        sum_v = torch.tensor(self.sum).cuda()
        count = torch.tensor(self.count).cuda()
        self.val = reduce_tensor(val, world_size).item()
        self.sum = reduce_tensor(sum_v, 1).item()
        self.count = reduce_tensor(count, 1).item()
        self.avg = self.sum / self.count


class GatherMeter:
    def __init__(self) -> None:
        self.preds = []
        self.labels = []
        self.num = 0

    def add_batch(self, preds, labels):
        self.preds.extend(list(preds))
        self.labels.extend(list(labels))
        self.num += preds.size(0)

    def sync(self):
        world_size = dist.get_world_size()
        tensor_list = [torch.zeros(self.num, dtype=torch.int64).cuda() for _ in range(world_size)]
        preds = torch.tensor(self.preds).cuda()
        labels = torch.tensor(self.labels).cuda()

        preds = gather_tensor(tensor_list, preds)
        self.preds = [t.cpu() for t in torch.cat(preds, dim=0)]
        labels = gather_tensor(tensor_list, labels)
        self.labels = [t.cpu() for t in torch.cat(labels, dim=0)]


class ClassificationReporter:
    """Compute the metrics used for the classification task."""
    def __init__(self, preds, labels) -> None:
        self.preds = preds
        self.labels = labels

    def get_dict_classification_report(self):
        
        report = classification_report(self.labels, self.preds, digits=4, zero_division=0, output_dict=True)

        accuracy = report['accuracy']
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1_score = report['macro avg']['f1-score']

        metric_dict = {"acc": accuracy, "pre": precision, "rec": recall, "f1": f1_score}

        return metric_dict
    
    def get_str_classification_report(self):
        report = classification_report(self.labels, self.preds, digits=4, zero_division=0, output_dict=False)

        return report


def save_checkpoint(config, epoch, model, max_f1, optimizer, lr_scheduler, logger, working_dir, is_best):
    """Save a checkpoint; only the best model (highest validation macro-F1) is kept."""
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_f1': max_f1,
                  'epoch': epoch,
                  'config': config}

    if is_best:
        best_path = os.path.join(working_dir, 'best.pth')
        torch.save(save_state, best_path)
        logger.info(f"{best_path} saved !!!")


