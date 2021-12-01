class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / (0.0001 + self.count)

    def __str__(self):
        """
        String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return "%.4f (%.4f)" % (self.val, self.avg)


# can be moved to another class
def create_dict_meters(ks):
    metric_names = (
        ["mr_i2t", "mr_t2i"] + [f"r@{k}_i2t" for k in ks] + [f"r@{k}_t2i" for k in ks]
    )
    metrics = {name: AverageMeter() for name in metric_names}
    return metrics
