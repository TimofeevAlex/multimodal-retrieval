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
def create_dict_meters():
    metric_names = (
        ["i2t_meanr", "i2t_medr", "i2t_r@10","i2t_r@5", "i2t_r@1", 
         "t2i_meanr", "t2i_medr", "t2i_r@10","t2i_r@5", "t2i_r@1"]
    )
    metrics = {name: AverageMeter() for name in metric_names}
    return metrics
