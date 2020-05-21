import numpy as np

import CONFIG


def fast_hist(a, b, n):
    """
    Return a histogram that's the confusion matrix of a and b
    :param a: np.ndarray with shape (HxW,)
    :param b: np.ndarray with shape (HxW,)
    :param n: num of classes
    :return: np.ndarray with shape (n, n)
    """
    # print(n)
    # print(b.max())
    k = (a >= 0) & (a < n)


    # a = np.floor(a)
    # a = a.astype(np.int)
    # print(a.max())
    # print(a.dtype)
    # print(a.shape)
    # print(type(a))

    return np.bincount((n * a[k].astype(int) + b[k]).astype(int), minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    """
    Calculate the IoU(Intersection over Union) for each class
    :param hist: np.ndarray with shape (n, n)
    :return: np.ndarray with shape (n,)
    """
    np.seterr(divide="ignore", invalid="ignore")
    res = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    np.seterr(divide="warn", invalid="warn")
    res[np.isnan(res)] = 0.
    return res


class ComputeIoU(object):
    """
    IoU: Intersection over Union
    """

    def __init__(self):
        self.cfsmatrix = np.zeros((CONFIG.NUM_CLASSES, CONFIG.NUM_CLASSES), dtype="uint64")  # confusion matrix
        self.ious = dict()

    def get_cfsmatrix(self):
        return self.cfsmatrix

    def get_ious(self):
        self.ious = dict(zip(range(CONFIG.NUM_CLASSES), per_class_iu(self.cfsmatrix)))  # {0: iou, 1: iou, ...}
        return self.ious

    def get_miou(self, ignore=None):
        self.get_ious()
        total_iou = 0
        count = 0
        for key, value in self.ious.items():
            if isinstance(ignore, list) and key in ignore or \
                    isinstance(ignore, int) and key == ignore:
                continue
            total_iou += value
            count += 1
        return total_iou / count

    def __call__(self, pred, label):
        """
        :param pred: [N, H, W]
        :param label:  [N, H, W}
        Channel == 1
        """

        pred = pred.cpu().detach().numpy()
        label = label.cpu().detach().numpy()

        assert pred.shape == label.shape

        self.cfsmatrix += fast_hist(pred.reshape(-1), label.reshape(-1), CONFIG.NUM_CLASSES).astype("uint64")


if __name__ == '__main__':
    print(CONFIG.NUM_CLASSES)
