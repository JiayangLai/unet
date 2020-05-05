from torch import nn
import torch
import numpy as np


def cal_pts_align_false(input_size, factor):
    inter = 1 / float(factor)
    dst_ws = np.arange(0, input_size, inter) + (inter - 1) / 2

    dst_ws[np.where(dst_ws < 0)] = 0
    dst_ws[np.where(dst_ws > (input_size - 1))] = (input_size - 1)

    return dst_ws


def find_pt(pt, pt_lim):
    if pt != pt_lim:
        x0 = np.floor(pt)
        accr_pt = pt - x0
    else:
        x1 = np.ceil(pt)
        x0 = x1 - 1
        accr_pt = pt - x0
    return accr_pt, int(x0)


def get_weights_4_corners(pt_w, pt_h):
    weight00 = 1 - pt_w - pt_h + pt_w * pt_h
    weight10 = pt_w * (1 - pt_h)
    weight01 = pt_h * (1 - pt_w)
    weight11 = pt_w * pt_h
    return torch.tensor([weight00, weight10, weight01, weight11])


class Bilinear_upsample(nn.Module):
    def __init__(self, scale_factor=2, align_corner=False):
        super(Bilinear_upsample, self).__init__()
        self.upsamp = self.cal(scale_factor,align_corner)

    def cal(self, scale_factor=2, align_corner=False):
        def core(X):
            X_out = torch.zeros([X.shape[0], X.shape[1], X.shape[2] * scale_factor, X.shape[3] * scale_factor])
            for batchi in range(X.shape[0]):
                x = X[batchi]
                for chi in range(X.shape[1]):
                    x_ch1 = x[0]
                    w_src = x_ch1.shape[0]
                    h_src = x_ch1.shape[1]
                    # 以2*2个像素点为最小单位
                    scale = (float(scale_factor), float(scale_factor))  # 倍数为2

                    n_dst_w = scale[0] * w_src
                    n_dst_h = scale[1] * h_src
                    n_interp_w = n_dst_w - 2  # 需要插入的点的个数
                    n_interp_h = n_dst_h - 2  # 需要插入的点的个数
                    if align_corner:
                        pts_w = (w_src - 1) * np.arange(0.0, n_dst_w, 1.0) / (n_interp_w + 1)  # 新的小数形式的坐标
                        pts_h = (h_src - 1) * np.arange(0.0, n_dst_h, 1.0) / (n_interp_h + 1)  # 新的小数形式的坐标
                    else:
                        pts_w = cal_pts_align_false(w_src, scale_factor)
                        pts_h = cal_pts_align_false(h_src, scale_factor)
                    # print(n_dst_w, n_dst_h)
                    x_ch1_dst = torch.zeros([int(n_dst_w), int(n_dst_h)])

                    for i, pt_w in enumerate(pts_w):
                        for j, pt_h in enumerate(pts_h):
                            accr_pt_w, x0 = find_pt(pt_w, pts_w[-1])
                            accr_pt_h, y0 = find_pt(pt_h, pts_h[-1])
                            weights = get_weights_4_corners(accr_pt_w, accr_pt_h)

                            q00 = x_ch1[x0, y0]
                            q10 = x_ch1[x0 + 1, y0]
                            q01 = x_ch1[x0, y0 + 1]
                            q11 = x_ch1[x0 + 1, y0 + 1]
                            qs = torch.tensor([q00, q10, q01, q11])
                            x_ch1_dst[i, j] = (weights * qs).sum()
                    X_out[batchi, chi, :, :] = x_ch1_dst
            return X_out

        return core

    def forward(self, x):
        if x.dim() != 4:
            print('Input tensor should be 4D only.')
            assert x.dim() == 4

        return self.upsamp(x)


if __name__ == '__main__':
    X = torch.tensor([[[[1.0, 2.0],
                        [3.0, 4.0]]]])
    scale_factor = 2
    out = Bilinear_upsample(scale_factor=scale_factor, align_corner=True)(X)
    print(out)
    out = Bilinear_upsample(scale_factor=scale_factor, align_corner=False)(X)
    print(out)