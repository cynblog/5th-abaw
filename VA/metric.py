import numpy as np
import torch


def concordance_correlation_coefficient(y_true, y_pred, eps=1e-8):

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    vx = y_true - mean_true
    vy = y_pred - mean_pred
    cor = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    # print('cor:{}'.format(cor))
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    numerator = 2 * cor * sd_true * sd_pred

    denominator = sd_true**2 + sd_pred**2 + (mean_true - mean_pred) ** 2
    # print('x_s:{}'.format(sd_true))
    # print('down:{}'.format(denominator))
    return numerator / denominator

def concordance_correlation_coefficient_gpu(y_true, y_pred, eps=1e-8):
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)
    vx = y_true - mean_true
    vy = y_pred - mean_pred
    cor = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)))
    # print('cor:{}'.format(cor))
    sd_true = torch.std(y_true)
    sd_pred = torch.std(y_pred)
    numerator = 2 * cor * sd_true * sd_pred

    denominator = sd_true**2 + sd_pred**2 + (mean_true - mean_pred) ** 2

    return numerator / denominator

if __name__ == '__main__':
    import time
    import numpy as np

    test = list(np.random.rand(1600000))

    start_time = time.time()
    test = test + [1]
    print(time.time()-start_time)

    start_time = time.time()
    test.extend([1])
    print(time.time()-start_time)
    # time.sleep(100)
