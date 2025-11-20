import torch
import torch.nn.functional as F

def to_var(x):
    return to_cuda(x).requires_grad_()

def to_cuda(x):

    if torch.cuda.is_available():
        x = x.cuda()
    return x

def flatten(lists):
    return [item for l in lists for item in l]

def pad_and_stack(tensors, pad_size=None, value=0):

    sizes = [s.shape[0] for s in tensors]

    if not pad_size:
        pad_size = max(sizes)

    padded = torch.stack([F.pad(input=sent[:pad_size],
                                pad=(0, 0, 0, max(0, pad_size-size)),
                                value=value)
                          for sent, size in zip(tensors, sizes)], dim=0)

    return padded, sizes


if __name__ == "__main__":
    pass