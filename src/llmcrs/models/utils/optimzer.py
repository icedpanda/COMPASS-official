from torch import optim


def get_optimizer(optimizer_name: str):
    # get optimizer from torch.optim
    return getattr(optim, optimizer_name)
