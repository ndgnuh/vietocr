from torch import optim

optimizers = {}


def r(name):
    def wrapped(callback):
        optimizers[name] = callback
    return wrapped


@r("adam")
def _(parameters, lr):
    return optim.Adam(
        parameters,
        lr=lr,
        betas=(0.9, 0.98),
        weight_decay=1e-5,
        amsgrad=True,
    )


@r("sgd_momentum")
def _(parameters, lr):
    return optim.SGD(parameters, lr=lr, momentum=0.9, nesterov=True)


@r("sgd")
def _(parameters, lr):
    return optim.SGD(parameters, lr=lr)


@r("yogi")
def _(parameters, lr):
    import torch_optimizer as toptim
    return toptim.Yogi(
        parameters,
        lr=lr,
        betas=(0.9, 0.98),
        weight_decay=1e-5,
    )


@r("adahessian")
def _(parameters, lr):
    import torch_optimizer as toptim
    opt = toptim.Adahessian(
        parameters,
        lr=lr,
        betas=(0.9, 0.98),
        weight_decay=1e-5,
    )
    opt.create_graph = True
    return opt


def get_optimizer(optimizer_config, parameters, lr):
    if isinstance(optimizer_config, str):
        return optimizers[optimizer_config](parameters, lr)

    OptimClass = getattr(optimizer_config.pop("name"))
    return OptimClass(parameters, **optimizer_config)
