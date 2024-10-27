import torch


class InputNorm(torch.nn.Module):

    def __init__(
        self,
        num_features: int,
        affine: bool = True,
        lambda_range: tuple[float, float] = (-5.0, 5.0),
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_features = num_features
        assert len(lambda_range) == 2, "Lambda range must be a tuple of size 2"
        self.lambda_range = lambda_range
        self.lambdas = torch.nn.Parameter(torch.ones(num_features, **factory_kwargs))
        self.affine = affine
        if self.affine:
            self.weight = torch.nn.Parameter(torch.ones(num_features, **factory_kwargs))  # scale / gamma
            self.bias = torch.nn.Parameter(torch.zeros(num_features, **factory_kwargs))  # center / beta
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(x)
        x = torch.where(mask, x, 0)
        l = torch.clamp(self.lambdas, *self.lambda_range)
        l = torch.where(x >= 0, l, 2 - l)
        log1p = torch.log1p(torch.abs(x))
        yj = torch.where(l == 0, log1p, (torch.exp(l * log1p) - 1) / l)
        x = torch.where(mask, torch.sign(x) * yj, 0)
        if self.affine:
            x = torch.where(mask, self.weight * x + self.bias, 0)  # "standardize" output
        return torch.where(mask, x, torch.nan)  # reintroduce nans