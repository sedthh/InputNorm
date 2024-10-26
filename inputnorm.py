import torch


class InputNorm(torch.nn.Module):

    def __init__(
        self,
        num_features: int,
        affine: bool = True,
        missing_imputation: bool = True,
        lambda_range: tuple[float, float] = (-5.0, 5.0),
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_features = num_features
        self.missing_imputation = missing_imputation
        if self.missing_imputation:
            self.missing = torch.nn.Parameter(torch.zeros(num_features, **factory_kwargs))
        else:
            self.register_parameter("missing", None)
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
        if self.missing_imputation:
            batch = x.shape[0]
            x = torch.where(torch.isfinite(x), x, torch.tile(self.missing, (batch, 1)))  # missing value imputation
        l = torch.clamp(self.lambdas, *self.lambda_range)
        l = torch.where(x >= 0, l, 2 - l)
        log1p = torch.log1p(torch.abs(x))
        yj = torch.where(l == 0, log1p, (torch.exp(l * log1p) - 1) / l)
        x = torch.sign(x) * yj
        if self.affine:
            x = self.weight * x + self.bias  # "standardize" output
        return x
    
