from model import GPModel
import torch 
from update_model import update_model
import gpytorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(
    N=2000,
    d=12,
    num_inducing_points=1024,
):
    train_x = torch.randn(N,d)
    train_y = torch.randn(N,1)

    model = GPModel(
        inducing_points=train_x[0:num_inducing_points].to(device), 
        likelihood=gpytorch.likelihoods.GaussianLikelihood().to(device),
    ).to(device)

    mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=train_x.size(-2))

    model = update_model(
        model=model,
        mll=mll,
        train_x=train_x,
        train_y=train_y,
    )

if __name__ == "__main__":
    main(
        N=50,
        d=3,
        num_inducing_points=50,
    )