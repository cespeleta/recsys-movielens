from scipy.stats import loguniform
from sklearn.model_selection import ParameterSampler


def main():
    # Define experiment variables
    experiment_name = "MF_find_lr"
    script = "src/main_cli.py"
    config = "configs/config_mf.yaml"
    target = "rating_scaled"
    n_iter = 10

    # Define parameter distributions
    distributions = {"model.learning_rate": loguniform(0.001, 0.1)}
    sampler = ParameterSampler(distributions, n_iter=n_iter, random_state=123)

    # Constant values for all the runs
    values = f"python {script}"
    values += f" --config {config}"
    values += f" --data.target {target}"
    values += f" --trainer.logger.name {experiment_name}"

    # Dynamic values, for example different learing rates
    for param in sampler:
        print(values, end=" ")
        for k in param:
            print(f"--{k} {param[k]}", end="")
        print()


if __name__ == "__main__":
    main()
