from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal

import bayes_sse.filters as bf
from bayes_sse._utils.stat_utils import stat_errors
from bayes_sse.bayesian_filter_type import (
    BayesianFilterType,
    is_linear_kalman_filter,
    is_sigma_point_filter,
    is_sqrt_sigma_point_filter,
)
from bayes_sse.inference.inference_model_generator import build_filterable_model
from bayes_sse.inference.stochastic_models.covariance_type import CovarianceType
from bayes_sse.inference.stochastic_models.noise_type import NoiseType
from bayes_sse.inference.stochastic_models.stochastic_models import (
    build_stochastic_process,
)
from bayes_sse.simulations.state_space_models import (
    gamma_process_gaussian_observation as gamma_gssm,
)

SCENARIO_NAME = "gamma-process-gaussian-observation"
HELP = "To run simulation of Gamma process/Gaussian observation scenario use: \
bayes_sse simulate gamma-process-gaussian-observation. \
Arguments: \
   --filter FILTER: filter being used; default: pf\
   --runs RUNS: number of Monte Carlo simulation runs; default: 500 \
   --points POINTS: number of time points generated per run; default: 200 \
   --particles PARTICLES: number of particles used by particle-based filters (default: 5000) \
   --seed SEED: optional NumPy random seed for reproducible runs; default: unset \
   --draw: open matplotlib plots after the simulation; blocks until the plot windows are closed \
   --no-draw: run without opening plot windows (default) \
   --save-plots DIR: save generated plots as PNG files in DIR"
SCENARIO_DESCRIPTION = (
    "Performs recursive Bayesian state estimation for a nonlinear state-space model \
whose state dynamics include gamma-distributed process noise and whose \
observations include Gaussian noise."
)

SUPPORTED_FILTERS = (
    BayesianFilterType.kf,
    BayesianFilterType.ekf,
    BayesianFilterType.ukf,
    BayesianFilterType.cdkf,
    BayesianFilterType.srukf,
    BayesianFilterType.srcdkf,
    BayesianFilterType.ckf,
    BayesianFilterType.srckf,
    BayesianFilterType.fdckf,
    BayesianFilterType.cqkf,
    BayesianFilterType.ghqf,
    BayesianFilterType.sghqf,
    BayesianFilterType.pf,
    BayesianFilterType.gspf,
)


@dataclass(frozen=True)
class RunConfig:
    filter_type: BayesianFilterType = BayesianFilterType.pf
    number_of_runs: int = 500
    data_points_count: int = 200
    n_particles: int = 5_000
    seed: int | None = None
    draw: bool = False
    plot_dir: Path | None = None


def run(config: RunConfig | None = None) -> None:
    config = config or RunConfig()
    _validate_config(config)

    if config.seed is not None:
        np.random.seed(config.seed)

    err_arr = np.zeros((config.number_of_runs, config.data_points_count - 1))

    filter_type = config.filter_type
    print(filter_type.name)

    figures = []
    plt = None
    fig_a = None
    ax_a = None
    if config.draw or config.plot_dir is not None:
        if not config.draw:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_a, ax_a = plt.subplots()
        figures.append((fig_a, "sample_estimate"))

    gssm_model = gamma_gssm.build(4e-2, 0.5, 3, 0.5, 0, 1e-2, CovarianceType.sqrt)
    inference_model = build_filterable_model(gssm_model, filter_type)

    for i in range(config.number_of_runs):
        x = np.zeros((gssm_model.state_dim, config.data_points_count), dtype=np.float64)
        z = np.zeros(
            (gssm_model.observation_dim, config.data_points_count), dtype=np.float64
        )

        x_noise = gssm_model.state_noise.sample(config.data_points_count)
        z_noise = gssm_model.observation_noise.sample(config.data_points_count)

        u_x = np.asarray(range(config.data_points_count))
        u_z = np.asarray(range(1, config.data_points_count + 1))

        z[0] = gssm_model.observation_func(x[:, 0], z_noise[:, 0], np.asarray([1]))
        for j in range(1, config.data_points_count):
            x[:, j] = gssm_model.transition_func(
                x[:, j - 1], x_noise[:, j - 1], u_x[j - 1]
            )
            z[:, j] = gssm_model.observation_func(x[:, j], z_noise[:, j], u_z[j])

        x_est = np.zeros((inference_model.state_dim, config.data_points_count))
        x_est[:, 0] = 2
        x_cov_est = 3 / 4 * np.eye(inference_model.state_dim)

        if is_sigma_point_filter(filter_type) or is_linear_kalman_filter(filter_type):
            spkf = _build_sigma_or_linear_filter(filter_type)

            if is_sqrt_sigma_point_filter(filter_type):
                x_cov_est = np.linalg.cholesky(x_cov_est)

            for k in range(1, config.data_points_count):
                x_est[:, k], x_cov_est, _ = spkf.estimate(
                    x_est[:, k - 1],
                    x_cov_est,
                    z[:, k],
                    inference_model,
                    u_x[k - 1],
                    u_z[k],
                )
        elif filter_type is BayesianFilterType.pf:
            resample_strategy = bf.Resampler.resolve(bf.ResamplingMethod.residual)
            pf = bf.BPF(ess_threshold_ratio=0.1, resampler=resample_strategy)
            particles = np.atleast_2d(
                multivariate_normal.rvs(x_est[:, 0], size=config.n_particles)
            )
            weights = np.tile(1 / config.n_particles, config.n_particles)
            data_set = bf.WeightedParticleSet(particles, weights)

            for k in range(1, config.data_points_count):
                x_est[:, k], data_set = pf.estimate(
                    data_set, z[:, k], inference_model, u_x[k - 1], u_z[k]
                )
        elif filter_type is BayesianFilterType.gspf:
            n_mixture = 4
            initial_cov = np.squeeze(inference_model.state_noise.covariance)
            gm_cov = np.zeros((n_mixture, 1, 1))
            gm_cov[0, :, :] = 1.75 * initial_cov
            gm_cov[1, :, :] = 0.05 * initial_cov
            gm_cov[2, :, :] = 0.85 * initial_cov
            gm_cov[3, :, :] = 2.25 * initial_cov
            gm_state_noise = build_stochastic_process(
                NoiseType.gaussian_mixture,
                mixture_size=n_mixture,
                mean=np.tile(
                    np.asarray(np.squeeze(inference_model.state_noise.mean)),
                    (n_mixture, 1),
                ),
                covariance=gm_cov,
                covariance_type=inference_model.state_noise.covariance_type,
                weights=1 / n_mixture * np.ones(n_mixture),
            )
            gmm_inference_model = inference_model.set_state_noise(gm_state_noise)

            resample_strategy = bf.Resampler.resolve(bf.ResamplingMethod.residual)
            gspf = bf.GSPF(config.n_particles, resampler=resample_strategy)
            gmi = bf.init_gmi(
                x_est[:, 0],
                gmm_inference_model.state_dim,
                config.n_particles,
                n_mixture,
            )
            for k in range(1, config.data_points_count):
                x_est[:, k], gmi = gspf.estimate(
                    gmi, z[:, k], gmm_inference_model, u_x[k - 1], u_z[k]
                )
        else:
            raise ValueError(f"Not supported filter type: {filter_type.name}")

        if (
            (config.draw or config.plot_dir is not None)
            and i == 0
            and fig_a is not None
            and ax_a is not None
        ):
            ax_a.plot(np.squeeze(x), linewidth=2.0, label="clean", linestyle="dashed")
            ax_a.plot(np.squeeze(z), linewidth=2.0, label="noisy", linestyle="dotted")
            ax_a.plot(
                np.squeeze(x_est), linewidth=2.0, label=f"{filter_type.name} estimate"
            )
            ax_a.grid(True, which="both", axis="both")
            ax_a.legend(loc="upper right")
            ax_a.set_title(
                f"{filter_type.name}: Nonlinear Time Variant State Estimation \n (non Gaussian noise)",
                fontsize=12,
            )
            plt.tight_layout()
            fig_a.set_size_inches(8, 6)

        err_arr[i, :] = x_est[:, 1:] - x[:, 1:]
        _print_progress(i, config.number_of_runs)

    x_mean_err, std_x_err, rmse_x_err = stat_errors(err_arr)

    if (config.draw or config.plot_dir is not None) and plt is not None:
        fig1, ax1 = plt.subplots()
        ax1.plot(x_mean_err, color="b", linewidth=2.0, label="mean rmse")
        ax1.plot(
            x_mean_err - std_x_err,
            color="r",
            linewidth=2.0,
            label="-1 sigma",
            linestyle="dashed",
        )
        ax1.plot(
            x_mean_err + std_x_err,
            color="r",
            linewidth=2.0,
            label="+1 sigma",
            linestyle="dashed",
        )
        ax1.legend(loc="upper right")
        ax1.grid(True, which="both", axis="both")
        plt.tight_layout()
        figures.append((fig1, "mean_error"))

        fig2, ax2 = plt.subplots()
        ax2.plot(rmse_x_err, color="g", linewidth=2.0)
        ax2.set_title(f"{filter_type.name}: RMSE (non Gaussian noise)", fontsize=12)
        ax2.grid(True, which="both", axis="both")
        figures.append((fig2, "rmse"))

        if config.plot_dir is not None:
            config.plot_dir.mkdir(parents=True, exist_ok=True)
            for fig, name in figures:
                plot_path = config.plot_dir / f"{filter_type.name}_{name}.png"
                fig.savefig(plot_path, dpi=150, bbox_inches="tight")
                print(f"saved plot: {plot_path}")

        if not config.draw:
            plt.close("all")
            return

        print("Showing plots. Close the matplotlib windows to return to the shell.")
        plt.show()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--filter",
        choices=[filter_type.name for filter_type in SUPPORTED_FILTERS],
        default=RunConfig.filter_type.name,
        help="Bayesian filter to use.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=RunConfig.number_of_runs,
        help="Number of Monte Carlo runs.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=RunConfig.data_points_count,
        help="Number of data points per run.",
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=RunConfig.n_particles,
        help="Particle count for particle filters.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Optional NumPy random seed."
    )
    parser.add_argument(
        "--save-plots",
        type=Path,
        default=None,
        help="Save plots as PNG files in this directory.",
    )
    draw_group = parser.add_mutually_exclusive_group()
    draw_group.add_argument(
        "--draw",
        dest="draw",
        action="store_true",
        help="Open matplotlib figures and block until the plot windows are closed.",
    )
    draw_group.add_argument(
        "--no-draw", dest="draw", action="store_false", help="Skip matplotlib figures."
    )
    parser.set_defaults(draw=RunConfig.draw)


def config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        filter_type=BayesianFilterType[args.filter],
        number_of_runs=args.runs,
        data_points_count=args.points,
        n_particles=args.particles,
        seed=args.seed,
        draw=args.draw,
        plot_dir=args.save_plots,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the BayesSSE gamma-process Gaussian-observation example."
    )
    add_arguments(parser)
    args = parser.parse_args(argv)
    run(config_from_args(args))
    return 0


def _build_sigma_or_linear_filter(filter_type: BayesianFilterType):
    if filter_type is BayesianFilterType.ukf:
        return bf.UKF(alpha=1, beta=2, kappa=0)
    if filter_type is BayesianFilterType.srukf:
        return bf.SRUKF(alpha=1, beta=2, kappa=0)
    if filter_type is BayesianFilterType.cdkf:
        return bf.CDKF(scale_factor=np.sqrt(3))
    if filter_type is BayesianFilterType.srcdkf:
        return bf.SRCDKF(scale_factor=np.sqrt(3))
    if filter_type is BayesianFilterType.ckf:
        return bf.CKF()
    if filter_type is BayesianFilterType.srckf:
        return bf.SRCKF()
    if filter_type is BayesianFilterType.fdckf:
        return bf.FDCKF()
    if filter_type is BayesianFilterType.cqkf:
        return bf.CQKF(order=9)
    if filter_type is BayesianFilterType.ghqf:
        return bf.GHQF(order=22)
    if filter_type is BayesianFilterType.sghqf:
        return bf.SGHQF(order=11, manner=3)
    if filter_type is BayesianFilterType.kf:
        return bf.KF()
    if filter_type is BayesianFilterType.ekf:
        return bf.EKF()

    raise ValueError(f"Not supported filter type: {filter_type.name}")


def _validate_config(config: RunConfig) -> None:
    if config.number_of_runs < 1:
        raise ValueError("--runs must be at least 1")
    if config.data_points_count < 2:
        raise ValueError("--points must be at least 2")
    if config.n_particles < 1:
        raise ValueError("--particles must be at least 1")
    if config.filter_type not in SUPPORTED_FILTERS:
        raise ValueError(f"Not supported filter type: {config.filter_type.name}")


def _print_progress(run_index: int, number_of_runs: int) -> None:
    completed_runs = run_index + 1
    progress = completed_runs * 100 / number_of_runs
    if completed_runs == number_of_runs or progress % 10 == 0.0:
        print(f"\t{progress} % completed")


if __name__ == "__main__":
    raise SystemExit(main())
