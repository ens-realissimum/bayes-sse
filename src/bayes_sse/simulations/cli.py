from __future__ import annotations

import argparse
from collections.abc import Sequence

from bayes_sse.simulations.scenarios.registry import Scenario, load_scenarios


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bayes-sse simulate",
        description="Command line tools to run a BayesSSE simulation scenario",
        usage="bayes-sse simulate <SCENARIO>",
    )
    subparsers = parser.add_subparsers(
        dest="scenario", metavar="SCENARIO", required=True
    )

    for scenario in load_scenarios():
        scenario_parser = subparsers.add_parser(
            scenario.name, help=scenario.help, description=scenario.description
        )
        scenario.module.add_arguments(scenario_parser)
        scenario_parser.set_defaults(scenario=scenario)

    args = parser.parse_args(argv)
    scenario: Scenario = args.scenario
    scenario.module.run(scenario.module.config_from_args(args))
    return 0
