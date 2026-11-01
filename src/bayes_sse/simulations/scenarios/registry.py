from __future__ import annotations
import argparse
from copy import Error
import importlib
import pkgutil
from dataclasses import dataclass
from typing import Any, Protocol, cast

from bayes_sse.simulations import scenarios


class ScenarioModule(Protocol):
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        ...

    def config_from_args(self, args: argparse.Namespace) -> Any:
        ...

    def run(self, config: Any) -> None:
        ...


@dataclass(frozen=True)
class Scenario:
    name: str
    help: str
    description: str
    module: ScenarioModule


def load_scenarios() -> tuple[Scenario, ...]:
    discovered_scenarios: list[Scenario] = []
    scenario_names: set[str] = set()

    for module_info in pkgutil.iter_modules(scenarios.__path__, f"{scenarios.__name__}."):
        if module_info.name == __name__ or module_info.ispkg:
            continue

        module = importlib.import_module(module_info.name)
        _validate_scenario_module(module_info.name, module)

        name = _scenario_name(module_info.name, module)
        if name in scenario_names:
            raise Error(f"Duplicated scenarios: {name}")

        scenario_names.add(name)
        discovered_scenarios.append(
            Scenario(
                name = name,
                description = _scenario_description(module_info.name, module),
                help = _scenario_help(name, module),
                module = cast(ScenarioModule, module),
            )
        )

    return tuple(sorted(discovered_scenarios, key = lambda scenario: scenario.name))


def _scenario_name(module_name: str, module: object) -> str:
    fallback_name = module_name.rsplit(".", 1)[-1].replace("_", "-")
    name = getattr(module, "SCENARIO_NAME", fallback_name)
    if not isinstance(name, str):
        raise ImportError(f"Scenario module {module_name!r} has a non-string SCENARIO_NAME")

    return name


def _scenario_description(module_name: str, module: object) -> str:
    fallback_description = module_name.rsplit(".", 1)[-1].replace("_", " ")
    description = getattr(module, "SCENARIO_DESCRIPTION", fallback_description)

    if not isinstance(description, str):
        raise ImportError(f"Scenario module {module_name!r} has a non-string SCENARIO_DESCRIPTION")

    return description


def _scenario_help(name: str, module: object) -> str:
    help_text = getattr(module, "HELP", f"Run the {name} scenario.")
    if not isinstance(help_text, str):
        raise ImportError(f"Scenario {name!r} has a non-string HELP")
    return help_text


def _validate_scenario_module(module_name: str, module: object) -> None:
    missing = [
        attribute
        for attribute in ("add_arguments", "config_from_args", "run")
        if not hasattr(module, attribute)
    ]
    if missing:
        missing_attributes = ", ".join(missing)
        raise ImportError(f"Scenario module {module_name!r} is missing: {missing_attributes}")
