# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Reproduction support for the native abliteration pipeline.

- `load_reproduction_information`: loads a reproduce.json file from disk or URL.
- `check_reproduction_environment`: compares the local environment (system and
  package versions) against the original run and reports mismatches.
- `create_reproduction_folder`: writes a self-contained reproduction bundle
  (reproduce.json, SHA256SUMS, requirements.txt, config.json, README.md and a
  copy of the Optuna study journal) next to an exported model.
- `verify_model_hashes`: checks exported weight files against the hashes stored
  in the reproduction information.
"""

from __future__ import annotations

import json
import os
import platform
import re
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from enum import IntEnum
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlopen

import torch
from rich.table import Table

from .config import NativeConfig
from .utils import print

WEIGHT_EXTENSIONS = (".safetensors",)

# JSON schema version of reproduce.json files. Version 3 is the plugin-scorer
# schema, which stores generic scorer `scores`/`baseline_scores` records.
REPRODUCTION_SCHEMA_VERSION = "3"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_HF_REPO_URL_RE = re.compile(r"^https?://huggingface\.co/([^/\s?#]+)/([^/\s?#]+)")


def resolve_reproduction_source(path: str) -> tuple[str, str]:
    """
    Resolves a --reproduce argument into ("repo_id" | "url" | "path", value).

    Accepted forms:
    - a local path to a reproduce.json file,
    - a direct URL to a reproduce.json file,
    - a Hugging Face model URL (https://huggingface.co/<user>/<repo>) — the
      reproduction information is then read from `reproduce/reproduce.json`
      inside that repository,
    - a Hugging Face repository ID (<user>/<repo>) — same as above.
    """
    url_match = _HF_REPO_URL_RE.match(path)
    if url_match:
        return "repo_id", f"{url_match.group(1)}/{url_match.group(2)}"

    if not path.lower().startswith(("http://", "https://")):
        # A single-segment-pair reference that is not an existing filesystem
        # path is treated as a Hugging Face repository ID ("username/model").
        if (
            not Path(path).exists()
            and os.path.sep not in path
            and path.count("/") == 1
        ):
            candidate, filename = path.split("/", 1)
            if (
                candidate
                and filename
                and not candidate.endswith(":")
                and not filename.lower().endswith(".json")
            ):
                return "repo_id", path
        return "path", path

    return "url", path


def load_reproduction_information(path: str) -> dict[str, Any]:
    source_kind, source = resolve_reproduction_source(path)

    if source_kind == "repo_id":
        # Fetch the reproduction information from a Hugging Face repository.
        from huggingface_hub import hf_hub_download

        try:
            local_path = hf_hub_download(source, "reproduce/reproduce.json")
        except Exception as error:
            raise FileNotFoundError(
                f"Could not fetch 'reproduce/reproduce.json' from the Hugging Face "
                f"repository '{source}': {error}"
            ) from error
        return json.loads(Path(local_path).read_text(encoding="utf-8"))

    if source_kind == "url":
        # The path is a URL on the web.

        # Obtain raw download URL.
        source = source.replace("/blob/", "/raw/")  # Hugging Face, GitHub
        source = source.replace("/src/branch/", "/raw/branch/")  # Codeberg

        json_str = urlopen(source).read().decode("utf-8")
    else:
        # The path is (assumed to be) a local file system path.
        json_str = Path(source).read_text(encoding="utf-8")

    return json.loads(json_str)


# ---------------------------------------------------------------------------
# Environment checking
# ---------------------------------------------------------------------------


class MismatchSeverity(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __rich__(self) -> str:
        match self:
            case MismatchSeverity.LOW:
                return "[green]low[/]"
            case MismatchSeverity.MEDIUM:
                return "[yellow]medium[/]"
            case MismatchSeverity.HIGH:
                return "[red]high[/]"
            case MismatchSeverity.CRITICAL:
                return "[bold red]critical[/]"
            case _:
                raise ValueError(f"unknown MismatchSeverity value: {self}")


def get_package_mismatch_severity(package_name: str) -> MismatchSeverity:
    if package_name in _DISTRIBUTION_NAME_CANDIDATES:
        return MismatchSeverity.CRITICAL
    elif package_name in [
        "torch",
        "transformers",
    ]:
        return MismatchSeverity.HIGH
    elif package_name in [
        "accelerate",
        "bitsandbytes",
        "kernels",
        "optuna",
        "peft",
        "tokenizers",
        "triton",
    ]:
        return MismatchSeverity.MEDIUM
    else:
        return MismatchSeverity.LOW


def format_version_information(version_information: dict[str, Any]) -> str:
    ver = version_information["version"]
    metadata = version_information["metadata"]

    if "type" in metadata:
        match metadata["type"]:
            case "pypi":
                return ver
            case "git":
                return f"{ver}-git+{metadata['url']}@{metadata['commit_hash']}"
            case "local":
                # Append a random number to ensure that two local installations
                # are always considered to be different versions.
                import random

                return f"{ver}-local-{random.randint(2**16, 2**17)}"
            case _:
                raise ValueError(
                    f"unknown metadata.type value in version information: {metadata['type']}"
                )
    elif not metadata:
        # The distribution origin could not be determined (source checkout,
        # bare egg-info, ...). Use a stable value so that comparing two
        # equally undeterminable environments doesn't report a phantom
        # mismatch; the random suffix stays only for real local installs.
        return ver
    else:
        import random

        return f"{ver}-unknown-{random.randint(2**16, 2**17)}"


def get_system_info_dict() -> dict[str, Any]:
    """Collects system information for reproduction metadata."""
    cpu_brand = None
    try:
        import cpuinfo

        cpu_brand = cpuinfo.get_cpu_info().get("brand_raw")
    except Exception:
        pass

    accelerators: dict[str, Any] = {"type": None, "api_name": None, "api_version": None,
                                    "driver_version": None, "devices": []}
    if torch.cuda.is_available():
        accelerators["type"] = "CUDA"
        accelerators["api_name"] = "CUDA"
        try:
            accelerators["api_version"] = torch.version.cuda
        except Exception:
            pass
        try:
            import pynvml  # noqa: F401  (nvidia-ml-py)

            accelerators["driver_version"] = _nvidia_driver_version()
        except Exception:
            pass
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            accelerators["devices"].append(
                {
                    "name": properties.name,
                    "vram_gb": round(properties.total_memory / (1024**3), 2),
                }
            )
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        accelerators["type"] = "MPS"

    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
        },
        "os": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "cpu": {"brand": cpu_brand},
        "accelerators": accelerators,
    }


def _nvidia_driver_version() -> Optional[str]:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            return pynvml.nvmlSystemGetDriverVersion()
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def get_requirements_dict() -> dict[str, str]:
    """Collects direct project dependencies as a {package: version} mapping."""
    packages = [
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "peft",
        "optuna",
        "datasets",
        "numpy",
        "huggingface-hub",
        "rich",
        "pydantic",
    ]
    requirements: dict[str, str] = {}
    for package in packages:
        try:
            requirements[package] = version(package)
        except PackageNotFoundError:
            continue
    return requirements


# Possible distribution (pip package) names, newest first. The distribution is
# "AnlordAbliterator"; older checkouts may still have it installed under the
# previous name "anlord".
_DISTRIBUTION_NAME_CANDIDATES = ("AnlordAbliterator", "anlordabliterator", "anlord")


def _distribution_name() -> Optional[str]:
    """Returns the name under which the package distribution is installed, if any."""
    for name in _DISTRIBUTION_NAME_CANDIDATES:
        try:
            version(name)
            return name
        except PackageNotFoundError:
            continue
    return None


def get_anlord_version_info() -> dict[str, Any]:
    """
    Returns version information about the installed package, including its
    origin (PyPI, Git, or local installation) where determinable.
    """
    dist_name = _distribution_name()
    try:
        ver = version(dist_name) if dist_name else "unknown"
    except PackageNotFoundError:
        ver = "unknown"

    metadata: dict[str, Any] = {}
    if dist_name:
        try:
            dist = distribution(dist_name)
            direct_url_text = dist.read_text("direct_url.json")
            if direct_url_text:
                direct_url = json.loads(direct_url_text)
                url = direct_url.get("url", "")
                if direct_url.get("vcs_info"):
                    metadata = {
                        "type": "git",
                        "url": url,
                        "commit_hash": direct_url["vcs_info"].get("commit_id", ""),
                    }
                else:
                    metadata = {"type": "local", "url": url}
        except Exception:
            pass

    return {
        "version": ver,
        "is_standard_pypi": metadata.get("type") is None,
        "metadata": metadata,
    }


def check_reproduction_environment(
    settings: NativeConfig,
    reproduction_information: dict[str, Any],
) -> bool:
    """
    Compares the local system and package versions against the ones recorded in
    the reproduction information, prints mismatch tables, and returns whether
    reproduction should proceed.

    Because the pipeline runs non-interactively, `ignore_mismatches` decides the
    outcome when mismatches exist: None proceeds with a warning, True proceeds
    silently, False aborts.
    """
    mismatch_severity: MismatchSeverity | None = None

    system_mismatches: list[tuple[str, Any, Any, MismatchSeverity]] = []
    package_mismatches: list[tuple[str, Any, Any, MismatchSeverity]] = []

    def verify(
        mismatch_list: list[tuple[str, Any, Any, MismatchSeverity]],
        name: str,
        this: Any,
        original: Any,
        severity: MismatchSeverity,
    ):
        nonlocal mismatch_severity
        if this != original:
            mismatch_list.append((name, this, original, severity))
            if mismatch_severity is None:
                mismatch_severity = severity
            else:
                mismatch_severity = max(severity, mismatch_severity)

    if "system" in reproduction_information and reproduction_information["system"]:
        system = reproduction_information["system"]

        this_system = get_system_info_dict()

        verify(
            system_mismatches,
            "Python version",
            this_system["python"]["version"],
            system["python"]["version"],
            MismatchSeverity.LOW,
        )

        verify(
            system_mismatches,
            "Operating system",
            this_system["os"]["platform"],
            system["os"]["platform"],
            MismatchSeverity.LOW,
        )

        verify(
            system_mismatches,
            "CPU",
            this_system["cpu"]["brand"],
            system["cpu"]["brand"],
            MismatchSeverity.LOW,
        )

        this_accelerators = this_system["accelerators"]
        original_accelerators = system["accelerators"]

        verify(
            system_mismatches,
            "Accelerator type",
            this_accelerators["type"],
            original_accelerators["type"],
            MismatchSeverity.HIGH,
        )

        if (
            this_accelerators["type"]
            and this_accelerators["type"] == original_accelerators["type"]
        ):
            verify(
                system_mismatches,
                this_accelerators["api_name"] or "API version",
                this_accelerators["api_version"],
                original_accelerators["api_version"],
                MismatchSeverity.MEDIUM,
            )
            verify(
                system_mismatches,
                "Driver version",
                this_accelerators["driver_version"],
                original_accelerators["driver_version"],
                MismatchSeverity.MEDIUM,
            )
            verify(
                system_mismatches,
                "Devices",
                "\n".join(d["name"] for d in this_accelerators["devices"]),
                "\n".join(d["name"] for d in original_accelerators["devices"]),
                MismatchSeverity.MEDIUM,
            )
    else:
        print(
            (
                "[yellow]The provided JSON file does not contain system information. "
                "Some system parameters can affect reproducibility, but due to the lack of "
                "system information, it is impossible to verify that those parameters match "
                "the original environment. Reproduction may or may not produce a "
                "byte-for-byte identical model.[/]"
            )
        )

    requirements = get_requirements_dict()
    # The tool itself is compared under its current distribution name, on both
    # sides, so a rename of the pip package doesn't produce a phantom mismatch.
    tool_key = _distribution_name() or "AnlordAbliterator"
    requirements[tool_key] = format_version_information(get_anlord_version_info())
    requirements["torch"] = torch.__version__

    original_environment = reproduction_information["environment"]
    original_requirements = dict(original_environment.get("requirements") or {})
    # The tool's version information is stored in the environment under the
    # stable "anlord" key; remove any stale same-key entry from the recorded
    # requirements so both sides align on the single tool key.
    original_requirements.pop("anlord", None)
    original_requirements.pop("AnlordAbliterator", None)
    original_requirements[tool_key] = format_version_information(
        original_environment["anlord"]
    )
    original_requirements["torch"] = original_environment["pytorch_version"]

    package_names = sorted(requirements.keys() | original_requirements.keys())

    for package_name in package_names:
        verify(
            package_mismatches,
            package_name,
            requirements.get(package_name),
            original_requirements.get(package_name),
            get_package_mismatch_severity(package_name),
        )

    if system_mismatches or package_mismatches:
        print()
        print(
            (
                "[yellow]Your local environment doesn't perfectly match the environment "
                "used to produce the original model. The following components differ:[/]"
            )
        )

    if system_mismatches:
        table = Table()
        table.add_column("Component")
        table.add_column("This system", overflow="fold")
        table.add_column("Original system", overflow="fold")
        table.add_column("Severity", width=8)

        for component, this, original, severity in system_mismatches:
            table.add_row(f"[bold]{component}[/]", str(this), str(original), severity)

        print()
        print("[bold]System Mismatches[/]")
        print(table)

    if package_mismatches:
        table = Table()
        table.add_column("Package")
        table.add_column("This system", overflow="fold")
        table.add_column("Original system", overflow="fold")
        table.add_column("Severity", width=8)

        for package, this, original, severity in package_mismatches:
            table.add_row(f"[bold]{package}[/]", str(this), str(original), severity)

        print()
        print("[bold]Package Mismatches[/]")
        print(table)

    if system_mismatches or package_mismatches:
        assert mismatch_severity is not None
        print()
        print(
            (
                f"There is a {mismatch_severity.__rich__()} chance "
                "that reproduction won't produce a byte-for-byte identical model. "
                "However, the resulting model will very likely still behave similarly "
                "to the original model."
            )
        )

        if getattr(settings, "ignore_mismatches", None) is False:
            return False
        return True

    # There are no mismatches at all, so there is nothing to confirm.
    return True


# ---------------------------------------------------------------------------
# Bundle generation
# ---------------------------------------------------------------------------


def collect_model_hashes(model_dir: str | Path) -> dict[str, str]:
    """Computes SHA-256 hashes for all weight files in an exported model directory."""
    hashes: dict[str, str] = {}
    directory = Path(model_dir)
    if not directory.is_dir():
        return hashes
    for file_path in sorted(directory.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in WEIGHT_EXTENSIONS:
            from .utils import get_file_sha256

            hashes[file_path.name] = get_file_sha256(file_path)
    return hashes


def verify_model_hashes(
    model_dir: str | Path,
    expected_hashes: dict[str, str],
) -> dict[str, str]:
    """
    Verifies that the weight files in `model_dir` match the original SHA-256
    hashes. Prints one line per file and returns a {filename: status} report
    with values "match", "mismatch", or "not_found".
    """
    print("Verifying hashes of weight files...")

    report: dict[str, str] = {}
    directory = Path(model_dir)

    for filename, original_sha256 in expected_hashes.items():
        file_path = directory / filename

        if file_path.exists():
            from .utils import get_file_sha256

            sha256 = get_file_sha256(file_path)

            if sha256.lower() == original_sha256.lower():
                print(f"[bold]{filename}:[/] [green]Hash matches[/]")
                report[filename] = "match"
            else:
                print(f"[bold]{filename}:[/] [yellow]Hash doesn't match[/]")
                report[filename] = "mismatch"
        else:
            print(f"[bold]{filename}:[/] [red]File not found[/]")
            report[filename] = "not_found"

    return report


def generate_requirements_txt() -> str:
    """Collects direct project dependencies as a formatted string."""
    requirements = get_requirements_dict()
    dist_name = _distribution_name()
    if dist_name:
        requirements[dist_name] = version(dist_name)
    lines = [f"{package}=={ver}" for package, ver in sorted(requirements.items())]
    return "\n".join(lines) + "\n"


def version_or_unknown() -> str:
    dist_name = _distribution_name()
    if dist_name:
        try:
            return version(dist_name)
        except PackageNotFoundError:
            pass
    return "unknown"


def generate_reproduction_json(
    settings: NativeConfig,
    anlord_settings_dict: dict[str, Any],
    trial: Any,
    metrics: dict[str, Any],
    model_hashes: dict[str, str],
    include_system_information: bool,
    native_config_dict: dict[str, Any] | None = None,
) -> str:
    """
    Generates the contents of a reproduce.json file for the reproduce/ folder.

    `settings` is the native config (for context), `anlord_settings_dict` the
    serialized top-level settings used for the run, `trial` the selected
    Optuna trial (FrozenTrial), `metrics` the legacy refusals/KL metrics, and
    `model_hashes` the SHA-256 hashes of the exported weight files.
    `native_config_dict` records native-level overrides (response prefix,
    normalization, batch size, ...) that the top-level settings do not carry.
    """
    version_info = get_anlord_version_info()

    data: dict[str, Any] = {
        "version": REPRODUCTION_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc)
        .replace(microsecond=0, tzinfo=None)
        .isoformat(),
        "system": None,  # Defined here to preserve insertion order.
        "environment": {
            "anlord": {
                "version": version_info["version"],
                "is_standard_pypi": version_info["is_standard_pypi"],
                "metadata": version_info["metadata"],
            },
            "pytorch_version": torch.__version__,
            "requirements": get_requirements_dict(),
        },
        "settings": anlord_settings_dict,
        "native_config": native_config_dict,
        "parameters": {
            "direction_index": trial.user_attrs["direction_index"],
            "abliteration_parameters": trial.user_attrs["parameters"],
        },
        "scores": trial.user_attrs.get("scores", []),
        "metrics": metrics,
        "hashes": model_hashes,
    }

    if include_system_information:
        data["system"] = get_system_info_dict()
    else:
        del data["system"]

    return json.dumps(data, indent=4)


def generate_sha256sums(hashes: dict[str, str]) -> str:
    """Generates GNU Coreutils compatible SHA256SUMS file content."""
    lines = []

    for filename, sha256 in sorted(hashes.items()):
        # Use '*' to indicate binary mode for model weights.
        lines.append(f"{sha256} *{filename}")

    return "\n".join(lines) + "\n"


def generate_reproduction_readme(
    settings: NativeConfig,
    checkpoint_filename: str,
    trial: Any,
    include_system_information: bool,
) -> str:
    """Generates the contents of a README.md for the reproduce/ folder."""

    def system_instructions(include: bool) -> str:
        if include:
            return (
                "1. Ensure your system matches the specifications in the **System** section below. "
                "Exact reproducibility is only guaranteed if all aspects of your system are identical "
                "to the one the model was originally generated on.\n"
            )
        return ""

    system_report = ""
    if include_system_information:
        system = get_system_info_dict()
        accelerators = system["accelerators"]
        if accelerators["type"] is None:
            accelerator_report = "**No GPU or other accelerator detected.**"
        else:
            devices = accelerators["devices"]
            total_vram = sum(d.get("vram_gb", 0) for d in devices)
            vram_suffix = f" ({total_vram:.2f} GB total VRAM)" if total_vram > 0 else ""
            accelerator_report = (
                f"- **{accelerators['type']}:** Detected {len(devices)} device(s){vram_suffix}"
            )

        heterogeneous_warning = ""
        if accelerators["type"] == "CUDA" and len(devices) > 1:
            device_names = {d["name"] for d in devices}
            if len(device_names) > 1:
                heterogeneous_warning = (
                    "\n> [!WARNING]\n"
                    "> **Heterogeneous GPUs**\n>\n"
                    "> This model was generated using multiple non-identical GPUs. "
                    "Reproducibility *cannot* be guaranteed in this environment.\n"
                )

        system_report = f"""{heterogeneous_warning}## System

- **Python:** {system["python"]["version"]} ({system["python"]["implementation"]}, {system["python"]["compiler"]})
- **Operating system:** {system["os"]["platform"]} ({system["os"]["machine"]})
- **CPU:** {system["cpu"]["brand"] or "Unknown"}

### Accelerators

{accelerator_report}

"""

    pytorch_version = torch.__version__
    pytorch_install_command = f"pip install torch=={pytorch_version}"
    if "+" in pytorch_version:
        suffix = pytorch_version.split("+")[1]
        if suffix:
            pytorch_install_command += (
                f" --index-url https://download.pytorch.org/whl/{suffix}"
            )

    trial_scores = trial.user_attrs.get("scores", [])
    score_lines = "\n".join(
        (
            f"- **{score['name']}:** {score['score']['md_display']}"
            f" (baseline: {score['baseline']['md_display']})"
        )
        for score in trial_scores
    )

    def format_hf_link(path: str, commit: str | None = None, is_dataset: bool = False) -> str:
        prefix = "datasets/" if is_dataset else ""
        base_url = f"https://huggingface.co/{prefix}{path}"
        link = f"[{path}]({base_url})"
        if commit:
            commit_url = f"{base_url}/commit/{commit}"
            link += f" (Commit: [`{commit[:7]}`]({commit_url}))"
        return link

    good = settings.good_prompts
    bad = settings.bad_prompts

    return f"""# Reproduction guide

This directory contains the necessary information and assets to reproduce the
results obtained during this Anlord Abliterator run.

## Models

- **Base model:** {format_hf_link(settings.model, settings.model_commit)}

## Datasets

- **Good prompts:** {format_hf_link(good.dataset, good.commit, is_dataset=True)}
- **Bad prompts:** {format_hf_link(bad.dataset, bad.commit, is_dataset=True)}

## Selected trial

- **Trial number:** {trial.user_attrs["index"]}
{score_lines}

{system_report}## Environment

- **Anlord Abliterator:** v{version_or_unknown()}
- **PyTorch:** {pytorch_version}
- **Other dependencies:** See [`requirements.txt`](requirements.txt).

## Contents of this directory

- [`requirements.txt`](requirements.txt): The exact versions of all Python packages.
- [`config.json`](config.json): The exact configuration used, including the RNG seed.
- [`{checkpoint_filename}`]({checkpoint_filename}): The Optuna study journal containing the history of all trials.
- [`SHA256SUMS`](SHA256SUMS): Cryptographic hashes for all weight files.
- [`reproduce.json`](reproduce.json): A machine-readable file containing all reproducibility information.

## How to reproduce

> [!TIP]
> You can automate this process, including all verification steps, by downloading the `reproduce.json` file and running
> `anlord --reproduce reproduce.json`.

{system_instructions(include_system_information)}1. Install the exact version of Anlord Abliterator indicated in the **Environment** section above, from its original source.
1. Install the packages listed in `requirements.txt`: `pip install -r requirements.txt`
1. Install the correct version of PyTorch: `{pytorch_install_command}`
1. Run the pipeline with the reproduction file: `anlord --reproduce reproduce.json`
1. Verify that the weight files have been exactly reproduced by comparing their SHA-256 hashes against those in `SHA256SUMS`:
   `sha256sum -c SHA256SUMS` (or look at the hashes online if you uploaded to Hugging Face)

> [!TIP]
> To use the included Optuna study journal `{checkpoint_filename}`, place it in the study checkpoint directory before running the pipeline.
>
> This allows you to export other models from the Pareto front, or to run additional trials without having to re-run the stored trials.
"""


def settings_to_reproduction_dict(settings) -> dict[str, Any]:
    """
    Serializes the top-level settings for the reproduction bundle, replacing
    local file-system paths (output/cache directories) so that no private data
    is recorded.
    """
    data = settings.to_dict()
    for key in ("output_dir", "cache_dir", "log_file"):
        if key in data:
            data[key] = None
    # The reproduction path itself is not part of the restored configuration.
    data["reproduce"] = None
    return data


def native_config_to_reproduction_dict(native_config: NativeConfig) -> dict[str, Any]:
    """
    Serializes the native-level config for reproduction, replacing local
    file-system paths (which are stripped, not kept) so that no private data is
    recorded.
    """
    data = native_config.to_dict()
    for key in ("study_checkpoint_dir", "cache_dir", "output_dir"):
        if key in data:
            data[key] = None
    return data


def create_reproduction_folder(
    path: Path,
    native_config: NativeConfig,
    anlord_settings,
    checkpoint_path: str | Path | None,
    trial: Any,
    model_hashes: dict[str, str],
    include_system_information: bool,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """
    Writes the reproduction bundle into `path / "reproduce"` and returns that
    directory.

    `metrics` may carry the run summary (refusal counts, KL, trial count); any
    missing entries are filled in from the trial's recorded user attributes so
    a reproduction run can report the original numbers.
    """
    reproduce_dir = Path(path) / "reproduce"
    reproduce_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_filename = Path(checkpoint_path).name if checkpoint_path else "study.jsonl"

    # The export strategy actually used is recorded in the bundle.
    export_strategy_value = native_config.export_strategy or "merge"
    export_strategy_used = (
        export_strategy_value.value
        if hasattr(export_strategy_value, "value")
        else str(export_strategy_value)
    )
    anlord_settings_dict = settings_to_reproduction_dict(anlord_settings)
    anlord_settings_dict["export_strategy"] = export_strategy_used

    # Record native-level configuration with local paths stripped.
    native_config_dict = native_config_to_reproduction_dict(native_config)

    trial_attrs = getattr(trial, "user_attrs", {}) or {}
    metrics = dict(metrics or {})
    metrics.setdefault("model", native_config.model)
    metrics.setdefault("export_strategy", export_strategy_used)
    metrics.setdefault("initial_refusals", trial_attrs.get("base_refusals"))
    metrics.setdefault("final_refusals", trial_attrs.get("refusals"))
    metrics.setdefault("total_prompts", trial_attrs.get("n_bad_prompts"))
    metrics.setdefault("kl_divergence", trial_attrs.get("kl_divergence"))
    metrics.setdefault("best_trial", trial_attrs.get("index"))

    (reproduce_dir / "requirements.txt").write_text(
        generate_requirements_txt(),
        encoding="utf-8",
    )

    (reproduce_dir / "config.json").write_text(
        json.dumps(anlord_settings_dict, indent=2),
        encoding="utf-8",
    )

    if model_hashes:
        (reproduce_dir / "SHA256SUMS").write_text(
            generate_sha256sums(model_hashes),
            encoding="utf-8",
        )

    (reproduce_dir / "reproduce.json").write_text(
        generate_reproduction_json(
            native_config,
            anlord_settings_dict,
            trial=trial,
            metrics=metrics,
            model_hashes=model_hashes,
            include_system_information=include_system_information,
            native_config_dict=native_config_dict,
        ),
        encoding="utf-8",
    )

    (reproduce_dir / "README.md").write_text(
        generate_reproduction_readme(
            native_config,
            checkpoint_filename,
            trial,
            include_system_information=include_system_information,
        ),
        encoding="utf-8",
    )

    # Copy the Optuna study journal.
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint_file = Path(checkpoint_path)
        (reproduce_dir / checkpoint_file.name).write_bytes(checkpoint_file.read_bytes())

    return reproduce_dir


# ---------------------------------------------------------------------------
# Settings restoration
# ---------------------------------------------------------------------------


def settings_from_reproduction(data: dict[str, Any]):
    """
    Rebuilds a top-level Settings object from the reproduction information,
    ignoring unknown keys (for bundles produced by other versions).
    """
    from ..config import Settings

    valid_fields = {f.name for f in dataclass_fields(Settings)}
    filtered = {key: value for key, value in data.items() if key in valid_fields}
    return Settings(**filtered)
