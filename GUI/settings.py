"""Typed runtime configuration for the Gradio interface."""

from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


@dataclass(frozen=True, slots=True)
class DirectoryStatus:
    """A validated directory and the user-facing result of its validation."""

    path: Path | None
    message: str


class GUISettings(BaseSettings):
    """Environment-backed settings used by the Gradio application."""

    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    dataset_root: Path | None = None
    checkpoint_root: Path | None = None
    histogram_cache_root: Path | None = None
    max_images_per_class: int = Field(default=10, gt=0)
    model_device: str = "auto"

    @field_validator(
        "dataset_root",
        "checkpoint_root",
        "histogram_cache_root",
        mode="before",
    )
    @classmethod
    def blank_paths_as_none(cls, value: object) -> object:
        """Treat blank path variables like settings that were not supplied."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("model_device")
    @classmethod
    def normalize_model_device(cls, value: str) -> str:
        """Normalize the configured device while rejecting blank values."""
        normalized_value = value.strip().casefold()
        if not normalized_value:
            raise ValueError("MODEL_DEVICE must not be blank.")
        return normalized_value

    @property
    def resolved_model_device(self) -> str:
        """Resolve ``auto`` to an available PyTorch device."""
        if self.model_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.model_device


def get_existing_directory_status(
    configured_path: Path | None,
    environment_variable: str,
    display_name: str,
) -> DirectoryStatus:
    """Validate an existing absolute directory used by the GUI."""
    if configured_path is None:
        return DirectoryStatus(
            path=None,
            message=f"{environment_variable} is not set in {ENV_PATH}",
        )

    directory = configured_path.expanduser()
    if not directory.is_absolute():
        return DirectoryStatus(
            path=None,
            message=f"{environment_variable} must be an absolute path.",
        )
    if not directory.exists():
        return DirectoryStatus(
            path=None,
            message=f"{display_name} directory does not exist: {directory}",
        )
    if not directory.is_dir():
        return DirectoryStatus(
            path=None,
            message=f"{environment_variable} is not a directory: {directory}",
        )

    resolved_directory = directory.resolve()
    return DirectoryStatus(
        path=resolved_directory,
        message=f"{display_name} connected: {resolved_directory}",
    )


def get_cache_directory_status(
    configured_path: Path | None,
) -> DirectoryStatus:
    """Create and validate the configured absolute histogram-cache directory."""
    environment_variable = "HISTOGRAM_CACHE_ROOT"
    if configured_path is None:
        return DirectoryStatus(
            path=None,
            message=f"{environment_variable} is not set in {ENV_PATH}",
        )

    directory = configured_path.expanduser()
    if not directory.is_absolute():
        return DirectoryStatus(
            path=None,
            message=f"{environment_variable} must be an absolute path.",
        )

    try:
        if directory.exists() and not directory.is_dir():
            return DirectoryStatus(
                path=None,
                message=f"{environment_variable} is not a directory: {directory}",
            )
        directory.mkdir(parents=True, exist_ok=True)
        resolved_directory = directory.resolve()
        if not resolved_directory.is_dir():
            return DirectoryStatus(
                path=None,
                message=(
                    f"{environment_variable} is not a directory: "
                    f"{resolved_directory}"
                ),
            )
    except OSError as error:
        return DirectoryStatus(
            path=None,
            message=f"Histogram cache is unavailable: {error}",
        )

    return DirectoryStatus(
        path=resolved_directory,
        message=f"Histogram cache connected: {resolved_directory}",
    )
