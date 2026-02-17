"""Dataset source abstractions for local and Hugging Face-hosted benchmark data."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

logger = logging.getLogger(__name__)


def _handle_hf_access_error(exc: RepositoryNotFoundError, repo_id: str) -> None:
    """Re-raise HF Hub access errors with actionable guidance.

    Args:
        exc: The original exception from `huggingface_hub`.
        repo_id: The HF dataset repository ID that was being accessed.

    Raises:
        GatedRepoError: With instructions to request access on the dataset page.
        RepositoryNotFoundError: With instructions to set `HF_TOKEN`.
    """
    dataset_url = f"https://huggingface.co/datasets/{repo_id}"
    if isinstance(exc, GatedRepoError):
        raise GatedRepoError(
            f"Access denied to gated dataset '{repo_id}'.\n"
            f"Your Hugging Face token was recognized, but you have not been "
            f"granted access to this dataset.\n"
            f"Visit {dataset_url} and request access, then retry.",
            response=exc.response,
        ) from None
    has_token = bool(os.environ.get("HF_TOKEN"))
    if has_token:
        raise RepositoryNotFoundError(
            f"Could not access dataset '{repo_id}' (HTTP {exc.response.status_code}).\n"
            f"Your HF_TOKEN is set but the request was rejected. "
            f"Check that your token is valid and that you have been granted "
            f"access at {dataset_url}.",
            response=exc.response,
        ) from None
    raise RepositoryNotFoundError(
        f"Could not access dataset '{repo_id}' (HTTP {exc.response.status_code}).\n"
        f"This is a gated dataset that requires authentication.\n"
        f"1. Set the HF_TOKEN environment variable to a Hugging Face access token.\n"
        f"   (Create one at https://huggingface.co/settings/tokens)\n"
        f"2. Visit {dataset_url} and request access.\n"
        f"3. Retry after access is granted.",
        response=exc.response,
    ) from None


class DatasetSource(Protocol):
    """Common interface for benchmark data sources."""

    def local_root(self) -> Path:
        """Materialize source locally and return benchmark root path."""
        ...


@dataclass(frozen=True)
class LocalDatasetSource:
    """Local benchmark directory source."""

    root: Path

    def local_root(self) -> Path:
        p = Path(self.root)
        if not p.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {p}")
        return p


@dataclass(frozen=True)
class HFDatasetSource:
    """Hugging Face dataset repository source.

    This downloads a repo snapshot to local cache and returns that root.
    """

    repo_id: str
    revision: str | None = None
    cache_dir: Path | None = None
    allow_patterns: list[str] | None = None

    def local_root(self) -> Path:
        try:
            local = snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                revision=self.revision,
                cache_dir=(str(self.cache_dir) if self.cache_dir is not None else None),
                allow_patterns=self.allow_patterns,
            )
        except RepositoryNotFoundError as e:
            _handle_hf_access_error(e, self.repo_id)
        return Path(local)


def download_file(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
) -> Path:
    """Download a single file from a HF dataset repo.

    Respects the ``HF_HOME`` environment variable for cache location.

    Args:
        repo_id: HF dataset repository ID.
        filename: Path of the file within the repo.
        revision: Git revision (branch, tag, or commit hash).

    Returns:
        Local path to the downloaded file.
    """
    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
        )
    except RepositoryNotFoundError as e:
        _handle_hf_access_error(e, repo_id)
    return Path(local)


SourceLike = DatasetSource | str | Path


def resolve_source_root(source: SourceLike) -> Path:
    """Resolve a source-like input into a local benchmark root path."""
    if isinstance(source, (str, Path)):
        root = LocalDatasetSource(Path(source)).local_root()
    else:
        root = source.local_root()
    logger.debug("Resolved source root: %s", root)
    return root
