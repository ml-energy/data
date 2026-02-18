"""Dataset source abstractions for local and Hugging Face-hosted benchmark data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

logger = logging.getLogger(__name__)

_GATED_DATASET_MSG = (
    "Failed to download dataset '{repo_id}'. "
    "This is a gated dataset. Please ensure you have done the following:\n"
    "1. Visit https://huggingface.co/datasets/{repo_id} and request access.\n"
    "2. Set the HF_TOKEN environment variable to a Hugging Face access token.\n"
    "   (Create one at https://huggingface.co/settings/tokens)"
)


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
        except (GatedRepoError, RepositoryNotFoundError) as e:
            raise RuntimeError(_GATED_DATASET_MSG.format(repo_id=self.repo_id)) from e
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
    except (GatedRepoError, RepositoryNotFoundError) as e:
        raise RuntimeError(_GATED_DATASET_MSG.format(repo_id=repo_id)) from e
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
