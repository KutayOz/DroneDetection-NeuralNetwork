"""
Checksum utilities for model integrity verification.

Follows SRP: Only responsible for computing and verifying checksums.
"""

import hashlib
from pathlib import Path
from typing import Optional

from ..core.exceptions import ModelError


def compute_sha256(file_path: Path, chunk_size: int = 8192) -> str:
    """
    Compute SHA256 hash of a file.

    Args:
        file_path: Path to file
        chunk_size: Size of chunks to read (for large files)

    Returns:
        Hexadecimal hash string

    Raises:
        ModelError: If file cannot be read
    """
    if not file_path.exists():
        raise ModelError(f"File not found: {file_path}")

    sha256 = hashlib.sha256()

    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
    except IOError as e:
        raise ModelError(f"Failed to read file: {e}")

    return sha256.hexdigest()


def compute_md5(file_path: Path, chunk_size: int = 8192) -> str:
    """
    Compute MD5 hash of a file.

    Note: MD5 is faster but less secure than SHA256.
    Use only for quick integrity checks, not security.

    Args:
        file_path: Path to file
        chunk_size: Size of chunks to read

    Returns:
        Hexadecimal hash string
    """
    if not file_path.exists():
        raise ModelError(f"File not found: {file_path}")

    md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)

    return md5.hexdigest()


def verify_checksum(
    file_path: Path,
    expected_hash: str,
    algorithm: str = "sha256",
) -> bool:
    """
    Verify file checksum against expected value.

    Args:
        file_path: Path to file
        expected_hash: Expected hash value
        algorithm: Hash algorithm ("sha256" or "md5")

    Returns:
        True if checksum matches

    Raises:
        ValueError: If algorithm is not supported
    """
    if algorithm == "sha256":
        actual_hash = compute_sha256(file_path)
    elif algorithm == "md5":
        actual_hash = compute_md5(file_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return actual_hash.lower() == expected_hash.lower()


def compute_checksum_short(file_path: Path, length: int = 8) -> str:
    """
    Compute short checksum for display purposes.

    Args:
        file_path: Path to file
        length: Number of characters to return

    Returns:
        Truncated hash string
    """
    full_hash = compute_sha256(file_path)
    return full_hash[:length]


class ChecksumRegistry:
    """
    Registry for model checksums.

    Allows registering and verifying model checksums.
    """

    def __init__(self):
        self._checksums: dict[str, str] = {}

    def register(self, name: str, checksum: str) -> None:
        """
        Register a model checksum.

        Args:
            name: Model name
            checksum: Expected checksum
        """
        self._checksums[name] = checksum.lower()

    def get(self, name: str) -> Optional[str]:
        """
        Get registered checksum.

        Args:
            name: Model name

        Returns:
            Checksum or None if not registered
        """
        return self._checksums.get(name)

    def verify(self, name: str, file_path: Path) -> bool:
        """
        Verify model against registered checksum.

        Args:
            name: Model name
            file_path: Path to model file

        Returns:
            True if checksum matches or model not registered
        """
        expected = self._checksums.get(name)
        if expected is None:
            return True  # Not registered, assume valid

        return verify_checksum(file_path, expected)

    def verify_or_raise(self, name: str, file_path: Path) -> None:
        """
        Verify model and raise if mismatch.

        Args:
            name: Model name
            file_path: Path to model file

        Raises:
            ModelError: If checksum mismatch
        """
        expected = self._checksums.get(name)
        if expected is None:
            return  # Not registered, assume valid

        if not verify_checksum(file_path, expected):
            actual = compute_sha256(file_path)
            raise ModelError(
                f"Checksum mismatch for {name}",
                details={
                    "expected": expected,
                    "actual": actual,
                    "file": str(file_path),
                },
            )


# Global registry instance
_global_registry = ChecksumRegistry()


def get_checksum_registry() -> ChecksumRegistry:
    """Get global checksum registry."""
    return _global_registry
