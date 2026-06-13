"""NVIDIA FLARE backend (optional extra: ``pip install -e ".[nvflare]"``).

Imported lazily by :mod:`fltest.frameworks`; if NVFlare isn't installed the import fails
quietly and the backend simply isn't registered.
"""

from fltest.frameworks.nvflare.adapter import NVFlareAdapter

__all__ = ["NVFlareAdapter"]
