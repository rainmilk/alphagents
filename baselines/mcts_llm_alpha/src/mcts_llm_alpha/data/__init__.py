#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data provider module stub.

When running with the main project's DataLoader (pandas mode), Qlib is not
required. This stub provides placeholder classes to satisfy imports.
"""

from typing import Dict, List, Optional, Any
import pandas as pd

# Flag: Qlib is NOT available in pandas mode
QLIB_AVAILABLE = False


class MarketDataManager:
    """Placeholder data manager for pandas-based evaluation mode."""

    def __init__(self, provider: Any = None):
        self.provider = provider

    def get_universe(self, start_date: str = None, end_date: str = None) -> List[str]:
        return []


def create_data_provider(use_qlib: bool = False, provider_uri: str = None):
    """Factory that returns a placeholder provider for pandas mode."""
    class PandasDataProvider:
        def initialize(self):
            pass

    return PandasDataProvider()
