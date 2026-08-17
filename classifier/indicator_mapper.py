"""Runtime indicator mapping from table.csv."""

from pathlib import Path
from typing import Any
import csv

from classifier.config import settings


class IndicatorMappingError(Exception):
    """Indicator mapping errors."""

    pass


class IndicatorMapper:
    """Loads and provides indicator mappings from table.csv."""

    def __init__(self, csv_path: str | None = None):
        """Initialize indicator mapper.

        Args:
            csv_path: Path to table.csv (defaults to project root)
        """
        if csv_path is None:
            csv_path = str(Path(__file__).parent.parent / "table.csv")

        self.csv_path = Path(csv_path)
        self._mappings: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load mappings from CSV file."""
        if not self.csv_path.exists():
            raise IndicatorMappingError(f"Indicator mapping file not found: {self.csv_path}")

        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self._mappings = list(reader)
        except Exception as e:
            raise IndicatorMappingError(f"Failed to load indicator mappings: {e}") from e

        if not self._mappings:
            raise IndicatorMappingError("No indicator mappings found in CSV")

        # Validate required columns
        required_columns = {"indicator", "dataset", "field", "derived_calculation"}
        if self._mappings:
            actual_columns = set(self._mappings[0].keys())
            missing = required_columns - actual_columns
            if missing:
                raise IndicatorMappingError(f"Missing required columns: {missing}")

    def get_all(self) -> list[dict[str, Any]]:
        """Get all indicator mappings."""
        return self._mappings.copy()

    def get_by_indicator(self, indicator: str) -> dict[str, Any] | None:
        """Get mapping for a specific indicator."""
        for mapping in self._mappings:
            if mapping["indicator"] == indicator:
                return mapping.copy()
        return None

    def get_indicators_list(self) -> list[str]:
        """Get list of all indicator names."""
        return [m["indicator"] for m in self._mappings]

    def get_datasets_for_indicator(self, indicator: str) -> list[str]:
        """Get required datasets for an indicator."""
        mapping = self.get_by_indicator(indicator)
        if mapping:
            return [mapping["dataset"]]
        return []

    def to_prompt_format(self) -> list[dict[str, Any]]:
        """Format mappings for prompt template."""
        return [
            {
                "indicator": m["indicator"],
                "dataset": m["dataset"],
                "field": m["field"],
                "derived_calculation": m["derived_calculation"],
            }
            for m in self._mappings
        ]


# Global instance
indicator_mapper = IndicatorMapper()