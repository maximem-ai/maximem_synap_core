"""
Template loading and merging with MACA configuration.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from functools import lru_cache

from ..models import CompactionTemplate, CompactionConfig, CompactionStrategy
from ..interfaces import ITemplateLoader


def _ratio(value) -> float:
    """MACA's target_ratio, clamped to a share that leaves the ±5 band meaningful."""
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return 0.30
    return min(0.90, max(0.10, ratio))


class TemplateLoader(ITemplateLoader):
    """
    Loads compaction templates from YAML files and merges with MACA config.
    """

    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize template loader.

        Args:
            templates_dir: Path to templates directory. Defaults to yaml/ relative to this file.
        """
        if templates_dir is None:
            self.templates_dir = Path(__file__).parent / "yaml"
        else:
            self.templates_dir = Path(templates_dir)

        self._templates_cache: Dict[str, CompactionTemplate] = {}
        self._base_template: Optional[CompactionTemplate] = None

    @lru_cache(maxsize=20)
    def load_template(self, category: str) -> CompactionTemplate:
        """
        Load a template by category name.

        Args:
            category: Template category (e.g., "customer_support", "technical_debugging")

        Returns:
            CompactionTemplate loaded from YAML

        Raises:
            FileNotFoundError: If template file doesn't exist
            ValueError: If template is invalid
        """
        # Try exact match first
        template_path = self.templates_dir / f"{category}.yaml"

        if not template_path.exists():
            # Fall back to base template
            template_path = self.templates_dir / "base.yaml"
            if not template_path.exists():
                raise FileNotFoundError(
                    f"No template found for category '{category}' and no base template exists"
                )

        return self._load_template_file(template_path)

    def _load_template_file(self, path: Path) -> CompactionTemplate:
        """Load and parse a template YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Handle template inheritance
        if data.get('extends'):
            base = self.load_template(data['extends'])
            data = self._merge_template_data(base, data)

        return CompactionTemplate(
            category=data.get('category', 'unknown'),
            version=data.get('version', '1.0.0'),
            priority_extraction=data.get('priority_extraction', {}),
            summarization_guidelines=data.get('summarization_guidelines', {}),
            preservation_rules=data.get('preservation_rules', {}),
            special_handling=data.get('special_handling', []),
            compression_rates=data.get('compression_rates', {
                'high_density': 0.35,
                'medium_density': 0.25,
                'low_density': 0.10
            })
        )

    def _merge_template_data(
        self, base: CompactionTemplate, override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge override data with base template.
        Override values take precedence, but lists are merged.
        """
        merged = {
            'category': override.get('category', base.category),
            'version': override.get('version', base.version),
            'priority_extraction': self._merge_dict_of_lists(
                base.priority_extraction,
                override.get('priority_extraction', {})
            ),
            'summarization_guidelines': self._merge_dict_of_lists(
                base.summarization_guidelines,
                override.get('summarization_guidelines', {})
            ),
            'preservation_rules': self._merge_dict_of_lists(
                base.preservation_rules,
                override.get('preservation_rules', {})
            ),
            'special_handling': self._merge_lists(
                base.special_handling,
                override.get('special_handling', [])
            ),
            'compression_rates': {
                **base.compression_rates,
                **override.get('compression_rates', {})
            }
        }
        return merged

    def _merge_dict_of_lists(
        self, base: Dict[str, List], override: Dict[str, List]
    ) -> Dict[str, List]:
        """Merge dictionaries containing lists, extending lists rather than replacing."""
        merged = {k: list(v) for k, v in base.items()}
        for key, values in override.items():
            if key in merged:
                # Extend and deduplicate
                merged[key] = list(dict.fromkeys(merged[key] + values))
            else:
                merged[key] = values
        return merged

    def _merge_lists(self, base: List, override: List) -> List:
        """Merge lists, preserving order and removing duplicates."""
        return list(dict.fromkeys(base + override))

    def merge_with_maca(
        self,
        template: CompactionTemplate,
        maca_config: Dict[str, Any]
    ) -> CompactionConfig:
        """
        Merge base template with MACA customizations.

        Args:
            template: Base CompactionTemplate
            maca_config: MACA configuration dict (full config)

        Returns:
            Complete CompactionConfig ready for use
        """
        # Extract MACA compaction config
        compaction_section = maca_config.get('compaction_config', {})
        retrieval_section = maca_config.get('retrieval_config', {})
        agent_hints = maca_config.get('agent_hints', {}).get('compaction_agent', {})

        # Determine strategy
        strategy_str = compaction_section.get('strategy', 'adaptive')
        strategy = CompactionStrategy(strategy_str)

        # Build config
        config = CompactionConfig(
            strategy=strategy,

            # From MACA's thresholds, only what v3 reads: the size target
            # and the idle cadence. When to compact otherwise follows the
            # agent's model (budgets.py), not trigger_at_tokens /
            # trigger_at_messages; see CompactionConfig.message_trigger.
            target_ratio=_ratio(compaction_section.get('adaptive_thresholds', {}).get(
                'target_ratio', 0.30
            )),
            steering_enabled=bool(compaction_section.get('adaptive_thresholds', {}).get(
                'steering_enabled', False
            )),
            idle_minutes=compaction_section.get('adaptive_thresholds', {}).get(
                'idle_minutes', 5
            ),
            recompaction_idle_minutes=compaction_section.get('adaptive_thresholds', {}).get(
                'recompaction_idle_minutes', 5
            ),

            # Preservation priorities from MACA
            preservation_priorities=compaction_section.get('preservation_priorities', [
                "user_stated_facts",
                "explicit_preferences",
                "current_objective",
                "unresolved_issues",
                "key_decisions"
            ]),

            # Summarization candidates from MACA
            summarization_candidates=compaction_section.get('summarization_candidates', [
                "chitchat",
                "resolved_issues",
                "repeated_clarifications"
            ]),

            # Template
            template=template,

            # Core extraction categories (always included)
            core_extraction_categories=[
                "explicit_facts",
                "decisions",
                "constraints",
                "user_preferences",
                "open_questions",
                "conflicts_detected"
            ],

            # Template-driven categories
            template_extraction_categories=self._get_template_categories(template),

            # Custom categories from MACA
            custom_extraction_categories=self._get_maca_custom_categories(maca_config),

            # Validation threshold
            validation_threshold=0.8,

            # Instance context
            instance_id=maca_config.get('metadata', {}).get('instance_id'),
            instance_category=template.category
        )

        # Apply agent hints for preserve signals
        if 'preserve_signals' in agent_hints:
            config.preservation_priorities = list(dict.fromkeys(
                config.preservation_priorities + agent_hints['preserve_signals']
            ))

        return config

    def _get_template_categories(self, template: CompactionTemplate) -> List[str]:
        """Extract additional extraction categories from template."""
        # Categories mentioned in high priority extraction that aren't core
        core = {
            "explicit_facts", "decisions", "constraints",
            "user_preferences", "open_questions", "conflicts_detected"
        }

        template_categories = []
        for priority_list in template.priority_extraction.values():
            for category in priority_list:
                if category not in core:
                    template_categories.append(category)

        return list(dict.fromkeys(template_categories))

    def _get_maca_custom_categories(self, maca_config: Dict[str, Any]) -> List[str]:
        """Extract custom memory categories from MACA config."""
        custom = []
        memory_categories = maca_config.get('memory_categories', {})

        if 'custom_categories' in memory_categories:
            for cat in memory_categories['custom_categories']:
                if cat.get('enabled', True):
                    custom.append(cat.get('name', ''))

        return [c for c in custom if c]

    def get_available_templates(self) -> List[str]:
        """List all available template categories."""
        templates = []
        for path in self.templates_dir.glob("*.yaml"):
            templates.append(path.stem)
        return sorted(templates)

    def select_template_for_category(
        self, classification_result: Dict[str, Any]
    ) -> str:
        """
        Select the best template based on Pass 1 classification result.

        Args:
            classification_result: Output from Pass 1 classification

        Returns:
            Template category name
        """
        primary = classification_result.get('primary_category', '').lower().replace('/', '_')
        subcategory = classification_result.get('subcategory', '').lower()

        # Try subcategory first (most specific)
        if subcategory and (self.templates_dir / f"{subcategory}.yaml").exists():
            return subcategory

        # Try primary category
        category_mapping = {
            'technical_engineering': 'technical_debugging',
            'creative': 'creative_writing',
            'analytical': 'analytical_research',
            'learning_educational': 'learning_educational',
            'planning_coordination': 'planning_coordination',
            'transactional': 'transactional',
            'personal_support': 'customer_support',
            'mixed_hybrid': 'base'
        }

        template_name = category_mapping.get(primary, 'base')

        if (self.templates_dir / f"{template_name}.yaml").exists():
            return template_name

        return 'base'


# Singleton instance
_template_loader: Optional[TemplateLoader] = None


def get_template_loader() -> TemplateLoader:
    """Get or create the singleton TemplateLoader instance."""
    global _template_loader
    if _template_loader is None:
        _template_loader = TemplateLoader()
    return _template_loader
