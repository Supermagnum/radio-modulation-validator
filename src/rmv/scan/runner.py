"""Run validation for generated IQ and record results in the database."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rmv.classifier import ModulationClassifier
from rmv.scan.database import FindingsDB
from rmv.scan.discover import GRProject
from rmv.scan.iq_generator import GeneratedIQ
from rmv.validate import run_validate_file

logger = logging.getLogger(__name__)


@dataclass
class Issue:
    severity: str
    description: str
    block_name: str | None = None


@dataclass
class ProjectValidationRun:
    project: GRProject
    total_blocks: int
    passed: int
    soft_failed: int
    hard_failed: int
    skipped: int
    issues: list[Issue] = field(default_factory=list)
    run_at: str = ""
    block_results: list[dict[str, object]] = field(default_factory=list)


def run_validation(
    project: GRProject,
    generated_iqs: list[GeneratedIQ],
    db: FindingsDB,
    classifier: ModulationClassifier,
    *,
    threshold: float = 0.70,
) -> ProjectValidationRun:
    """Validate each GeneratedIQ and append rows to the database."""
    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    project_id = db.upsert_project(
        path=str(project.path),
        name=project.name,
        gr_version=project.gr_version,
        readme_path=str(project.readme_path) if project.readme_path else None,
        scan_status="ok",
    )

    passed = soft_failed = hard_failed = skipped = 0
    issues: list[Issue] = []
    block_results: list[dict[str, object]] = []

    for gen in generated_iqs:
        if gen.skipped or not gen.iq_path.is_file():
            skipped += 1
            if gen.skip_reason:
                issues.append(
                    Issue("info", gen.skip_reason, gen.block_name)
                )
                db.add_issue(
                    project_id=project_id,
                    block_id=None,
                    severity="info",
                    description=f"{gen.block_name}: {gen.skip_reason}",
                )
            block_results.append(
                {
                    "block": gen.block_name,
                    "skipped": True,
                    "reason": gen.skip_reason,
                }
            )
            continue

        block_id = db.upsert_block(
            project_id=project_id,
            block_name=gen.block_name,
            block_file=None,
            expected_family=gen.expected_family,
            expected_order=gen.expected_order,
            gr_version=project.gr_version,
        )
        db.supersede_open_issues_for_block(project_id=project_id, block_id=block_id)

        try:
            result = run_validate_file(
                gen.iq_path,
                classifier,
                threshold=threshold,
                output_dir=None,
            )
        except Exception as exc:
            logger.exception("Validation failed for %s", gen.iq_path)
            hard_failed += 1
            desc = f"{gen.block_name}: validation error: {exc}"
            issues.append(Issue("hard_fail", desc, gen.block_name))
            db.add_issue(
                project_id=project_id,
                block_id=block_id,
                severity="hard_fail",
                description=desc,
            )
            continue

        db.add_validation(
            block_id=block_id,
            iq_file=str(gen.iq_path),
            predicted_family=result.predicted_family,
            predicted_order=result.predicted_order,
            family_confidence=result.family_confidence,
            order_confidence=result.order_confidence,
            family_pass=result.family_pass,
            order_pass=result.order_pass,
            hard_fail=bool(result.hard_fail),
            notes=result.notes,
        )

        if result.hard_fail:
            hard_failed += 1
            desc = result.hard_fail_reason or "Hard fail"
            issues.append(Issue("hard_fail", f"{gen.block_name}: {desc}", gen.block_name))
            db.add_issue(
                project_id=project_id,
                block_id=block_id,
                severity="hard_fail",
                description=desc,
            )
        elif not result.order_pass or not result.family_pass:
            soft_failed += 1
            desc = (
                f"{gen.block_name}: expected {result.expected_family}/{result.expected_order}, "
                f"got {result.predicted_family}/{result.predicted_order}"
            )
            issues.append(Issue("soft_fail", desc, gen.block_name))
            db.add_issue(
                project_id=project_id,
                block_id=block_id,
                severity="soft_fail",
                description=desc,
            )
        else:
            passed += 1
            low_family = result.family_confidence < threshold
            low_order = result.order_confidence < threshold
            if low_family or low_order:
                warn = (
                    f"{gen.block_name}: correct label, low confidence "
                    f"(family={result.family_confidence:.2f}, "
                    f"order={result.order_confidence:.2f}, "
                    f"threshold={threshold:.2f})"
                )
                issues.append(Issue("warning", warn, gen.block_name))
                db.add_issue(
                    project_id=project_id,
                    block_id=block_id,
                    severity="warning",
                    description=warn,
                )

        if "Approximation" in gen.spec_note or "protocol" in gen.spec_note.lower():
            info = f"{gen.block_name}: {gen.spec_note}"
            issues.append(Issue("info", info, gen.block_name))
            db.add_issue(
                project_id=project_id,
                block_id=block_id,
                severity="info",
                description=info,
            )

        block_results.append(
            {
                "block": gen.block_name,
                "expected_family": result.expected_family,
                "expected_order": result.expected_order,
                "predicted_family": result.predicted_family,
                "predicted_order": result.predicted_order,
                "family_confidence": result.family_confidence,
                "order_confidence": result.order_confidence,
                "family_pass": result.family_pass,
                "order_pass": result.order_pass,
                "hard_fail": result.hard_fail,
            }
        )

    total = passed + soft_failed + hard_failed + skipped
    return ProjectValidationRun(
        project=project,
        total_blocks=total,
        passed=passed,
        soft_failed=soft_failed,
        hard_failed=hard_failed,
        skipped=skipped,
        issues=issues,
        run_at=run_at,
        block_results=block_results,
    )
