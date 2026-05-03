"""
CLI: pilot crawl of the OBS Bologna catalogue (Computer Engineering).

This is the operator-facing entry point for Step 2 of the data ingestion
plan. It owns:

  1. Playwright lifecycle (launch / context / page).
  2. Calling :class:`BolognaPilotScraper` to drive the navigation.
  3. Persisting the structured result into ``ScrapedPage`` rows so the
     existing chunking + embedding pipeline can ingest it without changes.
  4. Optionally dumping the raw structured JSON for inspection.

Why we don't reuse ``ingest_acibadem``:
    The generic crawler is link-graph driven; Bologna requires a *guided*
    walk (faculty -> department -> tabs). Mixing the two would erode the
    clean responsibilities established in Step 1.

Politeness:
    All delays are enforced inside ``BolognaPilotScraper`` (see the
    ``_polite_pause`` contract). The CLI just exposes the knobs.

Example
-------
::

    python manage.py ingest_obs_bologna \\
        --max-courses 50 \\
        --output-json data/scraped/obs_bologna_cse_pilot.json
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError, CommandParser

from chatbot.ingestion.content_cleaner import (
    clean_plain_text,
    content_hash,
)
from chatbot.ingestion.metadata_enricher import (
    enrich_bologna_course,
    enrich_bologna_info_page,
    enrich_bologna_program,
    lookup_bologna_program,
    merge_extracted_artifacts,
)
from chatbot.ingestion.obs_bologna import (
    INFO_KEY_TO_ASPX,
    BolognaPilotConfig,
    BolognaPilotResult,
    BolognaPilotScraper,
    render_course_markdown,
    render_info_page_markdown,
    render_program_overview_markdown,
    serialise_to_json,
)
from chatbot.ingestion.storage import upsert_page
from chatbot.ingestion.url_policy import normalize_url
from chatbot.models import ScrapedPage

logger = logging.getLogger(__name__)


# Same UA used by ResponsibleCrawler so OBS server logs see one consistent
# identity from the project. Specification-mandated transparency.
USER_AGENT = (
    "AcibademRagBot/1.0 (+responsible crawling; pilot=obs-bologna-cse; "
    "contact: your-team@example.com)"
)


def _override_entry_url(base: BolognaPilotConfig, url: str) -> BolognaPilotConfig:
    """Return a clone of ``base`` whose ``bologna_entry_url`` is forced to
    ``url``. Used by ``--explore-dom`` when the operator wants to dump a
    page outside the (cur_unit, cur_sunit) program (e.g. a sub-tab).
    """

    class _Patched(BolognaPilotConfig):
        @property
        def bologna_entry_url(self) -> str:  # type: ignore[override]
            return url

    return _Patched(
        base_url=base.base_url,
        cur_unit=base.cur_unit,
        cur_sunit=base.cur_sunit,
        faculty_name=base.faculty_name,
        department_name=base.department_name,
        max_courses=base.max_courses,
        polite_min_seconds=base.polite_min_seconds,
        polite_max_seconds=base.polite_max_seconds,
        nav_timeout_ms=base.nav_timeout_ms,
    )


class Command(BaseCommand):
    help = (
        "Pilot crawl of OBS Bologna for Mühendislik Fakültesi / Bilgisayar "
        "Mühendisliği. Captures program outcomes and per-course Description / "
        "Learning Outcomes / Weekly Plan tabs, then upserts structured "
        "Markdown into ScrapedPage rows."
    )

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--cur-unit",
            type=int,
            default=14,
            help="OIBS curUnit (faculty id). Default 14 = Mühendislik ve Doğa Bilimleri.",
        )
        parser.add_argument(
            "--cur-sunit",
            type=int,
            default=6246,
            help="OIBS curSunit (program id). Default 6246 = Bilgisayar Mühendisliği.",
        )
        parser.add_argument(
            "--faculty-name",
            default="Mühendislik ve Doğa Bilimleri Fakültesi",
            help="Display name used in ScrapedPage section + Markdown headers.",
        )
        parser.add_argument(
            "--department-name",
            default="Bilgisayar Mühendisliği",
            help="Display name used in ScrapedPage title + Markdown headers.",
        )
        parser.add_argument(
            "--bologna-url",
            default="",
            help=(
                "Only used by --explore-dom: override the URL to dump. "
                "When omitted, the index.aspx for (cur-unit, cur-sunit) is used."
            ),
        )
        parser.add_argument(
            "--max-courses",
            type=int,
            default=60,
            help="Cap on per-course drill-downs to keep the pilot bounded.",
        )
        parser.add_argument(
            "--min-delay",
            type=float,
            default=1.0,
            help="Minimum polite delay between interactions in seconds (spec: >=1.0).",
        )
        parser.add_argument(
            "--max-delay",
            type=float,
            default=2.0,
            help="Maximum polite delay between interactions in seconds (spec: <=2.0).",
        )
        parser.add_argument(
            "--nav-timeout-ms",
            type=int,
            default=30_000,
            help="Per-navigation timeout in milliseconds.",
        )
        parser.add_argument(
            "--output-json",
            default="",
            help="If given, writes the structured pilot result to this JSON path.",
        )
        parser.add_argument(
            "--no-persist",
            action="store_true",
            help="Skip writing ScrapedPage rows (useful for dry-run JSON-only runs).",
        )
        parser.add_argument(
            "--headed",
            action="store_true",
            help="Launch Chromium in headed mode for visual debugging.",
        )
        parser.add_argument(
            "--explore-dom",
            default="",
            help=(
                "Exploration mode: open the Bologna entry page, write every "
                "clickable element (text/href/onclick) to this JSON path, "
                "then exit without clicking anything. Use this to tune "
                "selector keywords against the real DOM."
            ),
        )
        parser.add_argument(
            "--screenshot",
            default="",
            help="In --explore-dom mode, also write a full-page PNG to this path.",
        )
        parser.add_argument("--log-level", default="INFO")

    # -- Entrypoint ------------------------------------------------------

    def handle(self, *args, **options):
        self._setup_logging(str(options["log_level"]))
        min_delay = float(options["min_delay"])
        max_delay = float(options["max_delay"])
        if min_delay < 1.0 or max_delay > 5.0 or min_delay > max_delay:
            raise CommandError(
                "Polite delays must satisfy 1.0 <= min <= max <= 5.0 "
                "(project specification rate-limit window)."
            )

        config = BolognaPilotConfig(
            cur_unit=int(options["cur_unit"]),
            cur_sunit=int(options["cur_sunit"]),
            faculty_name=str(options["faculty_name"]),
            department_name=str(options["department_name"]),
            max_courses=int(options["max_courses"]),
            polite_min_seconds=min_delay,
            polite_max_seconds=max_delay,
            nav_timeout_ms=int(options["nav_timeout_ms"]),
        )

        # ``--bologna-url`` is only meaningful in explore mode; when given,
        # it overrides the computed entry URL by injecting it directly into
        # the scraper via ``BolognaPilotConfig.bologna_entry_url`` (a
        # property derived from cur_unit / cur_sunit). For explore we patch
        # it monkey-style on the instance to keep the property elsewhere.
        explore_url_override = str(options.get("bologna_url") or "")

        explore_path = str(options.get("explore_dom") or "")
        if explore_path:
            screenshot_path = str(options.get("screenshot") or "")
            self._run_explore(
                config=config,
                headed=bool(options["headed"]),
                dump_path=explore_path,
                screenshot_path=screenshot_path or None,
                explore_url_override=explore_url_override or None,
            )
            return

        result = self._run_pilot(config=config, headed=bool(options["headed"]))

        self._maybe_dump_json(result, str(options["output_json"]))

        if options["no_persist"]:
            self.stdout.write(self.style.WARNING("--no-persist: skipping ScrapedPage upserts."))
        else:
            counts = self._persist_to_scraped_pages(result, config=config)
            total = sum(counts.values())
            self.stdout.write(
                self.style.SUCCESS(
                    f"Persisted {total} ScrapedPage rows "
                    f"(overview={counts['overview']} "
                    f"info_pages={counts['info_pages']} "
                    f"courses={counts['courses']})."
                )
            )

        self._print_summary(result)

    # -- Playwright wrapper ---------------------------------------------

    def _run_pilot(self, *, config: BolognaPilotConfig, headed: bool) -> BolognaPilotResult:
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:  # pragma: no cover - import error path
            raise CommandError(
                "Playwright is not installed. Run `pip install playwright && "
                "playwright install chromium` and retry."
            ) from exc

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not headed)
            try:
                context = browser.new_context(user_agent=USER_AGENT, locale="tr-TR")
                page = context.new_page()
                # The ``BolognaPilotScraper`` enforces the polite pauses
                # itself; the CLI only injects a default navigation timeout.
                page.set_default_timeout(config.nav_timeout_ms)
                scraper = BolognaPilotScraper(page=page, config=config)
                return scraper.run()
            finally:
                browser.close()

    def _run_explore(
        self,
        *,
        config: BolognaPilotConfig,
        headed: bool,
        dump_path: str,
        screenshot_path: str | None,
        explore_url_override: str | None,
    ) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:  # pragma: no cover - import error path
            raise CommandError(
                "Playwright is not installed. Run `pip install playwright && "
                "playwright install chromium` and retry."
            ) from exc

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not headed)
            try:
                context = browser.new_context(user_agent=USER_AGENT, locale="tr-TR")
                page = context.new_page()
                page.set_default_timeout(config.nav_timeout_ms)
                scraper = BolognaPilotScraper(page=page, config=config)
                if explore_url_override:
                    # The scraper's ``dump_dom_to_file`` reads
                    # ``self.config.bologna_entry_url``. Patch the property
                    # on this single instance for the override case.
                    scraper.config = _override_entry_url(config, explore_url_override)
                payload = scraper.dump_dom_to_file(dump_path, screenshot_path=screenshot_path)
            finally:
                browser.close()

        self.stdout.write(
            self.style.SUCCESS(
                f"Exploration done: clickables={payload['clickable_count']} "
                f"final_url={payload['final_url']!r} title={payload['title']!r}"
            )
        )
        self.stdout.write(self.style.NOTICE(f"DOM dump written to {dump_path}"))
        if screenshot_path:
            self.stdout.write(self.style.NOTICE(f"Screenshot written to {screenshot_path}"))

    # -- Output helpers --------------------------------------------------

    def _maybe_dump_json(self, result: BolognaPilotResult, path_str: str) -> None:
        if not path_str:
            return
        path = Path(path_str).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialise_to_json(result), encoding="utf-8")
        self.stdout.write(self.style.NOTICE(f"Structured JSON written to {path}"))

    def _persist_to_scraped_pages(
        self, result: BolognaPilotResult, *, config: BolognaPilotConfig
    ) -> dict[str, int]:
        """Materialise the pilot result as fine-grained ``ScrapedPage`` rows.

        Three families of rows are written:

          1. **Program overview** — one row holding the program-level
             outcomes / Markdown summary. Idempotency key:
             ``(canonical_url, "pilot:<slug>-program-overview")``.
          2. **Info pages** — one row per ``progXxx.aspx`` captured by
             the scraper (about, contact, goals, ...). Each row's
             metadata gets the matching ``content_type`` (e.g.
             ``bologna_about``), so a retriever query like
             "department X iletişim bilgisi" can land directly on the
             contact row instead of mixing it with Goals or Profile.
             Idempotency key: ``(canonical_url, "pilot:<slug>-info-<key>")``.
          3. **Courses** — one row per captured course, as before.
             Idempotency key: ``(canonical_url, "pilot:<slug>-course-<code>")``.

        The plain-text cleaner runs over each body before it is written
        so URLs / e-mails are stripped from the prose and stored on
        ``metadata.related_urls`` / ``metadata.contact_emails`` instead.
        """
        canonical_url = normalize_url(config.bologna_entry_url)
        section = result.program.department_name[:256]
        program_slug = self._program_slug(config)
        registry_entry = lookup_bologna_program(config.cur_unit, config.cur_sunit)
        program_language = registry_entry.language if registry_entry else "tr"
        program_level = registry_entry.level if registry_entry else "lisans"
        counts = {"overview": 0, "info_pages": 0, "courses": 0}

        # 1) Program overview row -------------------------------------
        overview_md = render_program_overview_markdown(result)
        if overview_md.strip():
            cleaned = clean_plain_text(overview_md, source_kind="obs")
            metadata = enrich_bologna_program(
                faculty=result.program.faculty_name,
                department=result.program.department_name,
                cur_unit=config.cur_unit,
                cur_sunit=config.cur_sunit,
                program_url=result.program.program_url or canonical_url,
                level=program_level,
                language=program_language,
            )
            metadata = merge_extracted_artifacts(
                metadata, urls=cleaned.urls, emails=cleaned.emails
            )
            store_result = upsert_page(
                url=canonical_url,
                url_variant=f"pilot:{program_slug}-program-overview"[:128],
                title=f"{result.program.department_name} — Program Genel Bakış"[:512],
                section=section,
                source_type=ScrapedPage.SOURCE_OBS,
                content=cleaned.text or overview_md,
                content_hash=content_hash(cleaned.text or overview_md),
                metadata=metadata,
            )
            logger.info(
                "BOLOGNA persist program_overview action=%s page_id=%s",
                store_result.action,
                store_result.page_id,
            )
            counts["overview"] += 1
        else:
            self.stdout.write(
                self.style.WARNING("Program overview produced empty Markdown — skipped.")
            )

        # 2) One row per captured info-page ---------------------------
        for info_key, info_body in (result.program.info_pages or {}).items():
            cleaned = clean_plain_text(info_body or "", source_kind="obs")
            if not cleaned.text.strip():
                logger.info(
                    "BOLOGNA persist info_page_skipped_empty key=%s", info_key
                )
                continue
            md = render_info_page_markdown(
                info_key, cleaned.text, program=result.program
            )
            aspx = INFO_KEY_TO_ASPX.get(info_key, "")
            page_url = config.page_url(aspx) if aspx else canonical_url
            metadata = enrich_bologna_info_page(
                info_key,
                faculty=result.program.faculty_name,
                department=result.program.department_name,
                cur_unit=config.cur_unit,
                cur_sunit=config.cur_sunit,
                page_url=page_url,
                level=program_level,
                language=program_language,
            )
            metadata = merge_extracted_artifacts(
                metadata, urls=cleaned.urls, emails=cleaned.emails
            )
            store_result = upsert_page(
                url=canonical_url,
                url_variant=f"pilot:{program_slug}-info-{info_key}"[:128],
                title=self._info_page_title(info_key, result.program.department_name),
                section=section,
                source_type=ScrapedPage.SOURCE_OBS,
                content=md,
                content_hash=content_hash(md),
                metadata=metadata,
            )
            logger.info(
                "BOLOGNA persist info_page=%s action=%s page_id=%s",
                info_key,
                store_result.action,
                store_result.page_id,
            )
            counts["info_pages"] += 1

        # 3) One row per captured course ------------------------------
        seen_codes: set[str] = set()
        for course in result.courses:
            code_slug = (course.code or "").lower().replace(" ", "")
            if not code_slug:
                continue
            if code_slug in seen_codes:
                # The Bologna pages occasionally list the same course
                # under fall + spring; keep the first capture, log the rest.
                logger.info("BOLOGNA persist duplicate_course_skipped code=%s", course.code)
                continue
            seen_codes.add(code_slug)

            md = render_course_markdown(course, program=result.program)
            if not md.strip():
                continue
            cleaned = clean_plain_text(md, source_kind="obs")
            metadata = enrich_bologna_course(
                code=course.code,
                name=course.name_tr,
                faculty=result.program.faculty_name,
                department=result.program.department_name,
                cur_unit=config.cur_unit,
                cur_sunit=config.cur_sunit,
                semester=course.semester,
                course_type=course.course_type,
                ects=course.ects,
                credit_theory=course.credit_theory,
                credit_practice=course.credit_practice,
                credit_lab=course.credit_lab,
                credit_total=course.credit_total,
                delivery_mode=course.language,
                detail_url=course.detail_url,
                postback_target=course.postback_target,
                level=program_level,
                language=program_language,
            )
            metadata = merge_extracted_artifacts(
                metadata, urls=cleaned.urls, emails=cleaned.emails
            )
            url_variant = f"pilot:{program_slug}-course-{code_slug}"[:128]
            store_result = upsert_page(
                url=canonical_url,
                url_variant=url_variant,
                title=f"{course.code} — {course.name_tr}"[:512],
                section=section,
                source_type=ScrapedPage.SOURCE_OBS,
                content=cleaned.text or md,
                content_hash=content_hash(cleaned.text or md),
                metadata=metadata,
            )
            logger.info(
                "BOLOGNA persist course=%s action=%s page_id=%s",
                course.code,
                store_result.action,
                store_result.page_id,
            )
            counts["courses"] += 1
        return counts

    @staticmethod
    def _program_slug(config: BolognaPilotConfig) -> str:
        """Return the registry slug for the program, falling back to a
        cur_sunit-based identifier so unmapped programs still get a
        unique-but-stable url_variant."""
        entry = lookup_bologna_program(config.cur_unit, config.cur_sunit)
        if entry is not None:
            return entry.program_slug
        return f"unit{config.cur_unit}-sunit{config.cur_sunit}"

    @staticmethod
    def _info_page_title(info_key: str, department_name: str) -> str:
        """Build the ScrapedPage.title for one info-page row.

        Pattern: ``"<Department> — <Section Label>"``. We reuse the
        Markdown render's display label so titles match the chunk
        headings the chatbot will eventually surface as citations.
        """
        from chatbot.ingestion.obs_bologna import INFO_KEY_TO_LABEL

        label = INFO_KEY_TO_LABEL.get(info_key, info_key.replace("_", " ").title())
        return f"{department_name} — {label}"[:512]

    # -- Misc ------------------------------------------------------------

    def _print_summary(self, result: BolognaPilotResult) -> None:
        program = result.program
        line = (
            f"Bologna pilot done: faculty={program.faculty_name!r} "
            f"department={program.department_name!r} "
            f"program_outcomes={len(program.program_outcomes)} "
            f"courses_captured={len(result.courses)} "
            f"pages_visited={result.pages_visited} "
            f"warnings={len(result.warnings)} errors={len(result.errors)}"
        )
        style = self.style.SUCCESS if not result.errors else self.style.WARNING
        self.stdout.write(style(line))
        for warning in result.warnings[:10]:
            self.stdout.write(self.style.WARNING(f"  warning: {warning}"))
        if len(result.warnings) > 10:
            self.stdout.write(
                self.style.WARNING(f"  ... ({len(result.warnings) - 10} more warnings)")
            )
        for error in result.errors[:10]:
            self.stdout.write(self.style.ERROR(f"  error: {error}"))

    def _setup_logging(self, level: str) -> None:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        )
