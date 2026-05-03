"""
URL-direct scraper for the public OBS Bologna catalogue (Acibadem).

Step 2 pilot scope (per project lead):
    Faculty    -> Mühendislik ve Doğa Bilimleri Fakültesi   (curUnit=14)
    Department -> Bilgisayar Mühendisliği                   (curSunit=6246)

How the OIBS application is structured (verified by exploration):
    Each tab in the program detail page is a *separate* ASP.NET aspx page
    that takes the same ``curSunit`` query parameter, e.g.

        progAbout.aspx?lang=tr&curSunit=6246
        progLearnOutcomes.aspx?lang=tr&curSunit=6246
        progGraduationReq.aspx?lang=tr&curSunit=6246
        progCourses.aspx?lang=tr&curSunit=6246

    Knowing this, we don't need to click any tab — we ``page.goto()`` each
    URL directly. Every visit is followed by a 1–2 second polite pause, so
    the project specification rate-limit window is honoured by construction.

What is left for in-page extraction:
    Only ``progCourses.aspx`` requires DOM work, and even there it is a
    plain HTML table. We extract one row per course and follow each course
    link to grab Description / Learning Outcomes / Weekly Plan.

Why we kept ``dump_dom_to_file``:
    The exploration mode (``--explore-dom``) stays as a permanent debugging
    tool. When the OIBS template changes or we add a new department,
    re-running it on the new entry URL is still the fastest way to confirm
    the URL pattern.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses (also the public contract for Step 3 chunking work)
# ---------------------------------------------------------------------------


@dataclass
class BolognaCourse:
    """One course's normalised view, ready to be turned into semantic chunks."""

    code: str
    name_tr: str
    semester: int | None = None
    credit_theory: float | None = None
    credit_practice: float | None = None
    credit_lab: float | None = None
    credit_total: float | None = None
    ects: float | None = None
    course_type: str | None = None  # Zorunlu / Seçmeli
    language: str | None = None
    description_tr: str | None = None
    learning_outcomes: list[str] = field(default_factory=list)
    weekly_plan: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    assessment: list[str] = field(default_factory=list)
    detail_url: str | None = None
    # ASP.NET ``__doPostBack`` coordinates for the course-detail link.
    # Captured but not auto-followed in the pilot — drilling in requires
    # leaving the curriculum view and is deferred to a later step.
    postback_target: str | None = None
    postback_argument: str | None = None
    raw_text: str | None = None  # full text dump, kept for debugging only


@dataclass
class BolognaProgram:
    faculty_name: str
    department_name: str
    program_outcomes: list[str] = field(default_factory=list)
    program_description: str | None = None
    program_url: str | None = None
    # Free-form sections captured from the various ``progXxx.aspx`` pages.
    # Keys are stable identifiers (``about``, ``goals``, ``graduation`` …);
    # values are the cleaned page text. Step 3 will turn each of these into
    # one heading-aware Markdown section.
    info_pages: dict[str, str] = field(default_factory=dict)


@dataclass
class BolognaPilotResult:
    program: BolognaProgram
    courses: list[BolognaCourse] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    pages_visited: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "program": asdict(self.program),
            "courses": [asdict(c) for c in self.courses],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "pages_visited": self.pages_visited,
        }


@dataclass
class BolognaPilotConfig:
    """Knobs for a single pilot run.

    The pair ``(cur_unit, cur_sunit)`` identifies a program in OIBS:
        14 / 6246  -> Bilgisayar Mühendisliği (default pilot)
        14 / 6247  -> Biyomedikal Mühendisliği
        14 / 6248  -> Moleküler Biyoloji ve Genetik
    Other faculties (Tıp, Sağlık, Eczacılık …) live under their own
    ``cur_unit`` codes and can be scraped by the same code by changing
    these two numbers.
    """

    base_url: str = "https://obs.acibadem.edu.tr/oibs/bologna"
    cur_unit: int = 14
    cur_sunit: int = 6246
    faculty_name: str = "Mühendislik ve Doğa Bilimleri Fakültesi"
    department_name: str = "Bilgisayar Mühendisliği"

    max_courses: int = 60
    polite_min_seconds: float = 1.0
    polite_max_seconds: float = 2.0
    nav_timeout_ms: int = 30_000

    @property
    def bologna_entry_url(self) -> str:
        """Top-level entry used by ``--explore-dom`` and as a canonical URL
        for ScrapedPage rows. Pilot navigation does NOT start here."""
        return f"{self.base_url}/index.aspx?lang=tr&curOp=showPac&curUnit={self.cur_unit}&curSunit={self.cur_sunit}"

    def page_url(self, aspx_name: str) -> str:
        return f"{self.base_url}/{aspx_name}?lang=tr&curSunit={self.cur_sunit}"


# Sub-page list (label, aspx-name, free-text section key).
# The order is the same as the left menu in the program page, so the
# resulting Markdown reads top-to-bottom like a human-curated brochure.
PROGRAM_INFO_PAGES: tuple[tuple[str, str, str], ...] = (
    ("Eğitim Türü, Amaçlar ve Hedefler", "progGoalsObjectives.aspx", "goals_objectives"),
    ("Program Hakkında", "progAbout.aspx", "about"),
    ("Program Profili", "progProfile.aspx", "profile"),
    ("Program Yetkilileri", "progOfficials.aspx", "officials"),
    ("Alınacak Derece", "progDegree.aspx", "degree"),
    ("Kabul Koşulları", "progAdmissionReq.aspx", "admission"),
    ("Üst Kademeye Geçiş", "progAccessFurhterStudies.aspx", "further_studies"),
    ("Mezuniyet Koşulları", "progGraduationReq.aspx", "graduation"),
    ("Önceki Öğrenmenin Tanınması", "progRecogPriorLearning.aspx", "prior_learning"),
    ("Yeterlilik Koşulları ve Kuralları", "progQualifyReqReg.aspx", "qualification_rules"),
    ("İstihdam Olanakları", "progOccupationalProf.aspx", "occupation"),
    ("Akademik Personel", "progAcademicStaff.aspx", "academic_staff"),
    ("İletişim", "progContact.aspx", "contact"),
)


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------


_WS_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
_COURSE_CODE_RE = re.compile(r"^[A-ZÇĞİÖŞÜ]{2,5}\s?\d{2,4}[A-Z]?$", re.IGNORECASE)


def _normalize(text: str | None) -> str:
    if not text:
        return ""
    return _WS_RE.sub(" ", text).strip()


def _parse_first_number(text: str | None) -> float | None:
    if not text:
        return None
    match = _NUMBER_RE.search(text.replace(",", "."))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", "."))
    except ValueError:
        return None


def _split_outcome_bullets(text: str) -> list[str]:
    """Outcomes on Bologna pages come in many shapes (numbered, bulletted,
    sentence-per-line). Accept all three and trim noise."""
    if not text:
        return []
    raw_lines = re.split(r"[\r\n]+|(?<=[.;])\s+(?=[A-ZÇĞİÖŞÜ])", text)
    cleaned: list[str] = []
    for line in raw_lines:
        s = _normalize(line)
        if not s:
            continue
        s = re.sub(r"^(?:[-•▪●◦*]|\d+[\.\)])\s*", "", s).strip()
        if len(s) < 8:
            continue
        cleaned.append(s)
    seen: set[str] = set()
    deduped: list[str] = []
    for item in cleaned:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


# ---------------------------------------------------------------------------
# The scraper
# ---------------------------------------------------------------------------


class BolognaPilotScraper:
    """Drives a Playwright ``page`` through the Bologna pilot flow.

    Pilot strategy (URL-direct):
        1. ``page.goto(progAbout.aspx)`` → harvest each program meta-page.
        2. ``page.goto(progLearnOutcomes.aspx)`` → harvest program outcomes.
        3. ``page.goto(progCourses.aspx)`` → parse the curriculum table and
           collect (code, name, T/U/L, ECTS, language, type, link).
        4. For each course link, ``page.goto(course_url)`` → harvest text
           and split into Description / Learning Outcomes / Weekly Plan.
    """

    def __init__(self, page, config: BolognaPilotConfig | None = None) -> None:
        self.page = page
        self.config = config or BolognaPilotConfig()
        self.result = BolognaPilotResult(
            program=BolognaProgram(
                faculty_name=self.config.faculty_name,
                department_name=self.config.department_name,
                program_url=self.config.bologna_entry_url,
            )
        )

    # -- Politeness ------------------------------------------------------

    def _polite_pause(self) -> None:
        """Mandatory 1–2 second cool-down after every navigation.

        Project specification, non-negotiable: scraping must look like a
        human visitor to avoid IP bans / WAF challenges.
        """
        delay_ms = int(
            random.uniform(self.config.polite_min_seconds, self.config.polite_max_seconds) * 1000
        )
        try:
            self.page.wait_for_timeout(delay_ms)
        except Exception:
            pass

    def _goto(self, url: str) -> bool:
        """Navigate + networkidle + polite pause. Returns True on success.

        Every aspx page in OIBS is independent (no SPA routing), so the
        plain goto is enough — we do not need to walk through the iframe
        shell at index.aspx.
        """
        try:
            self.page.goto(url, wait_until="domcontentloaded", timeout=self.config.nav_timeout_ms)
        except Exception as exc:
            logger.warning("BOLOGNA goto_failed url=%s err=%s", url, exc)
            self.result.warnings.append(f"goto_failed:{url}")
            return False
        try:
            self.page.wait_for_load_state(
                "networkidle", timeout=min(self.config.nav_timeout_ms, 15_000)
            )
        except Exception:
            pass
        self._polite_pause()
        self.result.pages_visited += 1
        return True

    # -- DOM helpers -----------------------------------------------------

    def _list_clickables(self) -> list[dict[str, Any]]:
        """Snapshot of every potentially-clickable node + link on the page.

        Same shape used by ``dump_dom_to_file`` so the explore mode and the
        pilot share one DOM contract.
        """
        return self.page.evaluate(
            """() => {
                const sel = 'a, button, [role=tab], [role=button], input[type=submit], input[type=button], li, td, tr, span, div';
                const nodes = [...document.querySelectorAll(sel)];
                return nodes.map((n, i) => ({
                    i,
                    tag: n.tagName,
                    href: n.getAttribute('href') || '',
                    onclick: n.getAttribute('onclick') || '',
                    text: (n.innerText || n.textContent || '').trim().slice(0, 240)
                })).filter(r => r.text);
            }"""
        )

    def _collect_full_text(self) -> str:
        """Body inner_text of the main page + every same-origin frame."""
        parts: list[str] = []
        seen: set[str] = set()
        for frame in self.page.frames:
            try:
                loc = frame.locator("body")
                if loc.count() == 0:
                    continue
                chunk = (loc.inner_text(timeout=10_000) or "").strip()
                if chunk and chunk not in seen:
                    seen.add(chunk)
                    parts.append(chunk)
            except Exception:
                continue
        return "\n\n".join(parts)

    # -- Exploration / debugging ----------------------------------------

    def dump_dom_to_file(
        self,
        dump_path: str,
        *,
        screenshot_path: str | None = None,
    ) -> dict[str, Any]:
        """No-click exploration: write every clickable to JSON, optional PNG.

        Used by ``--explore-dom`` to verify the URL pattern of new programs
        without firing any postbacks.
        """
        cfg = self.config
        logger.info("BOLOGNA explore_start url=%s", cfg.bologna_entry_url)
        self._goto(cfg.bologna_entry_url)

        try:
            page_title = (self.page.title() or "").strip()
        except Exception:
            page_title = ""
        final_url = self.page.url

        frame_info: list[dict[str, str]] = []
        for frame in self.page.frames:
            try:
                frame_info.append({"url": frame.url, "name": frame.name or ""})
            except Exception:
                continue

        try:
            clickables = self._list_clickables()
        except Exception as exc:
            logger.warning("BOLOGNA explore_list_failed err=%s", exc)
            clickables = []

        payload: dict[str, Any] = {
            "entry_url": cfg.bologna_entry_url,
            "final_url": final_url,
            "title": page_title,
            "frames": frame_info,
            "clickable_count": len(clickables),
            "clickables": clickables,
        }

        out_path = Path(dump_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("BOLOGNA explore_dump path=%s clickables=%d", out_path, len(clickables))

        if screenshot_path:
            shot_path = Path(screenshot_path).expanduser()
            shot_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self.page.screenshot(path=str(shot_path), full_page=True)
                logger.info("BOLOGNA explore_screenshot path=%s", shot_path)
            except Exception as exc:
                logger.warning("BOLOGNA explore_screenshot_failed err=%s", exc)

        return payload

    # -- Public driver ---------------------------------------------------

    def run(self) -> BolognaPilotResult:
        cfg = self.config
        logger.info(
            "BOLOGNA pilot_start cur_unit=%s cur_sunit=%s",
            cfg.cur_unit,
            cfg.cur_sunit,
        )
        try:
            self._capture_program_outcomes()
            self._capture_info_pages()
            self._capture_courses()
        except Exception as exc:
            logger.exception("BOLOGNA pilot_fatal err=%s", exc)
            self.result.errors.append(f"fatal:{exc!r}")
        return self.result

    # -- Program-level captures -----------------------------------------

    def _capture_program_outcomes(self) -> None:
        """Visit progLearnOutcomes.aspx and split into bullet list."""
        url = self.config.page_url("progLearnOutcomes.aspx")
        if not self._goto(url):
            self.result.errors.append("program_outcomes_goto_failed")
            return
        text = self._collect_full_text()
        self.result.program.program_description = text[:6000] if text else None
        outcomes = _split_outcome_bullets(text)
        outcomes = [o for o in outcomes if self._is_real_outcome(o)]
        if outcomes:
            self.result.program.program_outcomes = outcomes
        else:
            self.result.warnings.append("program_outcomes_empty_after_filter")

    @staticmethod
    def _is_real_outcome(text: str) -> bool:
        """Filter the OIBS chrome that ``_split_outcome_bullets`` cannot
        distinguish from real bullets (page heading, trailing self-URL,
        single-word menu items)."""
        if not text or len(text) < 20:
            return False
        if text.startswith(("http://", "https://")):
            return False
        lowered = text.lower()
        # Headings that always sit above the actual list.
        bad_prefixes = (
            "program yeterlikleri",
            "program çıktıları",
            "program ciktilari",
            "program öğrenme çıktıları",
            "program ogrenme ciktilari",
            "no program öğrenme",
            "no program ogrenme",
        )
        if any(lowered.startswith(p) for p in bad_prefixes):
            return False
        # Same menu chrome we strip from info pages.
        menu_noise = {
            "bilgi paketi", "akademik birimler", "kurumsal bilgiler",
            "bologna süreci", "erasmus+ beyannamesi",
            "erasmus beyannamesi (i̇ngilizce)", "erasmus beyannamesi (ingilizce)",
            "ön lisans", "lisans", "yüksek lisans", "doktora",
            "öğrenciler i̇çin genel bilgiler", "öğrenciler için genel bilgiler",
            "şehir hakkında", "kampüs", "yemek", "sağlık hizmetleri",
            "spor ve sosyal yaşam", "öğrenci kulüpleri", "konaklama",
            "engelli öğrenci hizmetleri", "iletişim", "akts kataloğu",
            "bologna komisyonu", "üniversite hakkında", "yönetim",
        }
        return lowered not in menu_noise

    def _capture_info_pages(self) -> None:
        """Visit each progXxx.aspx and stash the cleaned text under a
        stable key (later turned into Markdown sections)."""
        for label, aspx_name, key in PROGRAM_INFO_PAGES:
            url = self.config.page_url(aspx_name)
            if not self._goto(url):
                self.result.warnings.append(f"info_goto_failed:{key}")
                continue
            text = self._collect_full_text()
            cleaned = self._strip_menu_chrome(text)
            if cleaned:
                self.result.program.info_pages[key] = cleaned
            else:
                self.result.warnings.append(f"info_empty:{key}")
            logger.info(
                "BOLOGNA info_page key=%s url=%s len=%d", key, url, len(cleaned or "")
            )

    @staticmethod
    def _strip_menu_chrome(text: str) -> str:
        """Drop the OIBS left-menu / header lines that appear on every page.

        We do not need a perfect strip — chunking will further trim — but
        removing the obvious chrome cuts retrieval noise dramatically.
        """
        if not text:
            return ""
        chrome_markers = (
            "Bilgi Paketi",
            "Bologna Komisyonu",
            "Erasmus Beyannamesi",
            "AKTS Kataloğu",
            "www.prolizyazilim.com",
        )
        # Remove all lines that exactly match menu chrome.
        lines = [line for line in text.splitlines()]
        kept: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped in chrome_markers:
                continue
            kept.append(stripped)
        return "\n".join(kept).strip()

    # -- Curriculum + courses -------------------------------------------

    def _capture_courses(self) -> None:
        url = self.config.page_url("progCourses.aspx")
        if not self._goto(url):
            self.result.errors.append("courses_goto_failed")
            return

        # Always dump the courses page so we can inspect the DOM if parsing
        # fails — saved next to the JSON output for easy review.
        self._dump_courses_debug()

        # Plain-text parser is preferred: progCourses.aspx is rendered as a
        # tab-separated table (verified by the debug dump) and the text
        # form contains every field we care about (code, name, T+U+L, type,
        # ECTS, semester). The anchor parser is only used to enrich rows
        # with their __doPostBack targets so a future step can drill in.
        rows = self._extract_courses_by_plain_text()
        if not rows:
            logger.info("BOLOGNA plain_text_parse_empty, trying table-row extractor")
            rows = self._extract_course_rows()
        if not rows:
            logger.info("BOLOGNA table_row_parse_empty, trying anchor-based extractor")
            rows = self._extract_courses_by_anchors()
        else:
            self._merge_postback_targets(rows)

        logger.info("BOLOGNA curriculum_rows=%d", len(rows))
        if not rows:
            self.result.warnings.append("courses_table_empty")
            return

        # Each row already carries its detail URL or postback coordinates.
        # Drill into the first ``max_courses`` only.
        for stub in rows[: self.config.max_courses]:
            course = BolognaCourse(
                code=stub["code"],
                name_tr=stub["name"],
                semester=stub.get("semester"),
                credit_theory=stub.get("credit_theory"),
                credit_practice=stub.get("credit_practice"),
                credit_lab=stub.get("credit_lab"),
                credit_total=stub.get("credit_total"),
                ects=stub.get("ects"),
                course_type=stub.get("type"),
                language=stub.get("language"),
                detail_url=stub.get("detail_url"),
                postback_target=stub.get("postback_target"),
                postback_argument=stub.get("postback_argument"),
            )
            # Only follow real HTTP detail URLs in the pilot. Courses
            # whose only path-in is a __doPostBack still get persisted
            # with their meta-data (kod / ad / T+U+L / AKTS / yarıyıl /
            # tip), which is already the high-signal RAG payload.
            if course.detail_url and not course.detail_url.startswith("javascript:"):
                try:
                    self._enrich_course_detail(course)
                except Exception as exc:
                    logger.warning(
                        "BOLOGNA course_detail_failed code=%s err=%s", course.code, exc
                    )
                    self.result.warnings.append(f"course_detail_failed:{course.code}")
            self.result.courses.append(course)

    def _dump_courses_debug(self) -> None:
        """Write the rendered HTML + plain text of progCourses.aspx to disk
        so a missed-parse can be debugged without re-running --explore-dom.
        Files live under ``data/scraped/`` next to the JSON pilot output.
        """
        debug_dir = Path("data/scraped")
        debug_dir.mkdir(parents=True, exist_ok=True)
        try:
            html = self.page.content()
            (debug_dir / "bologna_progcourses_debug.html").write_text(html or "", encoding="utf-8")
        except Exception as exc:
            logger.warning("BOLOGNA dump_courses_html_failed err=%s", exc)
        try:
            text = self._collect_full_text()
            (debug_dir / "bologna_progcourses_debug.txt").write_text(text or "", encoding="utf-8")
        except Exception as exc:
            logger.warning("BOLOGNA dump_courses_text_failed err=%s", exc)

    def _extract_courses_by_anchors(self) -> list[dict[str, Any]]:
        """Fallback parser: scan every ``<a>`` link on the page; treat
        anchors that look like course detail URLs as one course each.

        OIBS course detail URLs typically encode ``courseNo`` or include
        ``Course`` / ``Ders`` in the file name. We accept anything that
        leads back into the same OIBS folder and has a course-code-shaped
        text label.
        """
        anchors = self.page.evaluate(
            """() => {
                return [...document.querySelectorAll('a[href]')].map(a => ({
                    href: a.getAttribute('href') || '',
                    text: (a.innerText || a.textContent || '').trim()
                }));
            }"""
        )
        result: list[dict[str, Any]] = []
        seen_codes: set[str] = set()
        for a in anchors:
            text = _normalize(a.get("text") or "")
            href = (a.get("href") or "").strip()
            if not text or not href:
                continue
            # Pull a course code out of the link text. Some OIBS skins use
            # "CSE 101 — Programming I", others "CSE101: Programming I".
            code_match = re.match(
                r"^([A-ZÇĞİÖŞÜ]{2,5})\s*(\d{2,4}[A-Z]?)\b", text, re.IGNORECASE
            )
            if not code_match:
                continue
            href_lower = href.lower()
            looks_like_course = (
                "course" in href_lower
                or "ders" in href_lower
                or "syllabus" in href_lower
                or "curops" in href_lower
                or "showpac" in href_lower
                or "courseno" in href_lower
            )
            if not looks_like_course:
                continue
            code = (code_match.group(1) + code_match.group(2)).upper()
            if code in seen_codes:
                continue
            seen_codes.add(code)
            # Strip the code prefix from the title to keep just the name.
            name = text[code_match.end() :].lstrip(" -—:.\t").strip() or text
            result.append(
                {
                    "code": code,
                    "name": name,
                    "detail_url": self._absolutise(href),
                }
            )
        return result

    def _extract_courses_by_plain_text(self) -> list[dict[str, Any]]:
        """Parser for the OIBS curriculum view (the live structure verified
        by ``data/scraped/bologna_progcourses_debug.txt``).

        The page is rendered as one tab-separated block per semester, e.g.::

            1.Yarıyıl Ders Planı
            Ders Kodu  Ders Adı                T+U+L  Zorunlu/Seçmeli  AKTS  ...  Öğretim Şekli
            CHE 101    Genel Kimya             3+0+0  Zorunlu          6     ...  Yüz Yüze
            CSE 101    Programlamaya Giriş     2+2+0  Zorunlu          6     ...  Yüz Yüze

        Strategy:
          * Detect the "N.Yarıyıl Ders Planı" headers (number first, then word).
          * Skip the table header row ("Ders Kodu …").
          * Skip the "Toplam AKTS" sub-totals.
          * Split each remaining line by tabs (or 2+ spaces) and read fixed
            columns: code, name, T+U+L, type, ECTS, group, delivery mode.
        """
        text = self._collect_full_text()
        if not text:
            return []
        result: list[dict[str, Any]] = []
        seen_codes: set[str] = set()
        current_semester: int | None = None

        sem_header_re = re.compile(
            r"^(\d+)\s*[.\)]?\s*(?:yarıyıl|yariyil|semester|dönem|donem)\b",
            re.IGNORECASE,
        )
        header_row_re = re.compile(r"^\s*ders\s*kodu\b", re.IGNORECASE)
        total_row_re = re.compile(r"^\s*toplam\s*akts\b", re.IGNORECASE)
        code_re = re.compile(r"^([A-ZÇĞİÖŞÜ]{2,5})\s*(\d{2,4}[A-Z]?)$")
        tul_re = re.compile(r"^\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\s*$")

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            sem_match = sem_header_re.match(line)
            if sem_match:
                try:
                    current_semester = int(sem_match.group(1))
                except ValueError:
                    current_semester = None
                continue

            if header_row_re.match(line) or total_row_re.match(line):
                continue

            cells = [c.strip() for c in re.split(r"\t+|\s{2,}", line) if c.strip()]
            if len(cells) < 2:
                continue

            code_match = code_re.match(cells[0])
            if not code_match:
                continue
            code = (code_match.group(1) + code_match.group(2)).upper()
            if code in seen_codes:
                continue
            seen_codes.add(code)

            stub: dict[str, Any] = {
                "code": code,
                "name": cells[1] if len(cells) > 1 else code,
                "semester": current_semester,
            }

            for cell in cells[2:]:
                tul_match = tul_re.match(cell)
                if tul_match:
                    stub["credit_theory"] = float(tul_match.group(1))
                    stub["credit_practice"] = float(tul_match.group(2))
                    stub["credit_lab"] = float(tul_match.group(3))
                    continue
                lc = cell.lower()
                if "zorunlu" in lc or "compulsory" in lc:
                    stub["type"] = "Zorunlu"
                    continue
                if "seçmeli" in lc or "secmeli" in lc or "elective" in lc:
                    stub["type"] = "Seçmeli"
                    continue
                if lc in {"yüz yüze", "yuz yuze", "face to face"}:
                    stub["language"] = "Yüz Yüze"
                    continue
                if lc in {"uzaktan", "distance"}:
                    stub["language"] = "Uzaktan"
                    continue
                if "ects" not in stub:
                    num = _parse_first_number(cell)
                    if num is not None and num <= 60:  # ECTS sanity bound
                        stub["ects"] = num
            result.append(stub)
        return result

    def _merge_postback_targets(self, rows: list[dict[str, Any]]) -> None:
        """Enrich plain-text parser stubs with the ``__doPostBack`` target
        attached to each course-code anchor.

        We can't follow ``javascript:__doPostBack(...)`` URLs with
        ``page.goto`` (Playwright treats them as relative paths and the
        server returns a 404). What we *can* do is store the postback
        target string on each row so a later step can fire it via
        ``page.evaluate('__doPostBack(...)')`` if drill-down becomes
        required.
        """
        try:
            anchors = self.page.evaluate(
                """() => {
                    return [...document.querySelectorAll('a[href]')].map(a => ({
                        href: a.getAttribute('href') || '',
                        text: (a.innerText || a.textContent || '').trim()
                    }));
                }"""
            )
        except Exception as exc:
            logger.warning("BOLOGNA postback_anchor_scan_failed err=%s", exc)
            return
        postback_re = re.compile(
            r"__doPostBack\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]*)['\"]"
        )
        code_re = re.compile(
            r"^([A-ZÇĞİÖŞÜ]{2,5})\s*(\d{2,4}[A-Z]?)\b", re.IGNORECASE
        )
        target_by_code: dict[str, tuple[str, str]] = {}
        for a in anchors:
            href = (a.get("href") or "").strip()
            text = _normalize(a.get("text") or "")
            if not text or "__doPostBack" not in href:
                continue
            cm = code_re.match(text)
            if not cm:
                continue
            code = (cm.group(1) + cm.group(2)).upper()
            if code in target_by_code:
                continue
            pm = postback_re.search(href)
            if not pm:
                continue
            target_by_code[code] = (pm.group(1), pm.group(2))
        if not target_by_code:
            return
        for stub in rows:
            target = target_by_code.get(stub.get("code", ""))
            if target:
                # Stored as a stable, parseable string. The actual
                # detail page is reached via __doPostBack, not goto().
                stub["postback_target"] = target[0]
                stub["postback_argument"] = target[1]

    def _extract_course_rows(self) -> list[dict[str, Any]]:
        """Parse the curriculum table — every ``<tr>`` whose first cell
        looks like a course code becomes a stub.

        We also pull out ``href`` attributes from each cell so the pilot
        can follow per-course links without a second DOM scan."""
        rows = self.page.evaluate(
            """() => {
                const out = [];
                document.querySelectorAll('table tr').forEach(tr => {
                    const cells = [...tr.querySelectorAll('th,td')].map(td => ({
                        text: (td.innerText || '').trim(),
                        link: (td.querySelector('a[href]') || {}).getAttribute ?
                              (td.querySelector('a[href]').getAttribute('href') || '') : ''
                    }));
                    if (cells.length >= 2) out.push(cells);
                });
                return out;
            }"""
        )

        result: list[dict[str, Any]] = []
        current_semester: int | None = None
        for cells in rows:
            if not cells:
                continue
            first_text = _normalize(cells[0]["text"])
            if not first_text:
                continue

            # Some OIBS pages use a "Yarıyıl X" row as a section break.
            sem_match = re.match(r"(?:yariyil|yarıyıl|semester)\s*(\d+)", first_text, re.I)
            if sem_match:
                try:
                    current_semester = int(sem_match.group(1))
                except ValueError:
                    pass
                continue

            if not _COURSE_CODE_RE.match(first_text):
                continue

            stub: dict[str, Any] = {
                "code": first_text.replace(" ", ""),
                "name": _normalize(cells[1]["text"]) if len(cells) > 1 else "",
                "semester": current_semester,
            }
            # Course detail link is most often on the name cell, but
            # occasionally on the code cell — try both.
            for c in cells[:2]:
                href = (c.get("link") or "").strip()
                if href:
                    stub["detail_url"] = self._absolutise(href)
                    break

            tail = [_normalize(c["text"]) for c in cells[2:]]
            numeric_tail = [_parse_first_number(c) for c in tail]
            numeric_only = [n for n in numeric_tail if n is not None]
            if len(numeric_only) >= 1:
                stub["credit_theory"] = numeric_only[0]
            if len(numeric_only) >= 2:
                stub["credit_practice"] = numeric_only[1]
            if len(numeric_only) >= 3:
                stub["credit_lab"] = numeric_only[2]
            if len(numeric_only) >= 4:
                stub["credit_total"] = numeric_only[3]
            if len(numeric_only) >= 5:
                stub["ects"] = numeric_only[4]
            for c in tail:
                lc = c.lower()
                if "zorunlu" in lc or "compulsory" in lc:
                    stub["type"] = "Zorunlu"
                elif "seçmeli" in lc or "secmeli" in lc or "elective" in lc:
                    stub["type"] = "Seçmeli"
                if lc in {"tr", "türkçe", "turkce", "turkish"}:
                    stub["language"] = "Türkçe"
                elif lc in {"en", "english", "ingilizce"}:
                    stub["language"] = "English"
            result.append(stub)
        return result

    def _absolutise(self, href: str) -> str:
        """Turn a relative href (typical OIBS) into an absolute URL.

        ``javascript:`` href (the ``__doPostBack`` pseudo-link used by
        ASP.NET WebForms) is returned as-is so the caller can detect and
        handle it explicitly instead of getting a 404 on ``page.goto``.
        """
        if not href:
            return href
        if href.startswith("javascript:"):
            return href
        if href.startswith("http"):
            return href
        clean = href.lstrip("./")
        return f"{self.config.base_url}/{clean}"

    # -- Per-course drill-down ------------------------------------------

    def _enrich_course_detail(self, course: BolognaCourse) -> None:
        """Visit the course detail URL and pull Description / Outcomes /
        Weekly / Assessment from the page text.

        OIBS course detail pages render every section (Description, Aims,
        Outcomes, Topics, Assessment) on the same scrollable view; there
        are no inner tabs. We split the dump by section heading.

        ``javascript:__doPostBack`` URLs are skipped here on purpose:
        OIBS uses ASP.NET WebForms and the only way to open a course
        detail panel is to fire the postback in-page. That requires
        leaving the curriculum view, capturing the new content, then
        navigating back — handled by a follow-up step rather than the
        per-course goto path. Meta-data captured by the curriculum
        parser (code, name, T+U+L, ECTS, type, semester) is already
        rich enough for the RAG index.
        """
        if not course.detail_url:
            return
        if course.detail_url.startswith("javascript:") or "__doPostBack" in course.detail_url:
            self.result.warnings.append(f"course_postback_skipped:{course.code}")
            return
        if not self._goto(course.detail_url):
            self.result.warnings.append(f"course_goto_failed:{course.code}")
            return

        full_text = self._collect_full_text()
        if not full_text:
            self.result.warnings.append(f"course_empty:{course.code}")
            return

        cleaned = self._strip_menu_chrome(full_text)
        sections = self._split_course_sections(cleaned)

        course.description_tr = sections.get("description") or sections.get("aims") or None
        outcomes_text = sections.get("learning_outcomes")
        if outcomes_text:
            course.learning_outcomes = _split_outcome_bullets(outcomes_text)
        weekly_text = sections.get("weekly")
        if weekly_text:
            course.weekly_plan = _split_outcome_bullets(weekly_text)
        assessment_text = sections.get("assessment")
        if assessment_text:
            course.assessment = _split_outcome_bullets(assessment_text)
        prereq_text = sections.get("prerequisites")
        if prereq_text:
            course.prerequisites = _split_outcome_bullets(prereq_text)

        if not (course.description_tr or course.learning_outcomes or course.weekly_plan):
            course.raw_text = cleaned[:8000] or None
            self.result.warnings.append(f"course_only_raw_text:{course.code}")

    @staticmethod
    def _split_course_sections(text: str) -> dict[str, str]:
        """Best-effort splitter — scans line by line, swaps the active
        bucket whenever a known heading appears.

        Heading dictionary covers the standard OIBS section names in
        Turkish (the catalogue is mainly Turkish for Acibadem) plus their
        English equivalents that occasionally appear on bilingual courses.
        """
        if not text:
            return {}

        # Map heading-ish substrings to bucket keys.
        heading_map: tuple[tuple[tuple[str, ...], str], ...] = (
            (("ders i̇çeriği", "ders icerigi", "course content", "course description", "ders tanımı", "ders tanimi"), "description"),
            (("amaç", "amac", "course aims", "objective"), "aims"),
            (("öğrenme çıktıları", "ogrenme ciktilari", "öğrenim çıktıları", "ogrenim ciktilari", "learning outcomes", "course outcomes", "ders öğrenme", "ders ogrenme"), "learning_outcomes"),
            (("haftalık", "haftalik", "weekly subjects", "weekly plan", "topics"), "weekly"),
            (("değerlendirme", "degerlendirme", "assessment", "evaluation", "ölçme"), "assessment"),
            (("ön koşul", "on kosul", "prerequisite"), "prerequisites"),
        )

        buckets: dict[str, list[str]] = {}
        active: str | None = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            lowered = line.lower()
            matched_key: str | None = None
            for needles, key in heading_map:
                if any(n in lowered for n in needles):
                    matched_key = key
                    break
            if matched_key:
                active = matched_key
                # Keep heading contents that *follow* the marker on the same
                # line (e.g. ``Ders İçeriği: ...``).
                colon_idx = line.find(":")
                if colon_idx != -1 and colon_idx < len(line) - 1:
                    tail = line[colon_idx + 1 :].strip()
                    if tail:
                        buckets.setdefault(matched_key, []).append(tail)
                continue
            if active:
                buckets.setdefault(active, []).append(line)

        return {key: "\n".join(lines).strip() for key, lines in buckets.items() if lines}


# ---------------------------------------------------------------------------
# Markdown rendering for ScrapedPage.content
# ---------------------------------------------------------------------------


def render_program_overview_markdown(result: BolognaPilotResult) -> str:
    """Markdown summary used for the ``program-overview`` ScrapedPage row."""
    lines: list[str] = []
    p = result.program
    lines.append(f"# {p.department_name}")
    lines.append("")
    lines.append(f"**Fakülte:** {p.faculty_name}")
    if p.program_url:
        lines.append(f"**Bologna URL:** {p.program_url}")
    lines.append("")
    if p.program_description:
        lines.append("## Program Tanıtımı")
        lines.append("")
        lines.append(p.program_description.strip())
        lines.append("")
    if p.program_outcomes:
        lines.append("## Program Çıktıları (Program Outcomes)")
        lines.append("")
        for i, outcome in enumerate(p.program_outcomes, start=1):
            lines.append(f"{i}. {outcome}")
        lines.append("")

    # Each info page becomes its own ## section so chunking can split them.
    label_by_key = {key: label for label, _, key in PROGRAM_INFO_PAGES}
    for key, body in p.info_pages.items():
        if not body:
            continue
        lines.append(f"## {label_by_key.get(key, key)}")
        lines.append("")
        lines.append(body.strip())
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_course_markdown(course: BolognaCourse, program: BolognaProgram) -> str:
    """One Markdown document per course — heading-aware, ready for chunking."""
    lines: list[str] = []
    title = f"{course.code} — {course.name_tr}".strip()
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Fakülte:** {program.faculty_name}")
    lines.append(f"**Bölüm:** {program.department_name}")
    if course.semester is not None:
        lines.append(f"**Yarıyıl:** {course.semester}")
    if course.course_type:
        lines.append(f"**Tip:** {course.course_type}")
    if course.language:
        lines.append(f"**Eğitim Dili:** {course.language}")

    credit_bits: list[str] = []
    if course.credit_theory is not None:
        credit_bits.append(f"Teori {course.credit_theory:g}")
    if course.credit_practice is not None:
        credit_bits.append(f"Uygulama {course.credit_practice:g}")
    if course.credit_lab is not None:
        credit_bits.append(f"Laboratuvar {course.credit_lab:g}")
    if course.credit_total is not None:
        credit_bits.append(f"Kredi {course.credit_total:g}")
    if course.ects is not None:
        credit_bits.append(f"AKTS {course.ects:g}")
    if credit_bits:
        lines.append(f"**Krediler:** {' | '.join(credit_bits)}")
    if course.detail_url and not course.detail_url.startswith("javascript:"):
        lines.append(f"**Bologna URL:** {course.detail_url}")
    lines.append("")

    if course.description_tr:
        lines.append("## Ders İçeriği")
        lines.append("")
        lines.append(course.description_tr.strip())
        lines.append("")
    if course.learning_outcomes:
        lines.append("## Öğrenim Çıktıları")
        lines.append("")
        for i, outcome in enumerate(course.learning_outcomes, start=1):
            lines.append(f"{i}. {outcome}")
        lines.append("")
    if course.weekly_plan:
        lines.append("## Haftalık Plan")
        lines.append("")
        for i, item in enumerate(course.weekly_plan, start=1):
            lines.append(f"- Hafta {i}: {item}")
        lines.append("")
    if course.assessment:
        lines.append("## Değerlendirme")
        lines.append("")
        for item in course.assessment:
            lines.append(f"- {item}")
        lines.append("")
    if course.prerequisites:
        lines.append("## Ön Koşullar")
        lines.append("")
        for item in course.prerequisites:
            lines.append(f"- {item}")
        lines.append("")
    if not (course.description_tr or course.learning_outcomes or course.weekly_plan) and course.raw_text:
        lines.append("## Ham Sayfa İçeriği")
        lines.append("")
        lines.append(course.raw_text.strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def serialise_to_json(result: BolognaPilotResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Info-page helpers (used by the ``ingest_obs_bologna`` persist path)
# ---------------------------------------------------------------------------

# Lookup tables derived from PROGRAM_INFO_PAGES so callers don't have to
# iterate the tuple themselves. Both are keyed by the stable info-page key.
INFO_KEY_TO_ASPX: dict[str, str] = {key: aspx for _, aspx, key in PROGRAM_INFO_PAGES}
INFO_KEY_TO_LABEL: dict[str, str] = {key: label for label, _, key in PROGRAM_INFO_PAGES}


def render_info_page_markdown(
    info_key: str,
    body: str,
    *,
    program: BolognaProgram,
) -> str:
    """One Markdown document per OIBS program info page.

    Each captured ``info_pages[key]`` becomes its own ScrapedPage row;
    rendering it as a self-contained Markdown document means the chunker
    can stay agnostic about which info-page it is processing while
    still emitting heading-aware chunks.
    """
    title = INFO_KEY_TO_LABEL.get(info_key, info_key.replace("_", " ").title())
    lines: list[str] = [f"# {title}", ""]
    lines.append(f"**Fakülte:** {program.faculty_name}")
    lines.append(f"**Bölüm:** {program.department_name}")
    lines.append("")
    cleaned_body = (body or "").strip()
    if cleaned_body:
        lines.append(cleaned_body)
        lines.append("")
    return "\n".join(lines).strip() + "\n"
