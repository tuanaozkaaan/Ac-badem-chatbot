"""
OBS (ASP.NET) sayfalarında href dışı etkileşimleri keşfetmek ve Playwright ile tıklayıp içerik yakalamak.

Ana site crawler'ından ayrı tutulur; yalnızca Playwright OBS yolu tarafından çağrılır.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

from .content_cleaner import clean_html_to_text, content_hash

logger = logging.getLogger(__name__)

OBS_ACTION_SELECTOR = (
    'a[href], a[onclick], button, input[type="submit"], input[type="button"], '
    "[onclick], tr[onclick], td[onclick]"
)

OBS_INTEREST_KEYWORDS = (
    "müfredat",
    "mufredat",
    "ders",
    "program",
    "çıktı",
    "cikti",
    "içerik",
    "icerik",
    "bologna",
    "syllabus",
    "course",
    "outcome",
    "learning",
    "detay",
    "kredi",
    "akts",
    "öğretim",
    "ogretim",
    "plan",
    "koordinatör",
    "koordinator",
    "tanım",
    "tanim",
    "hafta",
    "laboratuvar",
    "uygulama",
    "notlandırma",
    "notlandirma",
    "değerlendirme",
    "degerlendirme",
)

OBS_SKIP_SUBSTRINGS = (
    "logout",
    "log out",
    "çıkış",
    "cikis",
    "sign out",
    "kapat",
    "cancel",
    "vazgeç",
    "vazgec",
)


@dataclass
class ObsCapture:
    """Tek bir aksiyon sonrası kaydedilecek metin bloğu (canonical_url + url_variant ile ayrılır)."""

    canonical_url: str
    url_variant: str
    title: str
    section: str
    content: str
    action_label: str


def _action_signature(href: str, onclick: str, text: str) -> str:
    raw = f"{href or ''}\x00{onclick or ''}\x00{(text or '')[:160]}"
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


def _url_variant_for_action(canonical_url: str, action_sig: str) -> str:
    raw = f"{canonical_url}\x00{action_sig}".encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()[:48]


def _score_action(href: str, onclick: str, text: str) -> int:
    blob = f"{href or ''} {onclick or ''} {text or ''}".lower()
    score = 0
    for kw in OBS_INTEREST_KEYWORDS:
        if kw in blob:
            score += 2
    low = blob.replace("_", "")
    if "dopostback" in low or "__dopostback" in low:
        score += 6
    if "window.open" in blob:
        score += 2
    if "javascript:" in (href or "").lower():
        score += 1
    for bad in OBS_SKIP_SUBSTRINGS:
        if bad in blob:
            score -= 25
    return score


def _list_dom_actions(page, selector: str) -> list[dict]:
    return page.evaluate(
        """sel => {
          const nodes = [...document.querySelectorAll(sel)];
          return nodes.map((n, i) => ({
            i,
            tag: n.tagName,
            href: n.getAttribute('href') || '',
            onclick: n.getAttribute('onclick') || '',
            text: (n.innerText || n.textContent || '').trim().slice(0, 240)
          }));
        }""",
        selector,
    )


def _click_action_index(page, selector: str, index: int) -> bool:
    return page.evaluate(
        """([sel, index]) => {
          const nodes = [...document.querySelectorAll(sel)];
          const n = nodes[index];
          if (!n) return false;
          n.click();
          return true;
        }""",
        [selector, index],
    )


def _find_index_for_signature(rows: list[dict], target_sig: str) -> int | None:
    for row in rows:
        sig = _action_signature(row.get("href") or "", row.get("onclick") or "", row.get("text") or "")
        if sig == target_sig:
            return int(row["i"])
    return None


def _collect_frame_plain(page) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for frame in page.frames:
        try:
            loc = frame.locator("body")
            if loc.count() == 0:
                continue
            chunk = loc.inner_text(timeout=15_000)
            s = (chunk or "").strip()
            if s and s not in seen:
                seen.add(s)
                parts.append(s)
        except Exception:
            continue
    return "\n\n".join(parts)


def explore_obs_action_pages(
    page,
    canonical_url: str,
    timeout_ms: int,
    max_clicks: int,
) -> tuple[list[ObsCapture], list[str]]:
    """
    Mevcut sayfadan aksiyon adaylarını listeler (debug log), skorlar ve en iyi adaylara tıklayıp
    metin yakalar. Her yakalanan içerik ayrı ScrapedPage satırı için ObsCapture döner.
    """
    max_clicks = max(0, min(int(max_clicks), 60))
    if max_clicks == 0:
        return [], []

    try:
        rows = _list_dom_actions(page, OBS_ACTION_SELECTOR)
    except Exception as exc:
        logger.warning("OBS list_actions_failed url=%s err=%s", canonical_url, exc)
        return [], []

    for row in rows:
        logger.info(
            "OBS_ACTION_CANDIDATE idx=%s tag=%s href=%r onclick=%r text=%r",
            row.get("i"),
            row.get("tag"),
            (row.get("href") or "")[:500],
            (row.get("onclick") or "")[:500],
            (row.get("text") or "")[:240],
        )

    ranked: list[tuple[int, str, dict]] = []
    seen_sig: set[str] = set()
    for row in rows:
        href = row.get("href") or ""
        onclick = row.get("onclick") or ""
        text = row.get("text") or ""
        sc = _score_action(href, onclick, text)
        if sc <= 0 and ("__doPostBack" not in href and "__doPostBack" not in onclick):
            continue
        sig = _action_signature(href, onclick, text)
        if sig in seen_sig:
            continue
        seen_sig.add(sig)
        ranked.append((sc, sig, row))

    if not ranked:
        for row in rows:
            href = row.get("href") or ""
            onclick = row.get("onclick") or ""
            if "__doPostBack" not in href and "__doPostBack" not in onclick:
                continue
            text = row.get("text") or ""
            sig = _action_signature(href, onclick, text)
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            ranked.append((5, sig, row))

    ranked.sort(key=lambda x: -x[0])
    chosen = ranked[:max_clicks]

    captures: list[ObsCapture] = []
    seen_content_hashes: set[str] = set()

    for _sc, sig, meta in chosen:
        label = (meta.get("text") or "").strip()[:120] or sig[:12]

        try:
            page.goto(canonical_url, wait_until="networkidle", timeout=timeout_ms)
            page.wait_for_timeout(3000)
        except Exception as exc:
            logger.warning("OBS explore_reload_failed url=%s err=%s", canonical_url, exc)
            continue

        try:
            fresh_rows = _list_dom_actions(page, OBS_ACTION_SELECTOR)
            idx = _find_index_for_signature(fresh_rows, sig)
        except Exception as exc:
            logger.warning("OBS explore_relist_failed url=%s err=%s", canonical_url, exc)
            continue

        if idx is None:
            logger.info("OBS action index not found after reload label=%r sig=%s", label, sig[:16])
            continue

        try:
            clicked = _click_action_index(page, OBS_ACTION_SELECTOR, idx)
            if not clicked:
                continue
        except Exception as exc:
            logger.warning("OBS explore_click_failed url=%s label=%r err=%s", canonical_url, label, exc)
            continue

        page.wait_for_timeout(2500)
        try:
            page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 25_000))
        except Exception:
            pass

        if "obs.acibadem.edu.tr" not in (page.url or "").lower():
            logger.info("OBS explore_left_host after click url=%s -> %s", canonical_url, page.url)
            try:
                page.goto(canonical_url, wait_until="networkidle", timeout=timeout_ms)
                page.wait_for_timeout(2000)
            except Exception:
                pass
            continue

        try:
            html = page.content()
        except Exception:
            html = ""

        text, title, section = clean_html_to_text(html, max_chars=200_000)
        plain = _collect_frame_plain(page)
        if plain:
            base = text.strip()
            merged = f"{base}\n\n---\n\n{plain}".strip() if base else plain
            text = merged[:200_000]
        try:
            pw_title = (page.title() or "").strip()
            if pw_title and (not (title or "").strip() or len(pw_title) > len((title or "").strip())):
                title = pw_title[:512]
        except Exception:
            pass

        title = f"{title[:400]} — {label}"[:512] if label else title[:512]

        ch = content_hash(text)
        if ch in seen_content_hashes or len(text.strip()) < 40:
            continue
        seen_content_hashes.add(ch)

        variant = _url_variant_for_action(canonical_url, sig)
        captures.append(
            ObsCapture(
                canonical_url=canonical_url,
                url_variant=variant,
                title=title,
                section=(section or "")[:256],
                content=text,
                action_label=label,
            )
        )
        logger.info(
            "OBS_ACTION_CAPTURED label=%r variant=%s text_len=%d",
            label,
            variant[:16],
            len(text),
        )

    seed_urls: list[str] = []
    seen_u: set[str] = set()
    for row in rows:
        oc = row.get("onclick") or ""
        for m in re.finditer(r"window\.open\(\s*['\"]([^'\"]+)['\"]", oc, re.I):
            u = m.group(1).strip()
            if u.startswith("http") and "obs.acibadem.edu.tr" in u.lower() and u not in seen_u:
                seen_u.add(u)
                seed_urls.append(u)
                logger.info(
                    "OBS_WINDOW_OPEN_URL url=%r text=%r",
                    u[:800],
                    (row.get("text") or "")[:120],
                )
        href = row.get("href") or ""
        if href.startswith("http") and "obs.acibadem.edu.tr" in href.lower() and href not in seen_u:
            seen_u.add(href)
            seed_urls.append(href)

    return captures, seed_urls
