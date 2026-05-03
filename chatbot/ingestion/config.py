from dataclasses import dataclass, field


@dataclass
class CrawlConfig:
    seed_urls: list[str]
    max_pages: int = 200
    min_delay_seconds: float = 1.0
    max_delay_seconds: float = 2.0
    timeout_seconds: int = 20
    user_agent: str = (
        "AcibademRagBot/1.0 (+responsible crawling; contact: your-team@example.com)"
    )
    enable_playwright_for_obs: bool = False
    # OBS: Playwright ile tıklanacak yüksek değerli aksiyon sayısı üst sınırı (0 = keşif kapalı).
    obs_max_action_clicks: int = 20
    max_content_chars: int = 200_000
    # Keywords used by the priority queue to surface high-value pages first
    # when ``--max-pages`` is small. The spec requires announcements and
    # contact pages, so both English and Turkish slugs sit at the top.
    high_value_keywords: tuple[str, ...] = field(
        default_factory=lambda: (
            "department",
            "faculty",
            "program",
            "admission",
            "course",
            "duyuru",
            "announcement",
            "contact",
            "iletisim",
            "about",
            "hakkinda",
            "akademik",
            "bolum",
            "fakulte",
            "lisans",
            "onlisans",
            "lisansustu",
            "ders",
            "mufredat",
            "ogrenci",
            "kayit",
            "burs",
            "ucret",
            "takvim",
            "uluslararasi",
            "yasam",
        )
    )
