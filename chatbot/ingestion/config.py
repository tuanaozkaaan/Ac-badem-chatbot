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
            "about",
            "akademik",
            "bolum",
            "lisans",
            "onlisans",
            "ders",
            "ogrenci",
        )
    )
