"use client";

/**
 * Sources card (Adım 5.2 — new).
 *
 * Renders the ``retrieved_chunks`` array returned by /api/v1/ask
 * (Adım 5.1 contract). Visually it sits inside the assistant
 * .msg-main column directly under the bubble it documents, so the
 * relationship is unambiguous on every assistant turn.
 *
 * Design constraints
 * ------------------
 * * Re-uses the existing dark-glass design tokens (--panel-soft,
 *   --panel-border, --accent...) instead of introducing new colours.
 * * Truncates to ``MAX_SOURCES`` so an unusually large hit list does
 *   not push the composer below the fold.
 * * The chunk metadata field used for the pill label is
 *   ``content_type`` (e.g. "bologna_course"), but with a small mapper
 *   to a human-friendly Turkish label.
 */
import type { RetrievedChunk } from "@/app/lib/types";

type Props = {
  chunks: RetrievedChunk[];
};

const MAX_SOURCES = 5;

const CONTENT_TYPE_LABEL_TR: Record<string, string> = {
  bologna_program: "Program",
  bologna_course: "Ders",
  bologna_officials: "Yönetim",
  bologna_academic_staff: "Akademik kadro",
  bologna_contact: "İletişim",
  bologna_admission: "Kabul",
  bologna_outcomes: "Kazanımlar",
  bologna_graduation: "Mezuniyet",
  bologna_further_studies: "Üst öğrenim",
  bologna_occupation: "Kariyer",
  bologna_degree: "Derece",
  contact: "İletişim",
  announcement: "Duyuru",
  news: "Haber",
  event: "Etkinlik",
};

function pillFromChunk(chunk: RetrievedChunk): { label: string; muted: boolean } {
  const ct = (chunk.content_type || "").trim();
  if (ct && CONTENT_TYPE_LABEL_TR[ct]) {
    return { label: CONTENT_TYPE_LABEL_TR[ct], muted: false };
  }
  if (ct) {
    return { label: ct.replace(/_/g, " "), muted: false };
  }
  if (chunk.course_code) {
    return { label: chunk.course_code, muted: false };
  }
  return { label: "Kaynak", muted: true };
}

function formatScore(score: number): string {
  // Cosine similarity is in [-1, 1]; the production minimum cutoff is 0.55
  // so two-decimal precision is a reasonable balance between density and
  // information content.
  if (!Number.isFinite(score)) return "—";
  return score.toFixed(2);
}

function hostnameOf(url: string): string {
  try {
    const u = new URL(url);
    return u.host + (u.pathname && u.pathname !== "/" ? u.pathname : "");
  } catch (_e) {
    return url;
  }
}

export default function SourcesCard({ chunks }: Props) {
  if (!chunks || chunks.length === 0) {
    return null;
  }
  const visible = chunks.slice(0, MAX_SOURCES);
  return (
    <section className="sources-card" aria-label="Kullanılan kaynaklar">
      <header className="sources-card-header">
        <span>Kaynaklar</span>
        <span className="sources-card-count">{chunks.length}</span>
      </header>
      <ul className="sources-list">
        {visible.map((chunk) => {
          const pill = pillFromChunk(chunk);
          const title = chunk.title?.trim() || hostnameOf(chunk.url);
          const url = chunk.url || "";
          return (
            <li key={chunk.chunk_id} className="source-item">
              <div className="source-item-top">
                <span className={`source-pill${pill.muted ? " muted" : ""}`}>{pill.label}</span>
                {url ? (
                  <a
                    className="source-title"
                    href={url}
                    target="_blank"
                    rel="noopener noreferrer"
                    title={url}
                  >
                    {title}
                  </a>
                ) : (
                  <span className="source-title">{title}</span>
                )}
                <span className="source-score" title="Cosine similarity">
                  {formatScore(chunk.score)}
                </span>
              </div>
              {url ? <span className="source-url">{hostnameOf(url)}</span> : null}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
