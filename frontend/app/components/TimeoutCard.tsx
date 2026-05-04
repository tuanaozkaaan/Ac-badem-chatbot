"use client";

/**
 * Assistant-side card rendered when the backend reports
 * ``answer_source: "LLM_TIMEOUT"`` (Adım 5.4 — task #1).
 *
 * Visually a sibling of LoadingCard so the user perceives "we tried, the
 * model just took too long" rather than a hard error. The Retry button
 * re-runs the question that triggered this turn through the parent's
 * ``onRetry`` callback; while the retry is in flight the parent disables
 * the button to avoid duplicate /ask calls.
 */
type Props = {
  retrying: boolean;
  onRetry: () => void;
};

export default function TimeoutCard({ retrying, onRetry }: Props) {
  return (
    <div className="timeout-card" role="status" aria-live="polite">
      <div className="timeout-card-icon" aria-hidden="true">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
          <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.8" />
          <path d="M12 7v5l3 2" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        </svg>
      </div>
      <div className="timeout-card-body">
        <strong>Yanıt üretimi gecikti</strong>
        <p>
          Modelde kısa bir yoğunluk olabilir. İsterseniz aynı soruyu yeniden deneyebiliriz; bu kez
          daha hızlı yanıt verme ihtimali yüksek.
        </p>
        <button
          type="button"
          className="timeout-retry-btn"
          onClick={onRetry}
          disabled={retrying}
        >
          {retrying ? (
            <span className="loading-dots" aria-hidden="true">
              <span />
              <span />
              <span />
            </span>
          ) : (
            <>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path
                  d="M3 12a9 9 0 0 1 15.5-6.3M21 4v6h-6M21 12a9 9 0 0 1-15.5 6.3M3 20v-6h6"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              Yeniden dene
            </>
          )}
        </button>
      </div>
    </div>
  );
}
