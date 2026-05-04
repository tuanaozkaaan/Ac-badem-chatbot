"use client";

/**
 * Suggestion chips rendered under an assistant bubble whose
 * ``answer_source`` is ``NO_INFO`` or ``FALLBACK`` (Adım 5.4 — task #2).
 *
 * The user typed something the chatbot could not answer; instead of
 * leaving them at a dead end we offer 3 alternative questions that
 * **do** map to indexed content. The list is intentionally curated and
 * static — generating suggestions from retrieval scores would not help
 * for queries that returned nothing in the first place.
 *
 * Click → calls back into the parent (Chat.tsx) which feeds the same
 * "send a question" path the composer uses, so the suggestion behaves
 * exactly like a normal user turn.
 */
type Props = {
  onPick: (text: string) => void;
  /** Disabled while a request is already in flight. */
  disabled?: boolean;
};

const SUGGESTIONS = [
  "Acıbadem Üniversitesi hakkında kısa bilgi verir misiniz?",
  "Mühendislik ve Doğa Bilimleri Fakültesi hangi bölümleri içerir?",
  "Bilgisayar mühendisliği dersleri nelerdir?",
] as const;

export default function NoInfoSuggestions({ onPick, disabled = false }: Props) {
  return (
    <div className="no-info-suggestions" role="group" aria-label="Önerilen sorular">
      <span className="no-info-suggestions-label">Bunları deneyebilirsiniz</span>
      <div className="no-info-suggestions-row">
        {SUGGESTIONS.map((text) => (
          <button
            key={text}
            type="button"
            className="no-info-chip"
            onClick={() => onPick(text)}
            disabled={disabled}
          >
            {text}
          </button>
        ))}
      </div>
    </div>
  );
}
