"use client";

/**
 * Welcome card shown when no messages exist yet. The four suggestion
 * cards mirror the SUGGESTIONS array in static/.../app.js — keep
 * `label` / `text` in sync if you change either.
 */
type Props = {
  visible: boolean;
  onSuggestionClick: (text: string) => void;
};

const SUGGESTIONS = [
  { label: "Genel", text: "Acıbadem Üniversitesi hakkında kısa bilgi verir misiniz?" },
  { label: "Kampüs", text: "Acıbadem Üniversitesi kampüs adresi ve ulaşım bilgisi nedir?" },
  {
    label: "Akademik",
    text: "Mühendislik ve Doğa Bilimleri Fakültesi hangi bölümleri içerir?",
  },
  {
    label: "İletişim",
    text: "Öğrenci işleri ve genel iletişim telefon/e-posta bilgileri nelerdir?",
  },
] as const;

export default function Welcome({ visible, onSuggestionClick }: Props) {
  return (
    <div className="welcome" style={{ display: visible ? "block" : "none" }}>
      <div className="welcome-card">
        <div className="welcome-card-inner">
          <div className="welcome-badge">Resmî içerik önceliği</div>
          <h2 className="welcome-title">Bugün size nasıl yardımcı olabilirim?</h2>
          <p className="welcome-subtitle">
            Başvuru, fakülteler, kampüs, iletişim veya programlar hakkında sorularınızı Türkçe veya
            İngilizce sorabilirsiniz. Yanıtlar, taranmış üniversite sayfalarındaki metinlere dayanır.
          </p>
          <div className="suggestions">
            {SUGGESTIONS.map((item) => (
              <button
                key={item.label}
                type="button"
                className="suggestion-card"
                onClick={() => onSuggestionClick(item.text)}
              >
                <span className="label">{item.label}</span>
                <span className="text">{item.text}</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
