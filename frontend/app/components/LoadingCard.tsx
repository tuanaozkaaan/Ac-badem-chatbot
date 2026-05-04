/**
 * Assistant-side loading placeholder — same layout the legacy chat.js
 * built imperatively (`showLoading()`), now declarative.
 */
export default function LoadingCard() {
  return (
    <div className="message-row assistant">
      <div className="msg-aside">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img className="avatar" src="/avatar.png" alt="ACUdost avatar" aria-hidden="true" />
      </div>
      <div className="msg-main">
        <div className="loading-card">
          <div
            style={{
              fontSize: "14px",
              fontWeight: 700,
              color: "var(--text)",
              letterSpacing: "-0.02em",
            }}
          >
            Yanıt hazırlanıyor
          </div>
          <div
            style={{
              fontSize: "12px",
              color: "var(--muted)",
              marginTop: "6px",
              lineHeight: 1.5,
              fontWeight: 500,
            }}
          >
            Veri tabanından ilgili metinler seçiliyor; model yanıtı üretiyor. Yoğun yüklemede bu adım
            birkaç dakika sürebilir.
          </div>
          <div className="loading-shimmer" />
          <div className="loading-shimmer narrow" />
        </div>
      </div>
    </div>
  );
}
