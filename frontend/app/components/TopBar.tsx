/**
 * Top header strip with avatar, brand title, tagline, and the "Yerel RAG"
 * pill. Visually identical to the .topbar block in templates/index.html.
 */
export default function TopBar() {
  return (
    <header className="topbar">
      <div className="topbar-inner">
        <div className="topbar-brand">
          {/* eslint-disable-next-line @next/next/no-img-element -- intentional: legacy CSS sizes the avatar; next/image's wrapper would break the .topbar-avatar class behaviour. */}
          <img className="topbar-avatar" src="/avatar.png" alt="ACUdost avatar" />
          <div className="topbar-title-block">
            <h1>ACUdost</h1>
            <p className="topbar-tagline">
              Acıbadem Üniversitesi — resmî kaynaklara dayalı yanıtlar
            </p>
          </div>
        </div>
        <span className="topbar-pill" title="Yerel model + veri tabanı">
          Yerel RAG
        </span>
      </div>
    </header>
  );
}
