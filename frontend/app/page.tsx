/**
 * Single-page shell for the chatbot UI. The entire interactive surface
 * lives in <Chat /> so the page component itself stays a server component
 * (no behaviour to ship to the browser, just the wrapping div).
 */
import Chat from "./components/Chat";

export default function Page() {
  return (
    <div className="app">
      <Chat />
    </div>
  );
}
