"use client";

/**
 * Bottom input strip. Identical layout/handling to the legacy textarea +
 * send button block (chat.js → autoResizeInput, sendMessage on Enter).
 *
 * Behavior notes
 * --------------
 * * Enter sends; Shift+Enter inserts a newline (mirrors the old keydown
 *   handler in static/.../app.js).
 * * The textarea grows up to 200px tall and clamps after that.
 * * `disabled` is forwarded to both the textarea and the button so the
 *   user cannot double-submit while a request is in-flight.
 */
import { useEffect, useRef } from "react";

type Props = {
  value: string;
  disabled: boolean;
  onChange: (value: string) => void;
  onSubmit: () => void;
};

export default function Composer({ value, disabled, onChange, onSubmit }: Props) {
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  // Mirror the legacy auto-resize behaviour: shrink back to min on every value
  // change, then grow to scrollHeight (capped at 200px by CSS max-height).
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [value]);

  // Auto-focus on mount, matching the legacy behaviour at the bottom of app.js.
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      onSubmit();
    }
  };

  return (
    <div className="composer-wrap">
      <div className="composer">
        <textarea
          ref={inputRef}
          className="input-box"
          placeholder="Sorunuzu buraya yazın…"
          rows={1}
          autoComplete="off"
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <div className="composer-footer">
          <span className="hint">Gönder: Enter · satır: Shift+Enter</span>
          <button
            type="button"
            className="send-btn"
            onClick={onSubmit}
            disabled={disabled || !value.trim()}
          >
            Gönder
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path
                d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"
                stroke="currentColor"
                strokeWidth={2}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
