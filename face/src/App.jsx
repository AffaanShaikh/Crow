import { useState, useEffect, useRef, useCallback } from "react";

// config
const API_BASE                = "http://localhost:8000/api/v1";
const WS_BASE                 = "ws://localhost:8000/api/v1";
const IMAGE_PROCESSOR_URL     = "http://localhost:8001/process-image";
const SESSION_ID              = crypto.randomUUID();
const DEFAULT_AI_IMAGE_URL    = "/src/assets/metaphor-refantazio-removebg-preview.png";
const DEFAULT_USER_IMAGE_URL  = "/src/assets/i-really-love-the-portraits-from-metaphor-refantazio-so-v0-y945244wls9g1-removebg-preview.png";

const ASR_SAMPLE_RATE   = 16000;
const ASR_BUFFER_SIZE   = 4096;
const ASR_CHUNK_SAMPLES = 512;

const TTS_MIN_SENTENCE_CHARS = 20;
const TTS_SENTENCE_END_RE    = /[.!?...]["')\]]*\s/;

const TOOL_LABELS = {
  list_calendar_events:   "Checking calendar",
  get_calendar_event:     "Reading event",
  create_calendar_event:  "Creating event",
  update_calendar_event:  "Updating event",
  delete_calendar_event:  "Deleting event",
  spotify_get_playback:   "Reading playback",
  spotify_play:           "Playing music",
  spotify_pause:          "Pausing music",
  spotify_next_track:     "Next track",
  spotify_previous_track: "Previous track",
  spotify_set_repeat:     "Setting repeat",
  spotify_set_shuffle:    "Setting shuffle",
  spotify_set_volume:     "Setting volume",
  spotify_search:         "Searching Spotify",
  spotify_get_playlists:  "Getting playlists",
};
const toolLabel = (name) => TOOL_LABELS[name] ?? `Running ${name}`;

const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=UnifrakturMaguntia&family=IM+Fell+English:ital@0;1&family=Cinzel+Decorative:wght@400;700;900&family=Special+Elite&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:           #1a0008;
    --bg2:          #230010;
    --panel:        #300016;
    --border:       #822040;
    --border-glow:  #c03060;
    --gold:         #f0c840;
    --gold-bright:  #fde878;
    --gold-dim:     #6e480e;
    --crimson:      #c4123c;
    --rose:         #e04870;
    --green:        #72bb44;
    --red:          #e85858;
    --purple:       #a78bfa;
    --purple-dim:   #2d1f5e;
    --text:         #f5e2d5;
    --text-dim:     #c09090;
    --text-muted:   #7a2e48;
    --user-bg:      #240010;
    --ai-bg:        #1c000c;
    --tool-bg:      #1e0010;
    --scrollbar:    #6a1c38;
    --frame-gold:   #d4a838;
    --frame-dark:   #56101e;
  }

  html, body, #root {
    height: 100%; width: 100%; background: var(--bg); color: var(--text);
    font-family: 'IM Fell English', serif; font-size: 17px; overflow: hidden;
  }

  body::before {
    content: ''; position: fixed; inset: 0;
    background-image:
      radial-gradient(1.5px 1.5px at 30px 30px, rgba(212,168,56,0.18) 0%, transparent 100%),
      radial-gradient(1px 1px at 70px 70px, rgba(130,32,64,0.28) 0%, transparent 100%);
    background-size: 100px 100px; pointer-events: none; z-index: 0;
  }

  body::after {
    content: ''; position: fixed; inset: 0;
    background: radial-gradient(ellipse 80% 80% at 50% 50%, transparent 25%, rgba(10,0,4,0.55) 100%);
    pointer-events: none; z-index: 9998;
  }

  ::-webkit-scrollbar       { width: 5px; }
  ::-webkit-scrollbar-track { background: var(--bg2); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  @keyframes fadeSlideIn { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
  @keyframes pulse-border { 0%,100%{box-shadow:0 0 6px rgba(192,48,96,0.25) inset} 50%{box-shadow:0 0 28px rgba(192,48,96,0.7) inset,0 0 36px rgba(240,200,64,0.15)} }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
  @keyframes avatar-breathe { 0%,100%{filter:brightness(1) drop-shadow(0 0 12px rgba(212,168,56,0.35))} 50%{filter:brightness(1.12) drop-shadow(0 0 26px rgba(212,168,56,0.72))} }
  @keyframes avatar-breathe-user { 0%,100%{filter:brightness(1) drop-shadow(0 0 12px rgba(196,18,60,0.35))} 50%{filter:brightness(1.1) drop-shadow(0 0 24px rgba(196,18,60,0.62))} }
  @keyframes avatar-active { 0%,100%{filter:brightness(1.15) drop-shadow(0 0 22px rgba(212,168,56,0.85))} 50%{filter:brightness(1.3) drop-shadow(0 0 42px rgba(212,168,56,1))} }
  @keyframes typing-dot { 0%,80%,100%{transform:scale(0.55);opacity:0.35} 40%{transform:scale(1);opacity:1} }
  @keyframes status-glow { 0%,100%{box-shadow:0 0 4px currentColor} 50%{box-shadow:0 0 14px currentColor,0 0 28px currentColor} }
  @keyframes gold-pulse { 0%,100%{text-shadow:0 0 8px rgba(240,200,64,0.25);color:var(--gold)} 50%{text-shadow:0 0 26px rgba(240,200,64,0.82),0 0 50px rgba(240,200,64,0.25);color:var(--gold-bright)} }
  @keyframes portrait-entrance { from{opacity:0;transform:translateY(-8px) scale(0.95)} to{opacity:1;transform:translateY(0) scale(1)} }
  @keyframes ornament-glow { 0%,100%{filter:drop-shadow(0 0 3px rgba(212,168,56,0.4))} 50%{filter:drop-shadow(0 0 12px rgba(212,168,56,0.9))} }
  @keyframes spin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
  @keyframes proc-pulse { 0%,100%{opacity:0.55} 50%{opacity:1} }
  @keyframes tool-glow { 0%,100%{box-shadow:0 0 6px rgba(167,139,250,0.3)} 50%{box-shadow:0 0 18px rgba(167,139,250,0.7)} }
  @keyframes rec-pulse { 0%,100%{opacity:1;box-shadow:0 0 8px var(--red)} 50%{opacity:0.6;box-shadow:0 0 20px var(--red),0 0 32px var(--red)} }
  @keyframes modal-in { from{opacity:0;transform:scale(0.96) translateY(8px)} to{opacity:1;transform:scale(1) translateY(0)} }
  @keyframes wake-pulse { 0%,100%{box-shadow:0 0 8px rgba(114,187,68,0.4)} 50%{box-shadow:0 0 22px rgba(114,187,68,0.9),0 0 40px rgba(114,187,68,0.3)} }

  .rag-drop-zone { transition: all 0.2s; }
  .rag-drop-zone.drag-over {
    border-color: var(--frame-gold) !important;
    background: rgba(212,168,56,0.06) !important;
    box-shadow: 0 0 20px rgba(212,168,56,0.2);
  }
`;

function injectCSS(css) {
  if (document.getElementById("app-styles")) return;
  const el = document.createElement("style");
  el.id = "app-styles";
  el.textContent = css;
  document.head.appendChild(el);
}

// Image processor hook
function useProcessedAvatar(imageUrl) {
  const [src, setSrc] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!imageUrl) return;
    let cancelled = false;
    setProcessing(true);
    setError(null);
    (async () => {
      try {
        const resp = await fetch(IMAGE_PROCESSOR_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_url: imageUrl }),
        });
        if (cancelled) return;
        if (resp.ok) {
          const data = await resp.json();
          if (!cancelled) setSrc(`data:image/png;base64,${data.image}`);
        } else { throw new Error(`HTTP ${resp.status}`); }
      } catch (err) {
        if (!cancelled) { setError(err.message); setSrc(imageUrl); }
      } finally {
        if (!cancelled) setProcessing(false);
      }
    })();
    return () => { cancelled = true; };
  }, [imageUrl]);
  return { src, processing, error };
}

// ASR utilities
function float32ToInt16(float32Arr) {
  const int16 = new Int16Array(float32Arr.length);
  for (let i = 0; i < float32Arr.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Arr[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return int16;
}

// TTS sentence streamer
class TTSSentenceStreamer {
  constructor(apiBase, isMutedFn = () => false) {
    this._apiBase = apiBase; this._isMuted = isMutedFn;
    this._buf = ""; this._queue = []; this._playing = false;
    this._current = null; this._enabled = true;
  }
  push(token) {
    if (!this._enabled || this._isMuted()) return;
    this._buf += token;
    if (this._buf.length >= TTS_MIN_SENTENCE_CHARS && TTS_SENTENCE_END_RE.test(this._buf)) {
      const s = this._buf.trim(); this._buf = "";
      if (s) this._synthesiseAndQueue(s);
    }
  }
  flush() {
    if (!this._enabled || this._isMuted()) return;
    const rem = this._buf.trim(); this._buf = "";
    if (rem.length > 2) this._synthesiseAndQueue(rem);
  }
  stop() {
    this._enabled = false; this._buf = ""; this._queue = [];
    if (this._current) { this._current.pause(); this._current = null; }
    this._playing = false;
  }
  reset() { this._enabled = true; this._buf = ""; this._queue = []; this._playing = false; this._current = null; }
  async _synthesiseAndQueue(text) {
    try {
      const res = await fetch(`${this._apiBase}/audio/tts`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok || !this._enabled) return;
      const blob = await res.blob();
      if (!this._enabled) return;
      this._queue.push(blob);
      if (!this._playing) this._playNext();
    } catch { }
  }
  _playNext() {
    if (!this._queue.length || !this._enabled || this._isMuted()) {
      this._playing = false; this._current = null; return;
    }
    this._playing = true;
    const url = URL.createObjectURL(this._queue.shift());
    const audio = new Audio(url);
    this._current = audio;
    audio.onended = () => { URL.revokeObjectURL(url); this._current = null; this._playNext(); };
    audio.onerror = () => { URL.revokeObjectURL(url); this._current = null; this._playNext(); };
    audio.play().catch(() => this._playNext());
  }
}

// Sub-components
function Timestamp({ ts }) {
  return (
    <span style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontFamily: "'Special Elite', monospace", fontStyle: "italic", marginTop: 7, display: "block", letterSpacing: "0.05em" }}>
      ✦ {ts}
    </span>
  );
}

function ProcessingRing({ accentCol }) {
  return (
    <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 6, zIndex: 4 }}>
      <div style={{ width: 48, height: 48, border: `1.5px solid ${accentCol}25`, borderTop: `1.5px solid ${accentCol}`, borderRadius: "50%", animation: "spin 0.9s linear infinite" }} />
      <span style={{ fontFamily: "'Special Elite', monospace", fontSize: "0.44rem", color: accentCol, letterSpacing: "0.14em", textTransform: "uppercase", animation: "proc-pulse 1.2s ease-in-out infinite" }}>processing</span>
    </div>
  );
}

function CharPortrait({ isUser, isStreaming, isOnline, avatarSrc, isProcessing }) {
  const accentCol = isUser ? "var(--rose)" : "var(--frame-gold)";
  const nameText = isUser ? "You" : "Elda";
  const isActive = isStreaming && !isUser;
  const fallbackSym = isUser ? "✦" : (isStreaming ? "☽" : "☾");
  const imgAnim = isActive ? "avatar-active 1.4s ease-in-out infinite" : isUser ? "avatar-breathe-user 6s ease-in-out infinite" : "avatar-breathe 5s ease-in-out infinite";

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 0, animation: "portrait-entrance 0.3s ease both", marginBottom: "-2", position: "relative", zIndex: 3 }}>
      <div style={{ position: "relative", width: 100, height: 120 }}>
        {isProcessing ? (
          <div style={{ width: 100, height: 120, display: "flex", alignItems: "center", justifyContent: "center", position: "relative" }}>
            <div style={{ fontSize: "1.6rem", color: accentCol, opacity: 0.18, lineHeight: 1, userSelect: "none", paddingBottom: "15px" }}>{fallbackSym}</div>
            <ProcessingRing accentCol={accentCol} />
          </div>
        ) : avatarSrc ? (
          <img src={avatarSrc} alt={nameText} draggable={false} style={{ width: 100, height: 120, objectFit: "contain", objectPosition: "bottom", display: "block", animation: imgAnim, userSelect: "none", pointerEvents: "none" }} />
        ) : (
          <div style={{ width: 100, height: 120, display: "flex", alignItems: "center", justifyContent: "center", fontSize: "3.2rem", color: accentCol, filter: `drop-shadow(0 0 12px ${accentCol})`, opacity: 0.75 }}>{fallbackSym}</div>
        )}
        {isActive && !isProcessing && (
          <div style={{ position: "absolute", bottom: 2, left: "50%", transform: "translateX(-50%)", width: 7, height: 7, borderRadius: "50%", background: "var(--gold)", color: "var(--gold)", animation: "status-glow 0.9s ease-in-out infinite", zIndex: 5 }} />
        )}
      </div>
    </div>
  );
}

function ToolCallPill({ toolName, done, success }) {
  const label = toolLabel(toolName);
  const borderColor = done ? (success ? "var(--green)" : "var(--red)") : "var(--purple)";
  const textColor = done ? (success ? "var(--green)" : "var(--red)") : "var(--purple)";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "7px 14px", background: "var(--tool-bg)", border: `1px solid ${borderColor}`, borderTop: `2px solid ${borderColor}`, clipPath: "polygon(0 6px, 6px 0, 100% 0, 100% 100%, 0 100%)", fontSize: "0.72rem", fontFamily: "'Special Elite', monospace", color: textColor, animation: done ? "fadeSlideIn 0.15s ease both" : "fadeSlideIn 0.15s ease both, tool-glow 1.5s ease-in-out infinite", maxWidth: "60%", boxShadow: done ? "none" : "0 0 12px rgba(167,139,250,0.2)" }}>
      <span style={{ display: "inline-block", animation: done ? "none" : "spin 1s linear infinite", fontSize: "0.75rem" }}>{done ? (success ? "✓" : "✗") : "◌"}</span>
      <span>{done ? `${label} ✦ ${success ? "done" : "failed"}` : `✦ ${label}...`}</span>
    </div>
  );
}

function ThinkingBubble({ content }) {
  return (
    <div style={{ padding: "9px 16px", background: "transparent", border: "1px dashed var(--border)", borderLeft: "2px solid var(--purple)", fontSize: "0.82rem", fontStyle: "italic", color: "var(--text-dim)", maxWidth: "80%", animation: "fadeSlideIn 0.15s ease both", fontFamily: "'IM Fell English', serif" }}>
      <span style={{ color: "var(--purple)", fontSize: "0.7rem", letterSpacing: "0.1em", fontFamily: "'Special Elite', monospace" }}>◈ cogitating</span>
      <div style={{ marginTop: 4 }}>{content}</div>
    </div>
  );
}

function MessageBubble({ msg, status, avatarSrc, isAvatarProcessing }) {
  const isUser = msg.role === "user";
  const isActive = !!msg.streaming;
  const isOnline = status === "online";
  const borderCol = isUser ? "var(--crimson)" : "var(--frame-gold)";
  const accentCol = isUser ? "var(--rose)" : "var(--gold-bright)";
  const nameLabel = isUser ? "You" : "Elda";

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: isUser ? "flex-end" : "flex-start", animation: "fadeSlideIn 0.25s ease both", marginBottom: 26, gap: 6 }}>
      <div style={{ marginLeft: isUser ? 0 : 8, marginRight: isUser ? 8 : 0 }}>
        <CharPortrait isUser={isUser} isStreaming={isActive} isOnline={isOnline} avatarSrc={avatarSrc} isProcessing={isAvatarProcessing} />
      </div>
      {msg.toolCalls?.map(tc => <ToolCallPill key={tc.call_id} toolName={tc.tool_name} done={tc.done} success={tc.success} />)}
      {msg.thinking && <ThinkingBubble content={msg.thinking} />}
      {(msg.content || msg.streaming) && (
        <div style={{ maxWidth: "73%", position: "relative", padding: 3, background: `linear-gradient(135deg, ${accentCol}44, ${borderCol}22)`, boxShadow: isActive ? `0 0 28px ${borderCol}66` : `0 2px 20px rgba(0,0,0,0.7)`, transition: "box-shadow 0.4s ease", clipPath: isUser ? "polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 0 100%)" : "polygon(0 10px, 10px 0, 100% 0, 100% 100%, 0 100%)" }}>
          <div style={{ background: isUser ? "var(--user-bg)" : "var(--ai-bg)", padding: "14px 20px 12px", clipPath: isUser ? "polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 0 100%)" : "polygon(0 8px, 8px 0, 100% 0, 100% 100%, 0 100%)", borderTop: `2px solid ${borderCol}` }}>
            <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 9, fontFamily: "'UnifrakturMaguntia', serif", fontSize: isUser ? "0.88rem" : "1.05rem", color: accentCol, letterSpacing: "0.04em", animation: !isUser ? "gold-pulse 3.5s ease-in-out infinite" : "none", textShadow: `0 0 14px ${accentCol}55` }}>
              <span style={{ fontFamily: "'Special Elite', monospace", fontSize: "0.68rem", opacity: 0.65, letterSpacing: "0.06em" }}>🙮</span>
              {nameLabel}
            </div>
            <div style={{ whiteSpace: "pre-wrap", wordBreak: "break-word", fontSize: "1.04rem", lineHeight: 1.8, color: "var(--text)", fontFamily: "'Special Elite', monospace" }}>
              {msg.content}
              {msg.streaming && <span style={{ display: "inline-block", width: 9, height: 16, background: "var(--gold)", marginLeft: 3, verticalAlign: "middle", animation: "blink 0.8s step-start infinite", opacity: 0.9 }} />}
            </div>
            <Timestamp ts={msg.ts} />
          </div>
        </div>
      )}
    </div>
  );
}

function TypingIndicator({ avatarSrc, isAvatarProcessing }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start", marginBottom: 26 }}>
      <div style={{ marginLeft: 14 }}>
        <CharPortrait isUser={false} isStreaming={false} isOnline={true} avatarSrc={avatarSrc} isProcessing={isAvatarProcessing} />
      </div>
      <div style={{ padding: "14px 22px", background: "var(--ai-bg)", borderTop: "2px solid var(--frame-gold)", clipPath: "polygon(0 8px, 8px 0, 100% 0, 100% 100%, 0 100%)", display: "flex", gap: 8, alignItems: "center", marginLeft: 14, boxShadow: "0 2px 20px rgba(0,0,0,0.7)" }}>
        {[0, 0.2, 0.4].map((delay, i) => <div key={i} style={{ width: 9, height: 9, borderRadius: "50%", background: "var(--gold)", animation: `typing-dot 1.3s ${delay}s ease-in-out infinite` }} />)}
      </div>
    </div>
  );
}

// Auth connect button
function AuthConnectBtn({ label, connected, onLogin, onLogout }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
      <span style={{ color: connected ? "var(--green)" : "var(--text-muted)", fontStyle: "italic" }}>{label}</span>
      <button onClick={connected ? onLogout : onLogin}
        style={{ fontFamily: "'Cinzel Decorative', serif", fontSize: "0.42rem", background: "transparent", border: `1px solid ${connected ? "var(--border)" : "var(--frame-gold)"}`, color: connected ? "var(--text-muted)" : "var(--gold)", padding: "2px 8px", borderRadius: 2, cursor: "pointer", letterSpacing: "0.1em", transition: "all 0.2s" }}
        onMouseEnter={e => { e.target.style.borderColor = "var(--frame-gold)"; e.target.style.color = "var(--gold)"; }}
        onMouseLeave={e => { e.target.style.borderColor = connected ? "var(--border)" : "var(--frame-gold)"; e.target.style.color = connected ? "var(--text-muted)" : "var(--gold)"; }}>
        {connected ? "unlink" : "connect"}
      </button>
    </div>
  );
}

// RAG Panel Modal
// Full-featured knowledge base management: upload, list, delete, test search,
// a modal overlay when the user clicks "✦ Knowledge" in the sidebar.
function RAGModal({ onClose }) {
  const [docs,        setDocs]        = useState([]);
  const [uploading,   setUploading]   = useState(false);
  const [dragOver,    setDragOver]    = useState(false);
  const [searchQ,     setSearchQ]     = useState("");
  const [searchRes,   setSearchRes]   = useState(null);
  const [searching,   setSearching]   = useState(false);
  const [uploadErr,   setUploadErr]   = useState("");
  const [activeTab,   setActiveTab]   = useState("docs"); // "docs" | "search"
  const fileInputRef = useRef(null);

  const loadDocs = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/rag/documents`);
      if (r.ok) { const d = await r.json(); setDocs(d.documents || []); }
    } catch { }
  }, []);

  useEffect(() => { loadDocs(); }, [loadDocs]);

  const uploadFile = async (file) => {
    if (!file) return;
    setUploading(true); setUploadErr("");
    const fd = new FormData();
    fd.append("file", file);
    fd.append("collection", "default");
    try {
      const r = await fetch(`${API_BASE}/rag/documents`, { method: "POST", body: fd });
      const d = await r.json();
      if (r.ok) { await loadDocs(); }
      else { setUploadErr(d.detail || "Upload failed"); }
    } catch (err) { setUploadErr(err.message); }
    finally { setUploading(false); }
  };

  const deleteDoc = async (docId, title) => {
    if (!confirm(`Delete "${title}" from the knowledge base?`)) return;
    await fetch(`${API_BASE}/rag/documents/${docId}?collection=default`, { method: "DELETE" });
    await loadDocs();
  };

  const doSearch = async () => {
    if (!searchQ.trim()) return;
    setSearching(true); setSearchRes(null);
    try {
      const r = await fetch(`${API_BASE}/rag/search`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: searchQ, collection: "default", k: 5, score_threshold: 0.2 }),
      });
      if (r.ok) { const d = await r.json(); setSearchRes(d); }
    } catch { }
    finally { setSearching(false); }
  };

  const onDrop = (e) => {
    e.preventDefault(); setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  };

  const panelStyle = {
    fontFamily: "'Special Elite', monospace", fontSize: "0.67rem",
    color: "var(--text-dim)",
  };

  const tabStyle = (active) => ({
    fontFamily: "'Cinzel Decorative', serif", fontSize: "0.5rem",
    letterSpacing: "0.15em", padding: "5px 14px",
    background: active ? "var(--panel)" : "transparent",
    border: "1px solid var(--border)",
    borderBottom: active ? "1px solid var(--panel)" : "1px solid var(--border)",
    color: active ? "var(--gold)" : "var(--text-muted)",
    cursor: "pointer", transition: "all 0.2s", marginBottom: -1,
  });

  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 9000, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(10,0,4,0.85)" }}
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div style={{ width: 660, maxHeight: "82vh", display: "flex", flexDirection: "column", background: "var(--bg2)", border: "1px solid var(--frame-gold)", borderTop: "3px solid var(--frame-gold)", animation: "modal-in 0.2s ease both", overflow: "hidden" }}>

        {/* Header */}
        <div style={{ padding: "14px 18px", display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--border)" }}>
          <span style={{ fontFamily: "'UnifrakturMaguntia', serif", fontSize: "1.3rem", color: "var(--gold)", animation: "gold-pulse 3.5s ease-in-out infinite" }}>Knowledge Base</span>
          <button onClick={onClose} style={{ background: "transparent", border: "1px solid var(--border)", color: "var(--text-muted)", padding: "2px 10px", cursor: "pointer", fontFamily: "'Special Elite', monospace", fontSize: "0.6rem", borderRadius: 2, transition: "all 0.2s" }}
            onMouseEnter={e => { e.target.style.borderColor = "var(--rose)"; e.target.style.color = "var(--rose)"; }}
            onMouseLeave={e => { e.target.style.borderColor = "var(--border)"; e.target.style.color = "var(--text-muted)"; }}>
            ✕ close
          </button>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", padding: "0 18px", borderBottom: "1px solid var(--border)", gap: 4 }}>
          <button style={tabStyle(activeTab === "docs")} onClick={() => setActiveTab("docs")}>✦ Documents</button>
          <button style={tabStyle(activeTab === "search")} onClick={() => setActiveTab("search")}>✦ Search</button>
        </div>

        <div style={{ overflowY: "auto", padding: "16px 18px", flex: 1 }}>

          {activeTab === "docs" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              {/* Drop zone */}
              <div className={`rag-drop-zone${dragOver ? " drag-over" : ""}`}
                style={{ border: "1px dashed var(--border)", padding: "22px 18px", textAlign: "center", cursor: "pointer", borderRadius: 2 }}
                onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={onDrop}
                onClick={() => fileInputRef.current?.click()}>
                <input ref={fileInputRef} type="file"
                  accept=".txt,.md,.pdf,.docx,.html,.py,.js,.ts,.json,.yaml,.csv"
                  style={{ display: "none" }}
                  onChange={e => { const f = e.target.files?.[0]; if (f) uploadFile(f); e.target.value = ""; }} />
                {uploading ? (
                  <div style={{ color: "var(--frame-gold)", ...panelStyle }}>
                    <div style={{ display: "inline-block", width: 16, height: 16, border: "1.5px solid rgba(212,168,56,0.3)", borderTop: "1.5px solid var(--frame-gold)", borderRadius: "50%", animation: "spin 0.8s linear infinite", marginRight: 8, verticalAlign: "middle" }} />
                    Processing document...
                  </div>
                ) : (
                  <>
                    <div style={{ color: "var(--frame-gold)", fontSize: "1.4rem", marginBottom: 6 }}>⊕</div>
                    <div style={{ ...panelStyle, color: "var(--text-dim)" }}>
                      Drop a file here or click to browse
                    </div>
                    <div style={{ ...panelStyle, color: "var(--text-muted)", marginTop: 4, fontSize: "0.6rem" }}>
                      PDF · DOCX · TXT · MD · HTML · code files · max 50 MB
                    </div>
                  </>
                )}
                {uploadErr && <div style={{ color: "var(--red)", ...panelStyle, marginTop: 8 }}>⚠ {uploadErr}</div>}
              </div>

              {/* Document list */}
              {docs.length === 0 ? (
                <div style={{ ...panelStyle, color: "var(--text-muted)", textAlign: "center", padding: "18px 0", fontStyle: "italic" }}>
                  No documents in the knowledge base yet.
                </div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  <div style={{ ...panelStyle, color: "var(--frame-gold)", fontSize: "0.54rem", letterSpacing: "0.15em", fontFamily: "'Cinzel Decorative', serif", marginBottom: 2 }}>
                    ✦ {docs.length} Document{docs.length !== 1 ? "s" : ""}
                  </div>
                  {docs.map(doc => (
                    <div key={doc.document_id} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 12px", background: "var(--panel)", border: "1px solid var(--border)", borderLeft: "2px solid var(--gold-dim)", animation: "fadeSlideIn 0.2s ease both" }}>
                      <div style={{ display: "flex", flexDirection: "column", gap: 2, overflow: "hidden" }}>
                        <span style={{ ...panelStyle, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 420 }}>{doc.title}</span>
                        <span style={{ ...panelStyle, color: "var(--text-muted)", fontSize: "0.6rem" }}>
                          {doc.file_type.toUpperCase()} · {doc.chunk_count} chunks · {(doc.char_count / 1000).toFixed(1)}k chars
                        </span>
                      </div>
                      <button onClick={() => deleteDoc(doc.document_id, doc.title)}
                        style={{ ...panelStyle, background: "transparent", border: "1px solid var(--border)", color: "var(--text-muted)", padding: "2px 8px", cursor: "pointer", borderRadius: 2, flexShrink: 0, transition: "all 0.2s" }}
                        onMouseEnter={e => { e.target.style.borderColor = "var(--red)"; e.target.style.color = "var(--red)"; }}
                        onMouseLeave={e => { e.target.style.borderColor = "var(--border)"; e.target.style.color = "var(--text-muted)"; }}>
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === "search" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <div style={{ ...panelStyle, color: "var(--text-dim)", fontStyle: "italic", marginBottom: 4 }}>
                Test retrieval quality - enter a question to see which chunks the model would receive.
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <input
                  value={searchQ}
                  onChange={e => setSearchQ(e.target.value)}
                  onKeyDown={e => { if (e.key === "Enter") doSearch(); }}
                  placeholder="Ask something about your documents..."
                  style={{ flex: 1, background: "var(--panel)", border: "1px solid var(--border)", color: "var(--text)", fontFamily: "'Special Elite', monospace", fontSize: "0.82rem", padding: "8px 12px", outline: "none", borderRadius: 2, transition: "border-color 0.2s" }}
                  onFocus={e => { e.target.style.borderColor = "var(--frame-gold)"; }}
                  onBlur={e => { e.target.style.borderColor = "var(--border)"; }}
                />
                <button onClick={doSearch} disabled={searching || !searchQ.trim()}
                  style={{ ...panelStyle, background: searching ? "transparent" : "var(--gold-dim)", border: "1px solid var(--frame-gold)", color: "var(--gold)", padding: "8px 16px", cursor: searching ? "not-allowed" : "pointer", borderRadius: 2, transition: "all 0.2s", flexShrink: 0 }}>
                  {searching ? "..." : "Search"}
                </button>
              </div>

              {searchRes && (
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <div style={{ ...panelStyle, color: "var(--text-muted)", fontSize: "0.62rem" }}>
                    {searchRes.results.length} result{searchRes.results.length !== 1 ? "s" : ""} · {searchRes.retrieval_ms}ms
                    {searchRes.reranked ? " · reranked" : " · dense only"}
                  </div>
                  {searchRes.results.length === 0 && (
                    <div style={{ ...panelStyle, color: "var(--text-muted)", fontStyle: "italic", padding: "14px 0" }}>
                      No relevant chunks found. Try lowering the score threshold or uploading more documents.
                    </div>
                  )}
                  {searchRes.results.map((r, i) => (
                    <div key={i} style={{ padding: "10px 14px", background: "var(--panel)", border: "1px solid var(--border)", borderLeft: `2px solid ${r.score > 0.6 ? "var(--green)" : r.score > 0.4 ? "var(--gold-dim)" : "var(--border)"}`, animation: "fadeSlideIn 0.15s ease both" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                        <span style={{ ...panelStyle, color: "var(--frame-gold)", fontSize: "0.6rem" }}>{r.title}</span>
                        <span style={{ ...panelStyle, color: r.score > 0.6 ? "var(--green)" : "var(--gold-dim)", fontSize: "0.6rem" }}>
                          {(r.score * 100).toFixed(0)}% match
                        </span>
                      </div>
                      <div style={{ ...panelStyle, color: "var(--text-dim)", lineHeight: 1.6, fontSize: "0.65rem" }}>{r.text}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// System Panel
function SystemPanel({ status, streaming, featureFlags, authStatus, ragDocs, onAuthLogin, onAuthLogout, onOpenRAG, wakeActive }) {
  const isOnline = status === "online";
  return (
    <div style={{ width: 210, flexShrink: 0, display: "flex", flexDirection: "column", gap: 13, padding: "20px 15px", borderRight: "1px solid var(--border)", background: "var(--bg2)", overflow: "auto", position: "relative", zIndex: 1 }}>
      <div style={{ fontFamily: "'Cinzel Decorative', serif", fontSize: "0.56rem", color: "var(--frame-gold)", letterSpacing: "0.2em", textAlign: "center", borderBottom: "1px solid var(--border)", paddingBottom: 10, animation: "ornament-glow 4s ease-in-out infinite", flexShrink: 0 }}>
        ✦ C O D E X ✦
      </div>

      <div style={{ border: "1px solid var(--border)", borderTop: "2px solid var(--frame-gold)", padding: "11px 13px", background: "var(--panel)", display: "flex", flexDirection: "column", gap: 5, flexShrink: 0 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span style={{ fontFamily: "'UnifrakturMaguntia', serif", fontSize: "1.45rem", color: "var(--gold)", animation: "gold-pulse 3.5s ease-in-out infinite" }}>Elda</span>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: isOnline ? "var(--green)" : "var(--red)", color: isOnline ? "var(--green)" : "var(--red)", animation: "status-glow 2.5s ease-in-out infinite" }} />
        </div>
        <div style={{ fontSize: "0.7rem", color: "var(--text-dim)", fontStyle: "italic" }}>qwen3</div>
        <div style={{ fontSize: "0.62rem", color: isOnline ? "var(--green)" : "var(--red)", fontFamily: "'Special Elite', monospace", letterSpacing: "0.1em" }}>
          {isOnline ? "ONLINE" : "OFFLINE"}
        </div>
      </div>

      {/* Status */}
      <div style={{ border: "1px solid var(--border)", padding: "11px 13px", background: "var(--panel)", fontFamily: "'Special Elite', monospace", fontSize: "0.67rem", color: "var(--text-dim)", display: "flex", flexDirection: "column", gap: 8, flexShrink: 0 }}>
        <div style={{ color: "var(--frame-gold)", fontSize: "0.54rem", letterSpacing: "0.18em", marginBottom: 2, fontFamily: "'Cinzel Decorative', serif" }}>✦ Status</div>
        {[
          ["LLM",    isOnline ? "connected" : "-",       isOnline ? "var(--green)" : "var(--red)"],
          ["AGENT",  featureFlags?.mcp ? "active" : "off", featureFlags?.mcp ? "var(--purple)" : "var(--text-muted)"],
          ["MEMORY", "In-context",                        "var(--gold)"],
          ["AVATAR", featureFlags?.avatar ? "On" : "Off", featureFlags?.avatar ? "var(--gold-bright)" : "var(--text-muted)"],
          ["ASR",    featureFlags?.asr ? "Whisper" : "Off", featureFlags?.asr ? "var(--rose)" : "var(--text-muted)"],
          ["TTS",    featureFlags?.tts ? "Kokoro" : "Off",  featureFlags?.tts ? "var(--rose)" : "var(--text-muted)"],
          ["RAG",    featureFlags?.rag ? `${ragDocs} docs` : "Off", featureFlags?.rag ? "var(--green)" : "var(--text-muted)"],
          ["WAKE",   featureFlags?.wake_word ? (wakeActive ? "listening" : "ready") : "Off", featureFlags?.wake_word ? (wakeActive ? "var(--green)" : "var(--gold-dim)") : "var(--text-muted)"],
        ].map(([label, value, col]) => (
          <div key={label} style={{ display: "flex", justifyContent: "space-between", gap: 4 }}>
            <span style={{ color: "var(--text-muted)", fontStyle: "italic" }}>{label}</span>
            <span style={{ color: col, textAlign: "right" }}>{value}</span>
          </div>
        ))}
      </div>

      {/* Knowledge base */}
      {featureFlags?.rag && (
        <button onClick={onOpenRAG}
          style={{ border: "1px solid var(--border)", padding: "11px 13px", background: "var(--panel)", fontFamily: "'Special Elite', monospace", fontSize: "0.67rem", color: "var(--text-dim)", display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer", textAlign: "left", transition: "all 0.2s", width: "100%", flexShrink: 0 }}
          onMouseEnter={e => { e.currentTarget.style.borderColor = "var(--frame-gold)"; e.currentTarget.style.color = "var(--gold)"; }}
          onMouseLeave={e => { e.currentTarget.style.borderColor = "var(--border)"; e.currentTarget.style.color = "var(--text-dim)"; }}>
          <span style={{ fontFamily: "'Cinzel Decorative', serif", fontSize: "0.54rem", letterSpacing: "0.18em", color: "inherit" }}>✦ Knowledge</span>
          <span style={{ fontSize: "0.6rem" }}>{ragDocs} docs ›</span>
        </button>
      )}

      {/* Auth connections */}
      <div style={{ border: "1px solid var(--border)", padding: "11px 13px", background: "var(--panel)", fontFamily: "'Special Elite', monospace", fontSize: "0.67rem", color: "var(--text-dim)", display: "flex", flexDirection: "column", gap: 8, flexShrink: 0 }}>
        <div style={{ color: "var(--frame-gold)", fontSize: "0.54rem", letterSpacing: "0.18em", marginBottom: 2, fontFamily: "'Cinzel Decorative', serif" }}>✦ Links</div>
        <AuthConnectBtn label="Google" connected={!!authStatus?.google} onLogin={() => onAuthLogin("google")} onLogout={() => onAuthLogout("google")} />
        <AuthConnectBtn label="Spotify" connected={!!authStatus?.spotify} onLogin={() => onAuthLogin("spotify")} onLogout={() => onAuthLogout("spotify")} />
      </div>

      <div style={{ border: "1px dashed var(--border)", padding: "13px 11px", background: "transparent", textAlign: "center", fontFamily: "'Cinzel Decorative', serif", fontSize: "0.5rem", color: "var(--text-muted)", letterSpacing: "0.14em", lineHeight: 2, flexShrink: 0 }}>
        ✦ Voice Controls ✦<br />
        <span style={{ fontFamily: "'Special Elite', monospace", fontSize: "0.6rem", opacity: 0.45 }}>ASR &amp; TTS</span>
      </div>

      {wakeActive && (
        <div style={{ border: "1px solid var(--green)", padding: "9px 13px", background: "var(--panel)", fontFamily: "'Special Elite', monospace", fontSize: "0.62rem", color: "var(--green)", textAlign: "center", letterSpacing: "0.1em", animation: "wake-pulse 1.5s infinite", flexShrink: 0 }}>
          ◎ wake listening...
        </div>
      )}

      {streaming && !wakeActive && (
        <div style={{ border: "1px solid var(--border-glow)", padding: "9px 13px", background: "var(--panel)", fontFamily: "'Special Elite', monospace", fontSize: "0.62rem", color: "var(--gold)", textAlign: "center", letterSpacing: "0.1em", animation: "pulse-border 1.5s infinite", flexShrink: 0 }}>
          ✦ responding...
        </div>
      )}
    </div>
  );
}

// IconBtn
function IconBtn({ onClick, disabled, title, children, active, style: extra }) {
  const activeRef = useRef(active);
  useEffect(() => { activeRef.current = active; }, [active]);
  return (
    <button onClick={onClick} disabled={disabled} title={title}
      style={{ width: 38, height: 38, flexShrink: 0, background: "transparent", border: `1px ${active ? "solid" : "dashed"} ${active ? "var(--frame-gold)" : "var(--border)"}`, borderRadius: 3, color: active ? "var(--gold)" : "var(--text-muted)", cursor: disabled ? "not-allowed" : "pointer", fontSize: "1rem", display: "flex", alignItems: "center", justifyContent: "center", transition: "all 0.22s", boxShadow: active ? "0 0 8px rgba(212,168,56,0.2)" : "none", ...extra }}
      onMouseEnter={e => { if (!disabled) { e.currentTarget.style.borderColor = "var(--frame-gold)"; e.currentTarget.style.color = "var(--gold)"; e.currentTarget.style.borderStyle = "solid"; } }}
      onMouseLeave={e => { if (!disabled && !activeRef.current) { e.currentTarget.style.borderColor = "var(--border)"; e.currentTarget.style.color = "var(--text-muted)"; e.currentTarget.style.borderStyle = "dashed"; } }}>
      {children}
    </button>
  );
}

// Main App
export default function App() {
  injectCSS(GLOBAL_CSS);

  const [messages,     setMessages]     = useState([]);
  const [input,        setInput]        = useState("");
  const [streaming,    setStreaming]     = useState(false);
  const [status,       setStatus]       = useState("checking");
  const [featureFlags, setFeatureFlags] = useState({});
  const [showTyping,   setShowTyping]   = useState(false);
  const [authStatus,   setAuthStatus]   = useState({});
  const [showRAG,      setShowRAG]      = useState(false);
  const [ragDocs,      setRagDocs]      = useState(0);
  const [wakeActive,   setWakeActive]   = useState(false);

  const [isRecording, setIsRecording] = useState(false);
  const [asrPartial,  setAsrPartial]  = useState("");
  const asrRef = useRef(null);

  const [isMuted,  setIsMuted]  = useState(false);
  const isMutedRef  = useRef(false);
  const ttsStreamer  = useRef(null);

  const messagesEndRef = useRef(null);
  const inputRef       = useRef(null);
  const wakeSSERef     = useRef(null);  // SSE connection for wake-word events

  const { src: aiAvatarSrc,   processing: aiAvatarProcessing }   = useProcessedAvatar(DEFAULT_AI_IMAGE_URL);
  const { src: userAvatarSrc, processing: userAvatarProcessing } = useProcessedAvatar(DEFAULT_USER_IMAGE_URL);

  const scrollToBottom = useCallback(() => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, []);
  useEffect(() => { scrollToBottom(); }, [messages, showTyping]);
  useEffect(() => { isMutedRef.current = isMuted; }, [isMuted]);

  useEffect(() => {
    ttsStreamer.current = new TTSSentenceStreamer(API_BASE, () => isMutedRef.current);
    return () => ttsStreamer.current?.stop();
  }, []);

  // Health check + auth status + RAG doc count
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.json())
      .then(data => {
        setStatus(data.status === "ok" || data.status === "degraded" ? "online" : "offline");
        const flags = { ...data.feature_flags, mcp: true };
        setFeatureFlags(flags);
        // Load RAG doc count if RAG is enabled
        if (flags.rag) {
          fetch(`${API_BASE}/rag/documents`)
            .then(r => r.json())
            .then(d => setRagDocs(d.total || 0))
            .catch(() => {});
        }
      })
      .catch(() => setStatus("offline"));

    fetch(`${API_BASE}/auth/status`)
      .then(r => r.json())
      .then(data => setAuthStatus(data.providers || {}))
      .catch(() => {});

    // Handle OAuth callback redirect
    const params = new URLSearchParams(window.location.search);
    const authProvider = params.get("auth");
    const authResult   = params.get("status");
    if (authProvider && authResult) {
      if (authResult === "ok") {
        fetch(`${API_BASE}/auth/status`)
          .then(r => r.json())
          .then(data => setAuthStatus(data.providers || {}))
          .catch(() => {});
      }
      window.history.replaceState({}, "", window.location.pathname);
    }

    setMessages([{ id: crypto.randomUUID(), role: "assistant", ts: now(), content: "System ready.\n" }]);
  }, []);

  // Wake-word SSE subscription
  // The backend pushes {"type":"wake_detected"} when the wake word fires.
  // On receipt: show visual indicator + auto-start ASR.
  useEffect(() => {
    if (!featureFlags?.wake_word) return;
    const es = new EventSource(`${API_BASE}/audio/wake/events`);
    wakeSSERef.current = es;
    es.onmessage = (evt) => {
      try {
        const event = JSON.parse(evt.data);
        if (event.type === "wake_detected") {
          setWakeActive(true);
          // Auto-start ASR after a brief visual flash (300ms)
          setTimeout(() => { setWakeActive(false); startRecording(); }, 300);
        } else if (event.type === "wake_listening") {
          setWakeActive(true);
        } else if (event.type === "wake_stopped") {
          setWakeActive(false);
        }
      } catch { }
    };
    es.onerror = () => { setWakeActive(false); };
    return () => { es.close(); wakeSSERef.current = null; };
  }, [featureFlags?.wake_word]);

  const handleAuthLogin  = useCallback(p => { window.location.href = `${API_BASE}/auth/${p}/login`; }, []);
  const handleAuthLogout = useCallback(async p => {
    await fetch(`${API_BASE}/auth/${p}/logout`, { method: "DELETE" }).catch(() => {});
    setAuthStatus(prev => ({ ...prev, [p]: false }));
  }, []);

  // When RAG modal closes, refresh doc count
  const handleCloseRAG = useCallback(async () => {
    setShowRAG(false);
    try {
      const r = await fetch(`${API_BASE}/rag/documents`);
      if (r.ok) { const d = await r.json(); setRagDocs(d.total || 0); }
    } catch { }
  }, []);

  // ASR start
  const startRecording = useCallback(async () => {
    if (asrRef.current) return;
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 } });
      let audioCtx;
      try { audioCtx = new AudioContext({ sampleRate: ASR_SAMPLE_RATE }); }
      catch { audioCtx = new AudioContext(); }

      const ws = new WebSocket(`${WS_BASE}/audio/asr/stream?session_id=${SESSION_ID}`);
      ws.binaryType = "arraybuffer";
      ws.onopen  = () => console.debug("ASR WS open");
      ws.onclose = () => console.debug("ASR WS closed");
      ws.onerror = e => console.warn("ASR WS error:", e);
      ws.onmessage = evt => {
        try {
          const event = JSON.parse(evt.data);
          if (event.type === "partial" && event.text) setAsrPartial(event.text);
          else if (event.type === "final" && event.text) {
            setInput(event.text); setAsrPartial(""); stopRecording();
          }
        } catch { }
      };

      const source    = audioCtx.createMediaStreamSource(mediaStream);
      const processor = audioCtx.createScriptProcessor(ASR_BUFFER_SIZE, 1, 1);
      const sinkNode  = audioCtx.createMediaStreamDestination();
      let sampleBuf   = new Int16Array(0);

      processor.onaudioprocess = e => {
        const float32 = e.inputBuffer.getChannelData(0);
        let samples;
        if (audioCtx.sampleRate !== ASR_SAMPLE_RATE) {
          const ratio = audioCtx.sampleRate / ASR_SAMPLE_RATE;
          const outLen = Math.floor(float32.length / ratio);
          const ds = new Float32Array(outLen);
          for (let i = 0; i < outLen; i++) ds[i] = float32[Math.round(i * ratio)];
          samples = float32ToInt16(ds);
        } else { samples = float32ToInt16(float32); }
        const merged = new Int16Array(sampleBuf.length + samples.length);
        merged.set(sampleBuf); merged.set(samples, sampleBuf.length); sampleBuf = merged;
        while (sampleBuf.length >= ASR_CHUNK_SAMPLES) {
          const chunk = sampleBuf.slice(0, ASR_CHUNK_SAMPLES); sampleBuf = sampleBuf.slice(ASR_CHUNK_SAMPLES);
          if (ws.readyState === WebSocket.OPEN) ws.send(chunk.buffer);
        }
      };

      source.connect(processor); processor.connect(sinkNode);
      asrRef.current = { ws, audioCtx, mediaStream, processor, sinkNode, source };
      setIsRecording(true); setAsrPartial("");
    } catch (err) { console.error("Failed to start recording:", err); setIsRecording(false); }
  }, []);

  // ASR stop
  const stopRecording = useCallback(() => {
    const asr = asrRef.current;
    if (!asr) return;
    try { asr.source?.disconnect(); asr.processor?.disconnect(); asr.ws?.close(); asr.mediaStream?.getTracks().forEach(t => t.stop()); asr.audioCtx?.close(); } catch { }
    asrRef.current = null; setIsRecording(false); setAsrPartial("");
  }, []);

  const toggleRecording = useCallback(() => { if (isRecording) stopRecording(); else startRecording(); }, [isRecording, startRecording, stopRecording]);

  const toggleMute = useCallback(() => {
    setIsMuted(m => { const next = !m; if (next) ttsStreamer.current?.stop(); return next; });
  }, []);

  // Send message
  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || streaming) return;
    setMessages(prev => [...prev, { id: crypto.randomUUID(), role: "user", content: text, ts: now() }]);
    setInput(""); setShowTyping(true);
    ttsStreamer.current?.reset();
    const ttsEnabled = featureFlags?.tts && !isMutedRef.current;
    const aiId = crypto.randomUUID();
    let aiContent = ""; let aiThinking = ""; let toolCalls = {};

    const upsertAI = patch => {
      setMessages(prev => {
        const exists = prev.find(m => m.id === aiId);
        if (exists) return prev.map(m => m.id === aiId ? { ...m, ...patch } : m);
        return [...prev, { id: aiId, role: "assistant", content: "", streaming: true, toolCalls: [], thinking: "", ts: now(), ...patch }];
      });
    };

    try {
      setStreaming(true);
      const res = await fetch(`${API_BASE}/agent/stream`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: SESSION_ID, message: text }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setShowTyping(false);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n"); buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          let event;
          try { event = JSON.parse(line.slice(6)); } catch { continue; }

          switch (event.type) {
            case "delta":
              if (event.content) {
                aiContent += event.content;
                upsertAI({ content: aiContent, streaming: true, toolCalls: Object.values(toolCalls), thinking: aiThinking || undefined });
                if (ttsEnabled) ttsStreamer.current?.push(event.content);
              }
              break;
            case "thinking":
              if (event.content) { aiThinking = event.content; upsertAI({ thinking: aiThinking, toolCalls: Object.values(toolCalls) }); }
              break;
            case "tool_start":
              toolCalls[event.call_id] = { call_id: event.call_id, tool_name: event.tool_name, done: false, success: false };
              upsertAI({ toolCalls: Object.values(toolCalls), streaming: true }); break;
            case "tool_done":
              if (toolCalls[event.call_id]) { toolCalls[event.call_id].done = true; toolCalls[event.call_id].success = event.success; }
              upsertAI({ toolCalls: Object.values(toolCalls), streaming: true }); break;
            case "done":
              if (ttsEnabled) ttsStreamer.current?.flush();
              upsertAI({ streaming: false, toolCalls: Object.values(toolCalls) }); break;
            case "error":
              upsertAI({ content: aiContent || `⚠ ${event.error}`, streaming: false }); break;
          }
        }
      }
    } catch (err) {
      if (err.name !== "AbortError") {
        setShowTyping(false);
        setMessages(prev => [...prev, { id: crypto.randomUUID(), role: "assistant", ts: now(), content: "⚠ Connection failed. Is the backend running at " + API_BASE + "?" }]);
      }
    } finally { setStreaming(false); setShowTyping(false); inputRef.current?.focus(); }
  }, [input, streaming, featureFlags?.tts]);

  const handleKeyDown = useCallback(e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  }, [sendMessage]);

  const clearSession = useCallback(async () => {
    ttsStreamer.current?.stop();
    await fetch(`${API_BASE}/chat/session/${SESSION_ID}`, { method: "DELETE" }).catch(() => {});
    setMessages([{ id: crypto.randomUUID(), role: "assistant", ts: now(), content: "Memory cleared. Starting a fresh session." }]);
  }, []);

  const canSend = !streaming && !!input.trim();

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column", background: "var(--bg)", position: "relative", zIndex: 1 }}>

      {/* RAG modal */}
      {showRAG && <RAGModal onClose={handleCloseRAG} />}

      {/* Top bar */}
      <div style={{ height: 54, borderBottom: "1px solid var(--frame-gold)", display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 24px", background: "var(--bg2)", flexShrink: 0, boxShadow: "0 2px 26px rgba(212,168,56,0.1)", position: "relative", zIndex: 2 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 15 }}>
          <span style={{ color: "var(--frame-gold)", fontSize: "0.95rem", opacity: 0.7 }}>🙜</span>
          <span style={{ fontFamily: "'UnifrakturMaguntia', serif", fontSize: "1.5rem", color: "var(--gold)", animation: "gold-pulse 4s ease-in-out infinite" }}>Crow</span>
          <span style={{ width: 1, height: 20, background: "var(--border)" }} />
          <span style={{ fontFamily: "'Special Elite', monospace", fontSize: "0.65rem", color: "var(--text-muted)", letterSpacing: "0.12em", fontStyle: "italic" }}>
            Session · {SESSION_ID.slice(0, 8).toUpperCase()}
          </span>
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          {(aiAvatarProcessing || userAvatarProcessing) && (
            <div style={{ display: "flex", alignItems: "center", gap: 5, fontFamily: "'Special Elite', monospace", fontSize: "0.6rem", color: "var(--frame-gold)", letterSpacing: "0.1em", animation: "proc-pulse 1s ease-in-out infinite", border: "1px solid var(--border)", padding: "3px 10px", borderRadius: 2 }}>
              <div style={{ width: 10, height: 10, border: "1.5px solid rgba(212,168,56,0.3)", borderTop: "1.5px solid var(--frame-gold)", borderRadius: "50%", animation: "spin 0.8s linear infinite", flexShrink: 0 }} />
              ✦ LOADING...
            </div>
          )}
          {featureFlags?.tts && (
            <button onClick={toggleMute} title={isMuted ? "Unmute" : "Mute"}
              style={{ fontFamily: "'Cinzel Decorative', serif", fontSize: "0.52rem", background: "transparent", border: "1px solid var(--border)", color: isMuted ? "var(--text-muted)" : "var(--rose)", padding: "4px 12px", borderRadius: 2, cursor: "pointer", letterSpacing: "0.12em", transition: "all 0.25s" }}
              onMouseEnter={e => { e.target.style.borderColor = "var(--frame-gold)"; e.target.style.color = "var(--gold)"; e.target.style.boxShadow = "0 0 12px rgba(212,168,56,0.25)"; }}
              onMouseLeave={e => { e.target.style.borderColor = "var(--border)"; e.target.style.color = isMuted ? "var(--text-muted)" : "var(--rose)"; e.target.style.boxShadow = "none"; }}>
              {isMuted ? "🔇-_- Muted" : "🔊 Voice ON"}
            </button>
          )}
          <span style={{ fontFamily: "'Special Elite', monospace", fontSize: "0.65rem", color: status === "online" ? "var(--green)" : "var(--red)", letterSpacing: "0.1em" }}>● {status.toUpperCase()}</span>
          <button onClick={clearSession}
            style={{ fontFamily: "'Cinzel Decorative', serif", fontSize: "0.52rem", background: "transparent", border: "1px solid var(--border)", color: "var(--text-dim)", padding: "4px 14px", borderRadius: 2, cursor: "pointer", letterSpacing: "0.12em", transition: "all 0.25s" }}
            onMouseEnter={e => { e.target.style.borderColor = "var(--frame-gold)"; e.target.style.color = "var(--gold)"; e.target.style.boxShadow = "0 0 12px rgba(212,168,56,0.25)"; }}
            onMouseLeave={e => { e.target.style.borderColor = "var(--border)"; e.target.style.color = "var(--text-dim)"; e.target.style.boxShadow = "none"; }}>
            Erase Memory
          </button>
          <span style={{ color: "var(--frame-gold)", fontSize: "0.95rem", opacity: 0.7 }}>🙞</span>
        </div>
      </div>

      {/* Main */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        <SystemPanel
          status={status} streaming={streaming} featureFlags={featureFlags}
          authStatus={authStatus} ragDocs={ragDocs} wakeActive={wakeActive}
          onAuthLogin={handleAuthLogin} onAuthLogout={handleAuthLogout}
          onOpenRAG={() => setShowRAG(true)}
        />

        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          <div style={{ flex: 1, overflowY: "auto", padding: "32px 36px 14px", display: "flex", flexDirection: "column" }}>
            {messages.map(msg => (
              <MessageBubble key={msg.id} msg={msg} status={status}
                avatarSrc={msg.role === "user" ? userAvatarSrc : aiAvatarSrc}
                isAvatarProcessing={msg.role === "user" ? userAvatarProcessing : aiAvatarProcessing}
              />
            ))}
            {showTyping && <TypingIndicator avatarSrc={aiAvatarSrc} isAvatarProcessing={aiAvatarProcessing} />}
            <div ref={messagesEndRef} />
          </div>

          {asrPartial && (
            <div style={{ padding: "4px 36px 0", fontFamily: "'Special Elite', monospace", fontSize: "0.78rem", fontStyle: "italic", color: "var(--rose)", letterSpacing: "0.05em" }}>
              ◎ {asrPartial}
            </div>
          )}

          <div style={{ borderTop: "1px solid var(--border)", padding: "13px 20px", background: "var(--bg2)", display: "flex", gap: 9, alignItems: "flex-end", boxShadow: "0 -2px 26px rgba(10,0,5,0.6)" }}>
            <div style={{ flex: 1 }}>
              <textarea ref={inputRef} value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={streaming}
                placeholder={isRecording ? "Listening... speak now" : streaming ? "responding.." : "Write some sins but no tragedies ..."}
                rows={1}
                style={{ width: "100%", background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 3, padding: "11px 17px", color: "var(--text)", fontFamily: "'Special Elite', monospace", fontSize: "1.04rem", lineHeight: 1.65, resize: "none", outline: "none", transition: "border-color 0.25s, box-shadow 0.25s", minHeight: 44, maxHeight: 140, overflow: "auto" }}
                onFocus={e => { e.target.style.borderColor = "var(--frame-gold)"; e.target.style.boxShadow = "0 0 16px rgba(212,168,56,0.18)"; }}
                onBlur={e  => { e.target.style.borderColor = "var(--border)"; e.target.style.boxShadow = "none"; }}
                onInput={e => { e.target.style.height = "auto"; e.target.style.height = Math.min(e.target.scrollHeight, 140) + "px"; }} />
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 6, flexShrink: 0 }}>
              <IconBtn title="Use tools (coming soon)" disabled>&</IconBtn>
              <IconBtn title="Attach file (coming soon)" disabled>+</IconBtn>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 6, flexShrink: 0 }}>
              <button onClick={featureFlags?.asr ? toggleRecording : undefined}
                title={featureFlags?.asr ? (isRecording ? "Stop recording" : "Start voice input") : "ASR disabled"}
                style={{ width: 38, height: 38, flexShrink: 0, borderRadius: 3, cursor: featureFlags?.asr ? "pointer" : "not-allowed", display: "flex", alignItems: "center", justifyContent: "center", fontSize: isRecording ? "0.75rem" : "0.9rem", background: isRecording ? "rgba(232,84,84,0.15)" : "transparent", border: `1px ${featureFlags?.asr ? "solid" : "dashed"} ${isRecording ? "var(--red)" : featureFlags?.asr ? "var(--frame-gold)" : "var(--border)"}`, color: isRecording ? "var(--red)" : featureFlags?.asr ? "var(--rose)" : "var(--text-muted)", animation: isRecording ? "rec-pulse 1.2s ease-in-out infinite" : "none", transition: "all 0.2s" }}>
                {isRecording ? "⏹" : "◎"}
              </button>

              <button onClick={sendMessage} disabled={!canSend} title="Send message"
                style={{ width: 38, height: 38, flexShrink: 0, background: canSend ? "var(--gold-dim)" : "transparent", border: `1px solid ${canSend ? "var(--frame-gold)" : "var(--border)"}`, borderRadius: 3, color: canSend ? "var(--gold)" : "var(--text-muted)", cursor: canSend ? "pointer" : "not-allowed", fontSize: "1rem", display: "flex", alignItems: "center", justifyContent: "center", transition: "all 0.25s", boxShadow: canSend ? "0 0 12px rgba(212,168,56,0.25)" : "none" }}
                onMouseEnter={e => { if (canSend) { e.currentTarget.style.background = "var(--crimson)"; e.currentTarget.style.boxShadow = "0 0 26px rgba(212,168,56,0.5)"; e.currentTarget.style.color = "var(--gold-bright)"; } }}
                onMouseLeave={e => { if (canSend) { e.currentTarget.style.background = "var(--gold-dim)"; e.currentTarget.style.boxShadow = "0 0 12px rgba(212,168,56,0.25)"; e.currentTarget.style.color = "var(--gold)"; } }}>
                ▶
              </button>
            </div>
          </div>

          <div style={{ padding: "4px 24px", borderTop: "1px solid var(--border)", background: "var(--bg)", display: "flex", justifyContent: "space-between", alignItems: "center", fontFamily: "'Special Elite', monospace", fontSize: "0.62rem", color: "var(--text-muted)", letterSpacing: "0.08em" }}>
            <span style={{ fontStyle: "italic", display: "flex", alignItems: "center", gap: 5 }}>
              {wakeActive ? "◎ wake word detected" : isRecording ? "◎ recording" : streaming ? "✦ processing..." : "■ ready"}
              {(aiAvatarProcessing || userAvatarProcessing) && <span style={{ color: "var(--frame-gold)", animation: "proc-pulse 1s ease-in-out infinite" }}>loading..</span>}
            </span>
            <span>crow ver. 0.2.0</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function now() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}