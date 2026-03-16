import { useState, useEffect, useRef, useCallback } from "react";

// ── Config ────────────────────────────────────────────────────────────────────
const API_BASE = "http://localhost:8000/api/v1";
const SESSION_ID = crypto.randomUUID();

// ── Tool name → human label ───────────────────────────────────────────────────
const TOOL_LABELS = {
  list_calendar_events:   "Checking calendar…",
  get_calendar_event:     "Reading event…",
  create_calendar_event:  "Creating event…",
  update_calendar_event:  "Updating event…",
  delete_calendar_event:  "Deleting event…",
};
const toolLabel = (name) => TOOL_LABELS[name] ?? `Running ${name}…`;

// ── CSS injection ─────────────────────────────────────────────────────────────
const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&family=Exo+2:ital,wght@0,300;0,400;1,300&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:          #080b10;
    --bg2:         #0d1117;
    --panel:       #0f1520;
    --border:      #1c2d45;
    --border-glow: #1e4a72;
    --amber:       #e8a045;
    --amber-dim:   #7a4d1a;
    --cyan:        #3dd9eb;
    --cyan-dim:    #0d4a52;
    --green:       #39d98a;
    --red:         #e85454;
    --purple:      #a78bfa;
    --purple-dim:  #2d1f5e;
    --text:        #c9d8e8;
    --text-dim:    #4a6a85;
    --text-muted:  #2a4055;
    --user-bg:     #0a1828;
    --ai-bg:       #091420;
    --tool-bg:     #0e0e1f;
    --scrollbar:   #1a2d42;
  }

  html, body, #root { height: 100%; width: 100%; background: var(--bg); color: var(--text);
    font-family: 'Exo 2', sans-serif; overflow: hidden; }

  body::after { content: ''; position: fixed; inset: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px,
      rgba(0,0,0,0.06) 2px, rgba(0,0,0,0.06) 4px);
    pointer-events: none; z-index: 9999; }

  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--scrollbar); border-radius: 2px; }

  @keyframes fadeSlideIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
  @keyframes pulse-border { 0%,100% { border-color:var(--border); } 50% { border-color:var(--border-glow); box-shadow:0 0 12px rgba(30,74,114,0.4); } }
  @keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0; } }
  @keyframes avatar-breathe { 0%,100% { transform:scale(1); filter:brightness(1); } 50% { transform:scale(1.01); filter:brightness(1.05); } }
  @keyframes typing-dot { 0%,80%,100% { transform:scale(0.6); opacity:0.4; } 40% { transform:scale(1); opacity:1; } }
  @keyframes status-glow { 0%,100% { box-shadow:0 0 4px currentColor; } 50% { box-shadow:0 0 12px currentColor; } }
  @keyframes scan-line { from { transform:translateY(-100%); } to { transform:translateY(400px); } }
  @keyframes tool-pulse { 0%,100% { opacity:0.7; } 50% { opacity:1; } }
  @keyframes spin { from { transform:rotate(0deg); } to { transform:rotate(360deg); } }
`;

function injectCSS(css) {
  if (document.getElementById("app-styles")) return;
  const el = document.createElement("style");
  el.id = "app-styles";
  el.textContent = css;
  document.head.appendChild(el);
}

// ── Sub-components ────────────────────────────────────────────────────────────

function Timestamp({ ts }) {
  return (
    <span style={{ fontSize: "0.62rem", color: "var(--text-muted)",
      fontFamily: "'Share Tech Mono', monospace", marginTop: 4, display: "block" }}>
      {ts}
    </span>
  );
}

// Tool-use pill shown inline in the message stream
function ToolCallBubble({ toolName, done, success }) {
  const label = toolLabel(toolName);
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 8,
      padding: "6px 12px", borderRadius: 4,
      background: "var(--tool-bg)",
      border: `1px solid ${done ? (success ? "var(--green)" : "var(--red)") : "var(--purple)"}`,
      fontSize: "0.72rem", fontFamily: "'Share Tech Mono', monospace",
      color: done ? (success ? "var(--green)" : "var(--red)") : "var(--purple)",
      animation: "fadeSlideIn 0.15s ease both",
      maxWidth: "60%",
    }}>
      {/* Spinner while running, checkmark/x when done */}
      <span style={{
        display: "inline-block",
        animation: done ? "none" : "spin 1s linear infinite",
        fontSize: "0.7rem",
      }}>
        {done ? (success ? "✓" : "✗") : "◌"}
      </span>
      <span>{done ? (success ? label.replace("…", " done") : `${label.replace("…","")} failed`) : label}</span>
    </div>
  );
}

// Thinking text shown between tool calls
function ThinkingBubble({ content }) {
  return (
    <div style={{
      padding: "8px 14px",
      borderRadius: 4,
      background: "transparent",
      border: "1px dashed var(--border)",
      borderLeft: "2px solid var(--purple)",
      fontSize: "0.8rem", fontStyle: "italic",
      color: "var(--text-dim)", maxWidth: "80%",
      animation: "fadeSlideIn 0.15s ease both",
    }}>
      {content}
    </div>
  );
}

function MessageBubble({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div style={{
      display: "flex", flexDirection: "column",
      alignItems: isUser ? "flex-end" : "flex-start",
      animation: "fadeSlideIn 0.2s ease both",
      marginBottom: 2, gap: 6,
    }}>
      {/* Tool calls rendered before the text content */}
      {msg.toolCalls?.map((tc) => (
        <ToolCallBubble
          key={tc.call_id}
          toolName={tc.tool_name}
          done={tc.done}
          success={tc.success}
        />
      ))}

      {/* Thinking text */}
      {msg.thinking && <ThinkingBubble content={msg.thinking} />}

      {/* Main text bubble — only render when there's content */}
      {(msg.content || msg.streaming) && (
        <div style={{
          maxWidth: "80%", padding: "10px 14px",
          borderRadius: isUser ? "8px 2px 8px 8px" : "2px 8px 8px 8px",
          background: isUser ? "var(--user-bg)" : "var(--ai-bg)",
          border: `1px solid ${isUser ? "var(--border-glow)" : "var(--border)"}`,
          borderLeft: !isUser ? "3px solid var(--cyan)" : undefined,
          borderRight: isUser ? "3px solid var(--amber)" : undefined,
          fontSize: "0.88rem", lineHeight: 1.65,
          color: "var(--text)", fontWeight: 300,
        }}>
          {!isUser && (
            <div style={{ fontFamily: "'Orbitron', monospace", fontSize: "0.55rem",
              color: "var(--cyan)", letterSpacing: "0.2em", marginBottom: 6 }}>
              ARIA
            </div>
          )}
          <div style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
            {msg.content}
            {msg.streaming && (
              <span style={{ display: "inline-block", width: 8, height: 14,
                background: "var(--cyan)", marginLeft: 2, verticalAlign: "middle",
                animation: "blink 0.8s step-start infinite", opacity: 0.85 }} />
            )}
          </div>
          <Timestamp ts={msg.ts} />
        </div>
      )}
    </div>
  );
}

function TypingIndicator() {
  return (
    <div style={{ display: "flex", gap: 5, padding: "10px 16px",
      background: "var(--ai-bg)", border: "1px solid var(--border)",
      borderLeft: "3px solid var(--cyan)", borderRadius: "2px 8px 8px 8px",
      width: "fit-content", alignItems: "center" }}>
      {[0, 0.2, 0.4].map((d, i) => (
        <div key={i} style={{ width: 7, height: 7, borderRadius: "50%",
          background: "var(--cyan)",
          animation: `typing-dot 1.2s ${d}s ease-in-out infinite` }} />
      ))}
    </div>
  );
}

function AvatarPanel({ status, streaming, featureFlags }) {
  const isOnline = status === "online";
  return (
    <div style={{ width: 280, flexShrink: 0, display: "flex", flexDirection: "column",
      gap: 12, padding: "16px 12px" }}>

      {/* Avatar frame */}
      <div style={{ position: "relative", width: "100%", aspectRatio: "1/1",
        border: "1px solid var(--border)", borderRadius: 4, background: "var(--panel)",
        overflow: "hidden",
        animation: streaming ? "pulse-border 1.5s infinite" : "none",
        boxShadow: isOnline ? "0 0 24px rgba(61,217,235,0.06) inset" : "none" }}>

        {/* Corner decorations */}
        {[["top","left"],["top","right"],["bottom","left"],["bottom","right"]].map(([v,h],i) => (
          <div key={i} style={{ position: "absolute", [v]: 0, [h]: 0, width: 12, height: 12,
            borderTop: v === "top" ? "2px solid var(--cyan)" : "none",
            borderBottom: v === "bottom" ? "2px solid var(--cyan)" : "none",
            borderLeft: h === "left" ? "2px solid var(--cyan)" : "none",
            borderRight: h === "right" ? "2px solid var(--cyan)" : "none" }} />
        ))}

        {streaming && (
          <div style={{ position: "absolute", left: 0, right: 0, height: 2,
            background: "linear-gradient(90deg, transparent, var(--cyan), transparent)",
            opacity: 0.4, animation: "scan-line 2s linear infinite" }} />
        )}

        <div style={{ position: "absolute", inset: 0, display: "flex",
          flexDirection: "column", alignItems: "center", justifyContent: "center",
          animation: isOnline && !streaming ? "avatar-breathe 4s ease-in-out infinite" : "none" }}>
          <div style={{ fontSize: "5rem", lineHeight: 1, userSelect: "none",
            filter: streaming ? "brightness(1.3) drop-shadow(0 0 12px var(--cyan))" : "brightness(0.8)",
            transition: "filter 0.5s ease" }}>
            {streaming ? "◈" : "◇"}
          </div>
          <div style={{ fontFamily: "'Orbitron', monospace", fontSize: "0.55rem",
            color: "var(--text-muted)", letterSpacing: "0.3em", marginTop: 8 }}>
            IMG GEN · PHASE 4
          </div>
        </div>
        <div style={{ position: "absolute", inset: 0,
          background: "radial-gradient(ellipse at center, transparent 40%, rgba(8,11,16,0.7) 100%)",
          pointerEvents: "none" }} />
      </div>

      {/* Name plate */}
      <div style={{ border: "1px solid var(--border)", borderRadius: 4,
        padding: "8px 12px", background: "var(--panel)", display: "flex",
        flexDirection: "column", gap: 4 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span style={{ fontFamily: "'Orbitron', monospace", fontSize: "0.85rem",
            fontWeight: 700, color: "var(--amber)", letterSpacing: "0.1em" }}>ARIA</span>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 6, height: 6, borderRadius: "50%",
              background: isOnline ? "var(--green)" : "var(--red)",
              color: isOnline ? "var(--green)" : "var(--red)",
              animation: "status-glow 2s ease-in-out infinite" }} />
            <span style={{ fontSize: "0.6rem", color: "var(--text-dim)",
              fontFamily: "'Share Tech Mono', monospace" }}>
              {isOnline ? "ONLINE" : "OFFLINE"}
            </span>
          </div>
        </div>
        <div style={{ fontSize: "0.65rem", color: "var(--text-dim)", fontStyle: "italic" }}>
          Local Intelligence Unit v0.1
        </div>
      </div>

      {/* System status */}
      <div style={{ border: "1px solid var(--border)", borderRadius: 4, padding: "10px 12px",
        background: "var(--panel)", fontFamily: "'Share Tech Mono', monospace",
        fontSize: "0.62rem", color: "var(--text-dim)", display: "flex",
        flexDirection: "column", gap: 6 }}>
        <div style={{ color: "var(--cyan)", fontSize: "0.58rem", letterSpacing: "0.15em", marginBottom: 4 }}>
          ■ SYSTEM STATUS
        </div>
        {[
          ["LLM Backend", isOnline ? "connected" : "—",  isOnline ? "var(--green)" : "var(--red)"],
          ["Agent Mode",  featureFlags?.mcp ? "active" : "off", featureFlags?.mcp ? "var(--purple)" : "var(--text-muted)"],
          ["Memory",      "In-context",      "var(--amber)"],
          ["Avatar",      "Phase 4",         "var(--text-muted)"],
          ["TTS",         "Phase 5",         "var(--text-muted)"],
          ["ASR",         "Phase 5",         "var(--text-muted)"],
          ["RAG",         "Phase 7",         "var(--text-muted)"],
        ].map(([label, value, color]) => (
          <div key={label} style={{ display: "flex", justifyContent: "space-between" }}>
            <span style={{ color: "var(--text-muted)" }}>{label}</span>
            <span style={{ color }}>{value}</span>
          </div>
        ))}
      </div>

      {/* Voice controls placeholder */}
      <div style={{ border: "1px dashed var(--border)", borderRadius: 4,
        padding: "10px 12px", textAlign: "center",
        fontFamily: "'Share Tech Mono', monospace", fontSize: "0.58rem",
        color: "var(--text-muted)", letterSpacing: "0.1em" }}>
        ◎ VOICE CONTROLS<br/>
        <span style={{ opacity: 0.5 }}>ASR / TTS · PHASE 5</span>
      </div>
    </div>
  );
}

// ── Main app ──────────────────────────────────────────────────────────────────
export default function App() {
  injectCSS(GLOBAL_CSS);

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [status, setStatus] = useState("checking");
  const [featureFlags, setFeatureFlags] = useState({});
  const [showTyping, setShowTyping] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const abortRef = useRef(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => { scrollToBottom(); }, [messages, showTyping]);

  // Health check on mount
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.json())
      .then(data => {
        setStatus(data.status === "ok" || data.status === "degraded" ? "online" : "offline");
        setFeatureFlags({
          ...data.feature_flags,
          // /health returns avatar/tts/asr/rag — add mcp flag from root
          mcp: true,  // assume true if backend started with mcp_enabled
        });
      })
      .catch(() => setStatus("offline"));

    setMessages([{
      id: crypto.randomUUID(),
      role: "assistant",
      content: "System online. I'm ARIA — your local intelligence interface. How can I assist you?",
      ts: now(),
    }]);
  }, []);

  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || streaming) return;

    const userMsg = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
      ts: now(),
    };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setShowTyping(true);

    // Prepare the AI message slot we'll mutate as events arrive
    const aiId = crypto.randomUUID();
    let aiContent = "";
    let aiThinking = "";
    // toolCalls: { [call_id]: { tool_name, done, success } }
    let toolCalls = {};

    const controller = new AbortController();
    abortRef.current = controller;

    const upsertAiMessage = (patch) => {
      setMessages(prev => {
        const exists = prev.find(m => m.id === aiId);
        if (exists) {
          return prev.map(m => m.id === aiId ? { ...m, ...patch } : m);
        } else {
          return [...prev, {
            id: aiId, role: "assistant", content: "", streaming: true,
            toolCalls: [], thinking: "", ts: now(), ...patch,
          }];
        }
      });
    };

    try {
      setStreaming(true);

      // ── Use /agent/stream — handles both tool-use and plain chat ──────
      const response = await fetch(`${API_BASE}/agent/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: SESSION_ID, message: text }),
        signal: controller.signal,
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      setShowTyping(false);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          let event;
          try { event = JSON.parse(line.slice(6)); } catch { continue; }

          switch (event.type) {

            // ── Standard streaming token ─────────────────────────────
            case "delta":
              if (event.content) {
                aiContent += event.content;
                upsertAiMessage({
                  content: aiContent,
                  streaming: true,
                  toolCalls: Object.values(toolCalls),
                  thinking: aiThinking || undefined,
                });
              }
              break;

            // ── LLM reasoning text before a tool call ────────────────
            case "thinking":
              if (event.content) {
                aiThinking = event.content;
                upsertAiMessage({
                  thinking: aiThinking,
                  toolCalls: Object.values(toolCalls),
                });
              }
              break;

            // ── Tool invocation started ──────────────────────────────
            case "tool_start":
              toolCalls[event.call_id] = {
                call_id: event.call_id,
                tool_name: event.tool_name,
                done: false,
                success: false,
              };
              upsertAiMessage({
                toolCalls: Object.values(toolCalls),
                streaming: true,
              });
              break;

            // ── Tool execution finished ──────────────────────────────
            case "tool_done":
              if (toolCalls[event.call_id]) {
                toolCalls[event.call_id].done = true;
                toolCalls[event.call_id].success = event.success;
              }
              upsertAiMessage({
                toolCalls: Object.values(toolCalls),
                streaming: true,
              });
              break;

            // ── Stream finished ──────────────────────────────────────
            case "done":
              upsertAiMessage({
                streaming: false,
                toolCalls: Object.values(toolCalls),
              });
              break;

            // ── Error ────────────────────────────────────────────────
            case "error":
              upsertAiMessage({
                content: aiContent || `⚠ ${event.error}`,
                streaming: false,
              });
              break;
          }
        }
      }
    } catch (err) {
      if (err.name !== "AbortError") {
        setShowTyping(false);
        setMessages(prev => [...prev, {
          id: crypto.randomUUID(),
          role: "assistant",
          content: "⚠ Connection failed. Ensure the backend is running at " + API_BASE,
          ts: now(),
        }]);
      }
    } finally {
      setStreaming(false);
      setShowTyping(false);
      inputRef.current?.focus();
    }
  }, [input, streaming]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  }, [sendMessage]);

  const clearSession = useCallback(async () => {
    await fetch(`${API_BASE}/chat/session/${SESSION_ID}`, { method: "DELETE" }).catch(() => {});
    setMessages([{
      id: crypto.randomUUID(), role: "assistant",
      content: "Memory cleared. Starting a fresh session.", ts: now(),
    }]);
  }, []);

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column", background: "var(--bg)" }}>

      {/* Top bar */}
      <div style={{ height: 48, borderBottom: "1px solid var(--border)",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "0 20px", background: "var(--bg2)", flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontFamily: "'Orbitron', monospace", fontSize: "0.8rem",
            fontWeight: 900, color: "var(--amber)", letterSpacing: "0.25em" }}>
            LOCAL·AI
          </span>
          <span style={{ width: 1, height: 16, background: "var(--border)" }} />
          <span style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: "0.65rem",
            color: "var(--text-dim)" }}>
            SESSION · {SESSION_ID.slice(0, 8).toUpperCase()}
          </span>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: "0.62rem",
            color: status === "online" ? "var(--green)" : "var(--red)" }}>
            ● {status.toUpperCase()}
          </span>
          <button onClick={clearSession} style={{
            fontFamily: "'Share Tech Mono', monospace", fontSize: "0.6rem",
            background: "transparent", border: "1px solid var(--border)",
            color: "var(--text-dim)", padding: "3px 10px", borderRadius: 3,
            cursor: "pointer", letterSpacing: "0.1em", transition: "all 0.2s" }}
            onMouseEnter={e => { e.target.style.borderColor = "var(--amber)"; e.target.style.color = "var(--amber)"; }}
            onMouseLeave={e => { e.target.style.borderColor = "var(--border)"; e.target.style.color = "var(--text-dim)"; }}>
            CLR MEM
          </button>
        </div>
      </div>

      {/* Main layout */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        <div style={{ borderRight: "1px solid var(--border)", background: "var(--bg2)",
          overflow: "hidden", display: "flex", flexDirection: "column" }}>
          <AvatarPanel status={status} streaming={streaming} featureFlags={featureFlags} />
        </div>

        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          {/* Messages */}
          <div style={{ flex: 1, overflowY: "auto", padding: "20px 24px",
            display: "flex", flexDirection: "column", gap: 10 }}>
            {messages.map(msg => <MessageBubble key={msg.id} msg={msg} />)}
            {showTyping && <TypingIndicator />}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div style={{ borderTop: "1px solid var(--border)", padding: "14px 20px",
            background: "var(--bg2)", display: "flex", gap: 10, alignItems: "flex-end" }}>
            <button title="Voice Input — Phase 5" style={{
              width: 38, height: 38, flexShrink: 0, background: "transparent",
              border: "1px dashed var(--border)", borderRadius: 4,
              color: "var(--text-muted)", cursor: "not-allowed",
              fontSize: "0.9rem", display: "flex", alignItems: "center", justifyContent: "center" }}>
              ◎
            </button>
            <div style={{ flex: 1 }}>
              <textarea ref={inputRef} value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={streaming}
                placeholder={streaming ? "ARIA is responding…" : "Enter message  ↵ send  ⇧↵ newline"}
                rows={1}
                style={{ width: "100%", background: "var(--panel)",
                  border: "1px solid var(--border)", borderRadius: 4,
                  padding: "10px 14px", color: "var(--text)",
                  fontFamily: "'Exo 2', sans-serif", fontSize: "0.88rem",
                  fontWeight: 300, lineHeight: 1.5, resize: "none", outline: "none",
                  transition: "border-color 0.2s", minHeight: 40, maxHeight: 140,
                  overflow: "auto" }}
                onFocus={e => e.target.style.borderColor = "var(--border-glow)"}
                onBlur={e => e.target.style.borderColor = "var(--border)"}
                onInput={e => { e.target.style.height = "auto";
                  e.target.style.height = Math.min(e.target.scrollHeight, 140) + "px"; }} />
            </div>
            <button onClick={sendMessage} disabled={streaming || !input.trim()} style={{
              width: 38, height: 38, flexShrink: 0,
              background: streaming || !input.trim() ? "transparent" : "var(--cyan-dim)",
              border: `1px solid ${streaming || !input.trim() ? "var(--border)" : "var(--cyan)"}`,
              borderRadius: 4,
              color: streaming || !input.trim() ? "var(--text-muted)" : "var(--cyan)",
              cursor: streaming || !input.trim() ? "not-allowed" : "pointer",
              fontSize: "1rem", display: "flex", alignItems: "center", justifyContent: "center",
              transition: "all 0.2s", fontFamily: "'Share Tech Mono', monospace" }}>
              ▶
            </button>
          </div>

          {/* Status bar */}
          <div style={{ padding: "4px 20px", borderTop: "1px solid var(--border)",
            background: "var(--bg)", display: "flex", justifyContent: "space-between",
            fontFamily: "'Share Tech Mono', monospace", fontSize: "0.58rem",
            color: "var(--text-muted)" }}>
            <span>{streaming ? "▶ GENERATING…" : "■ READY"}</span>
            <span>AGENT MODE · LOCAL</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function now() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}