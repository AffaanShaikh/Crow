import { useState, useEffect, useRef, useCallback } from "react";

// Config
const API_BASE                = "http://localhost:8000/api/v1";
const IMAGE_PROCESSOR_URL     = "http://localhost:8001/process-image";
const SESSION_ID              = crypto.randomUUID();
const DEFAULT_AI_IMAGE_URL    = "/src/assets/metaphor-refantazio-removebg-preview.png";
const DEFAULT_USER_IMAGE_URL  = "\\src\\assets\\i-really-love-the-portraits-from-metaphor-refantazio-so-v0-y945244wls9g1-removebg-preview.png";

// Tool name -> human label
const TOOL_LABELS = {
  list_calendar_events:   "Checking calendar",
  get_calendar_event:     "Reading event",
  create_calendar_event:  "Creating event",
  update_calendar_event:  "Updating event",
  delete_calendar_event:  "Deleting event",
};
const toolLabel = (name) => TOOL_LABELS[name] ?? `Running ${name}`;

// Global styles
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
    --portrait-ai:  #260010;
    --portrait-usr: #1e000c;
  }

  html, body, #root {
    height: 100%;
    width: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: 'IM Fell English', serif;
    font-size: 17px;
    overflow: hidden;
  }

  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      radial-gradient(1.5px 1.5px at 30px 30px, rgba(212,168,56,0.18) 0%, transparent 100%),
      radial-gradient(1px 1px at 70px 70px, rgba(130,32,64,0.28) 0%, transparent 100%);
    background-size: 100px 100px;
    pointer-events: none;
    z-index: 0;
  }

  body::after {
    content: '';
    position: fixed;
    inset: 0;
    background: radial-gradient(ellipse 80% 80% at 50% 50%, transparent 25%, rgba(10,0,4,0.55) 100%);
    pointer-events: none;
    z-index: 9998;
  }

  ::-webkit-scrollbar       { width: 5px; }
  ::-webkit-scrollbar-track { background: var(--bg2); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  @keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes pulse-border {
    0%,100% { box-shadow: 0 0 6px rgba(192,48,96,0.25) inset; }
    50%      { box-shadow: 0 0 28px rgba(192,48,96,0.7) inset, 0 0 36px rgba(240,200,64,0.15); }
  }
  @keyframes blink {
    0%,100% { opacity: 1; }
    50%      { opacity: 0; }
  }
  @keyframes avatar-breathe {
    0%,100% { filter: brightness(1) drop-shadow(0 0 12px rgba(212,168,56,0.35)); }
    50%      { filter: brightness(1.12) drop-shadow(0 0 26px rgba(212,168,56,0.72)); }
  }
  @keyframes avatar-breathe-user {
    0%,100% { filter: brightness(1) drop-shadow(0 0 12px rgba(196,18,60,0.35)); }
    50%      { filter: brightness(1.1) drop-shadow(0 0 24px rgba(196,18,60,0.62)); }
  }
  @keyframes avatar-active {
    0%,100% { filter: brightness(1.15) drop-shadow(0 0 22px rgba(212,168,56,0.85)); }
    50%      { filter: brightness(1.3) drop-shadow(0 0 42px rgba(212,168,56,1)); }
  }
  @keyframes typing-dot {
    0%,80%,100% { transform: scale(0.55); opacity: 0.35; }
    40%         { transform: scale(1);    opacity: 1; }
  }
  @keyframes status-glow {
    0%,100% { box-shadow: 0 0 4px currentColor; }
    50%      { box-shadow: 0 0 14px currentColor, 0 0 28px currentColor; }
  }
  @keyframes scan-line {
    from { transform: translateY(-100%); }
    to   { transform: translateY(600px); }
  }
  @keyframes gold-pulse {
    0%,100% { text-shadow: 0 0 8px rgba(240,200,64,0.25);  color: var(--gold); }
    50%      { text-shadow: 0 0 26px rgba(240,200,64,0.82), 0 0 50px rgba(240,200,64,0.25); color: var(--gold-bright); }
  }
  @keyframes portrait-entrance {
    from { opacity: 0; transform: translateY(-8px) scale(0.95); }
    to   { opacity: 1; transform: translateY(0)    scale(1); }
  }
  @keyframes ornament-glow {
    0%,100% { filter: drop-shadow(0 0 3px rgba(212,168,56,0.4)); }
    50%      { filter: drop-shadow(0 0 12px rgba(212,168,56,0.9)); }
  }
  @keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
  }
  @keyframes proc-pulse {
    0%,100% { opacity: 0.55; }
    50%      { opacity: 1; }
  }
  @keyframes tool-glow {
    0%,100% { box-shadow: 0 0 6px rgba(167,139,250,0.3); }
    50%      { box-shadow: 0 0 18px rgba(167,139,250,0.7); }
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
  const [src,        setSrc]        = useState(null);
  const [processing, setProcessing] = useState(false);
  const [error,      setError]      = useState(null);

  useEffect(() => {
    if (!imageUrl) return;
    let cancelled = false;
    setProcessing(true);
    setError(null);

    (async () => {
      try {
        const resp = await fetch(IMAGE_PROCESSOR_URL, {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body:    JSON.stringify({ image_url: imageUrl }),
        });
        if (cancelled) return;
        if (resp.ok) {
          const data = await resp.json();
          if (!cancelled) setSrc(`data:image/png;base64,${data.image}`);
        } else {
          throw new Error(`HTTP ${resp.status}`);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message);
          setSrc(imageUrl);
        }
      } finally {
        if (!cancelled) setProcessing(false);
      }
    })();

    return () => { cancelled = true; };
  }, [imageUrl]);

  return { src, processing, error };
}

// Timestamp
function Timestamp({ ts }) {
  return (
    <span style={{
      fontSize: "0.72rem",
      color: "var(--text-muted)",
      fontFamily: "'Special Elite', monospace",
      fontStyle: "italic",
      marginTop: 7,
      display: "block",
      letterSpacing: "0.05em",
    }}>
      ✦ {ts}
    </span>
  );
}

// Processing indicator overlay
function ProcessingRing({ accentCol }) {
  return (
    <div style={{
      position: "absolute",
      inset: 0,
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      gap: 6,
      zIndex: 4,
    }}>
      <div style={{
        width: 48, height: 48,
        border: `1.5px solid ${accentCol}25`,
        borderTop: `1.5px solid ${accentCol}`,
        borderRadius: "50%",
        animation: "spin 0.9s linear infinite",
      }} />
      <span style={{
        fontFamily: "'Special Elite', monospace",
        fontSize: "0.44rem",
        color: accentCol,
        letterSpacing: "0.14em",
        textTransform: "uppercase",
        animation: "proc-pulse 1.2s ease-in-out infinite",
      }}>
        processing
      </span>
    </div>
  );
}

// Portrait component
function CharPortrait({ isUser, isStreaming, isOnline, avatarSrc, isProcessing }) {
  const accentCol   = isUser ? "var(--rose)"      : "var(--frame-gold)";
  const nameText    = isUser ? "You"              : "Elda";
  const isActive    = isStreaming && !isUser;
  const fallbackSym = isUser ? "✦"               : (isStreaming ? "☽" : "☾");

  const imgAnim = isActive
    ? "avatar-active 1.4s ease-in-out infinite"
    : isUser
      ? "avatar-breathe-user 6s ease-in-out infinite"
      : "avatar-breathe 5s ease-in-out infinite";

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      gap: 0,
      animation: "portrait-entrance 0.3s ease both",
      marginBottom: "-2",
      position: "relative",
      zIndex: 3,
    }}>
      <div style={{
        position: "relative",
        width: 100,
        height: 120,
      }}>
        {isProcessing ? (
          <div style={{
            width: 100, height: 120,
            display: "flex", alignItems: "center", justifyContent: "center",
            position: "relative",
          }}>
            <div style={{
              fontSize: "1.6rem",
              color: accentCol,
              opacity: 0.18,
              lineHeight: 1,
              userSelect: "none",
              paddingBottom: "15px",
            }}>
              {fallbackSym}
            </div>
            <ProcessingRing accentCol={accentCol} />
          </div>
        ) : avatarSrc ? (
          <img
            src={avatarSrc}
            alt={nameText}
            draggable={false}
            style={{
              width: 100, height: 120,
              objectFit: "contain",
              objectPosition: "bottom",
              display: "block",
              animation: imgAnim,
              userSelect: "none",
              pointerEvents: "none",
            }}
          />
        ) : (
          <div style={{
            width: 100, height: 120,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: "3.2rem",
            color: accentCol,
            filter: `drop-shadow(0 0 12px ${accentCol})`,
            opacity: 0.75,
          }}>
            {fallbackSym}
          </div>
        )}

        {isActive && !isProcessing && (
          <div style={{
            position: "absolute",
            bottom: 2, left: "50%",
            transform: "translateX(-50%)",
            width: 7, height: 7,
            borderRadius: "50%",
            background: "var(--gold)",
            color: "var(--gold)",
            animation: "status-glow 0.9s ease-in-out infinite",
            zIndex: 5,
          }} />
        )}
      </div>
    </div>
  );
}

// Tool call bubble
function ToolCallBubble({ toolName, done, success }) {
  const label = toolLabel(toolName);
  const borderColor = done ? (success ? "var(--green)" : "var(--red)") : "var(--purple)";
  const textColor = done ? (success ? "var(--green)" : "var(--red)") : "var(--purple)";
  
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 8,
      padding: "7px 14px", 
      background: "var(--tool-bg)",
      border: `1px solid ${borderColor}`,
      borderTop: `2px solid ${borderColor}`,
      clipPath: "polygon(0 6px, 6px 0, 100% 0, 100% 100%, 0 100%)",
      fontSize: "0.72rem", 
      fontFamily: "'Special Elite', monospace",
      color: textColor,
      animation: done ? "fadeSlideIn 0.15s ease both" : "fadeSlideIn 0.15s ease both, tool-glow 1.5s ease-in-out infinite",
      maxWidth: "60%",
      boxShadow: done ? "none" : "0 0 12px rgba(167,139,250,0.2)",
    }}>
      <span style={{
        display: "inline-block",
        animation: done ? "none" : "spin 1s linear infinite",
        fontSize: "0.75rem",
      }}>
        {done ? (success ? "✓" : "✗") : "◌"}
      </span>
      <span>{done ? (success ? `${label} ✦ done` : `${label} ✦ failed`) : `✦ ${label}..`}</span>
    </div>
  );
}

// Thinking bubble
function ThinkingBubble({ content }) {
  return (
    <div style={{
      padding: "9px 16px",
      background: "transparent",
      border: "1px dashed var(--border)",
      borderLeft: "2px solid var(--purple)",
      fontSize: "0.82rem", 
      fontStyle: "italic",
      color: "var(--text-dim)", 
      maxWidth: "80%",
      animation: "fadeSlideIn 0.15s ease both",
      fontFamily: "'IM Fell English', serif",
    }}>
      <span style={{ color: "var(--purple)", fontSize: "0.7rem", letterSpacing: "0.1em", fontFamily: "'Special Elite', monospace" }}>
        ◈ cogitating
      </span>
      <div style={{ marginTop: 4 }}>{content}</div>
    </div>
  );
}

// Message bubble with portrait and tools
function MessageBubble({ msg, status, avatarSrc, isAvatarProcessing }) {
  const isUser   = msg.role === "user";
  const isActive = !!msg.streaming;
  const isOnline = status === "online";

  const borderCol = isUser ? "var(--crimson)"    : "var(--frame-gold)";
  const accentCol = isUser ? "var(--rose)"       : "var(--gold-bright)";
  const nameLabel = isUser ? "You"               : "Elda";

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: isUser ? "flex-end" : "flex-start",
      animation: "fadeSlideIn 0.25s ease both",
      marginBottom: 26,
      gap: 6,
    }}>
      {/* Portrait above */}
      <div style={{ marginLeft: isUser ? 0 : 8, marginRight: isUser ? 8 : 0 }}>
        <CharPortrait
          isUser={isUser}
          isStreaming={isActive}
          isOnline={isOnline}
          avatarSrc={avatarSrc}
          isProcessing={isAvatarProcessing}
        />
      </div>

      {/* Tool calls before content */}
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

      {/* Main text bubble - only render when there's content */}
      {(msg.content || msg.streaming) && (
        <div style={{
          maxWidth: "73%",
          position: "relative",
          padding: 3,
          background: `linear-gradient(135deg, ${accentCol}44, ${borderCol}22)`,
          boxShadow: isActive
            ? `0 0 28px ${borderCol}66`
            : `0 2px 20px rgba(0,0,0,0.7)`,
          transition: "box-shadow 0.4s ease",
          clipPath: isUser
            ? "polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 0 100%)"
            : "polygon(0 10px, 10px 0, 100% 0, 100% 100%, 0 100%)",
        }}>
          <div style={{
            background: isUser ? "var(--user-bg)" : "var(--ai-bg)",
            padding: "14px 20px 12px",
            clipPath: isUser
              ? "polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 0 100%)"
              : "polygon(0 8px, 8px 0, 100% 0, 100% 100%, 0 100%)",
            borderTop: `2px solid ${borderCol}`,
          }}>
            {/* Speaker header */}
            <div style={{
              display: "flex",
              alignItems: "center",
              gap: 5,
              marginBottom: 9,
              fontFamily: "'UnifrakturMaguntia', serif",
              fontSize: isUser ? "0.88rem" : "1.05rem",
              color: accentCol,
              letterSpacing: "0.04em",
              animation: !isUser ? "gold-pulse 3.5s ease-in-out infinite" : "none",
              textShadow: `0 0 14px ${accentCol}55`,
            }}>
              <span style={{
                fontFamily: "'Special Elite', monospace",
                fontSize: "0.68rem",
                opacity: 0.65,
                letterSpacing: "0.06em",
              }}>🙮</span>
              {nameLabel}
            </div>

            {/* Message content */}
            <div style={{
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              fontSize: "1.04rem",
              lineHeight: 1.8,
              color: "var(--text)",
              fontFamily: "'Special Elite', monospace",
            }}>
              {msg.content}
              {msg.streaming && (
                <span style={{
                  display: "inline-block",
                  width: 9, height: 16,
                  background: "var(--gold)",
                  marginLeft: 3,
                  verticalAlign: "middle",
                  animation: "blink 0.8s step-start infinite",
                  opacity: 0.9,
                }} />
              )}
            </div>
            <Timestamp ts={msg.ts} />
          </div>
        </div>
      )}
    </div>
  );
}

// Typing indicator
function TypingIndicator({ avatarSrc, isAvatarProcessing }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start", marginBottom: 26 }}>
      <div style={{ marginLeft: 14 }}>
        <CharPortrait
          isUser={false}
          isStreaming={false}
          isOnline={true}
          avatarSrc={avatarSrc}
          isProcessing={isAvatarProcessing}
        />
      </div>
      <div style={{
        padding: "14px 22px",
        background: "var(--ai-bg)",
        borderTop: "2px solid var(--frame-gold)",
        clipPath: "polygon(0 8px, 8px 0, 100% 0, 100% 100%, 0 100%)",
        display: "flex", gap: 8, alignItems: "center",
        marginLeft: 14,
        boxShadow: "0 2px 20px rgba(0,0,0,0.7)",
      }}>
        {[0, 0.2, 0.4].map((delay, i) => (
          <div key={i} style={{
            width: 9, height: 9, borderRadius: "50%",
            background: "var(--gold)",
            animation: `typing-dot 1.3s ${delay}s ease-in-out infinite`,
          }} />
        ))}
      </div>
    </div>
  );
}

// System panel (w/ MCP info)
function SystemPanel({ status, streaming, featureFlags }) {
  const isOnline = status === "online";
  return (
    <div style={{
      width: 210,
      flexShrink: 0,
      display: "flex",
      flexDirection: "column",
      gap: 13,
      padding: "20px 15px",
      borderRight: "1px solid var(--border)",
      background: "var(--bg2)",
      overflow: "hidden",
      position: "relative",
      zIndex: 1,
    }}>
      <div style={{
        fontFamily: "'Cinzel Decorative', serif",
        fontSize: "0.56rem",
        color: "var(--frame-gold)",
        letterSpacing: "0.2em",
        textAlign: "center",
        borderBottom: "1px solid var(--border)",
        paddingBottom: 10,
        animation: "ornament-glow 4s ease-in-out infinite",
      }}>
        ✦ C O D E X ✦
      </div>

      {/* Entity nameplate */}
      <div style={{
        border: "1px solid var(--border)",
        borderTop: "2px solid var(--frame-gold)",
        padding: "11px 13px",
        background: "var(--panel)",
        display: "flex", flexDirection: "column", gap: 5,
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span style={{
            fontFamily: "'UnifrakturMaguntia', serif",
            fontSize: "1.45rem",
            color: "var(--gold)",
            animation: "gold-pulse 3.5s ease-in-out infinite",
          }}>Elda</span>
          <div style={{
            width: 8, height: 8, borderRadius: "50%",
            background: isOnline ? "var(--green)" : "var(--red)",
            color: isOnline ? "var(--green)" : "var(--red)",
            animation: "status-glow 2.5s ease-in-out infinite",
          }} />
        </div>
        <div style={{ fontSize: "0.7rem", color: "var(--text-dim)", fontStyle: "italic" }}>llama3.2</div>
        <div style={{ fontSize: "0.62rem", color: isOnline ? "var(--green)" : "var(--red)", fontFamily: "'Special Elite', monospace", letterSpacing: "0.1em" }}>
          {isOnline ? "ONLINE" : "OFFLINE"}
        </div>
      </div>

      {/* Telemetry */}
      <div style={{
        border: "1px solid var(--border)",
        padding: "11px 13px",
        background: "var(--panel)",
        fontFamily: "'Special Elite', monospace",
        fontSize: "0.67rem",
        color: "var(--text-dim)",
        display: "flex", flexDirection: "column", gap: 8,
      }}>
        <div style={{ color: "var(--frame-gold)", fontSize: "0.54rem", letterSpacing: "0.18em", marginBottom: 2, fontFamily: "'Cinzel Decorative', serif" }}>
          ✦ Status
        </div>
        {[
          ["LLM",    isOnline ? "llama.cpp" : "-",  isOnline ? "var(--green)" : "var(--red)"],
          ["Agent",  featureFlags?.mcp ? "active" : "off", featureFlags?.mcp ? "var(--purple)" : "var(--text-muted)"],
          ["Memory", "In-context",                  "var(--gold)"],
          ["Avatar", featureFlags?.avatar ? "On" : "Off", featureFlags?.avatar ? "var(--gold-bright)" : "var(--text-muted)"],
          ["TTS",    "Coming soon", "var(--text-muted)"],
          ["ASR",    "Coming soon", "var(--text-muted)"],
          ["RAG",    "Coming soon", "var(--text-muted)"],
        ].map(([label, value, col]) => (
          <div key={label} style={{ display: "flex", justifyContent: "space-between", gap: 4 }}>
            <span style={{ color: "var(--text-muted)", fontStyle: "italic" }}>{label}</span>
            <span style={{ color: col, textAlign: "right" }}>{value}</span>
          </div>
        ))}
      </div>

      {/* Voice controls 'placeholder' */}
      <div style={{
        border: "1px dashed var(--border)",
        padding: "13px 11px",
        background: "transparent",
        textAlign: "center",
        fontFamily: "'Cinzel Decorative', serif",
        fontSize: "0.5rem",
        color: "var(--text-muted)",
        letterSpacing: "0.14em",
        lineHeight: 2,
      }}>
        ✦ Voice Controls ✦<br />
        <span style={{ fontFamily: "'Special Elite', monospace", fontSize: "0.6rem", opacity: 0.45 }}>ASR &amp; TTS</span>
      </div>

      {/* Live streaming badge */}
      {streaming && (
        <div style={{
          border: "1px solid var(--border-glow)",
          padding: "9px 13px",
          background: "var(--panel)",
          fontFamily: "'Special Elite', monospace",
          fontSize: "0.62rem",
          color: "var(--gold)",
          textAlign: "center",
          letterSpacing: "0.1em",
          animation: "pulse-border 1.5s infinite",
        }}>
          ✦ responding..
        </div>
      )}
    </div>
  );
}

// Icon button helper
function IconBtn({ onClick, disabled, title, children, active, style: extra }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      style={{
        width: 38, height: 38,
        flexShrink: 0,
        background: "transparent",
        border: `1px ${active ? "solid" : "dashed"} ${active ? "var(--frame-gold)" : "var(--border)"}`,
        borderRadius: 3,
        color: active ? "var(--gold)" : "var(--text-muted)",
        cursor: disabled ? "not-allowed" : "pointer",
        fontSize: "1rem",
        display: "flex", alignItems: "center", justifyContent: "center",
        transition: "all 0.22s",
        boxShadow: active ? "0 0 8px rgba(212,168,56,0.2)" : "none",
        ...extra,
      }}
      onMouseEnter={e => {
        if (!disabled) {
          e.currentTarget.style.borderColor = "var(--frame-gold)";
          e.currentTarget.style.color       = "var(--gold)";
          e.currentTarget.style.borderStyle = "solid";
        }
      }}
      onMouseLeave={e => {
        if (!disabled && !active) {
          e.currentTarget.style.borderColor = "var(--border)";
          e.currentTarget.style.color       = "var(--text-muted)";
          e.currentTarget.style.borderStyle = "dashed";
        }
      }}
    >
      {children}
    </button>
  );
}

// Main app
export default function App() {
  injectCSS(GLOBAL_CSS);

  const [messages,     setMessages]     = useState([]);
  const [input,        setInput]        = useState("");
  const [streaming,    setStreaming]    = useState(false);
  const [status,       setStatus]       = useState("checking");
  const [featureFlags, setFeatureFlags] = useState({});
  const [showTyping,   setShowTyping]   = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef       = useRef(null);
  const abortRef       = useRef(null);

  const {
    src: aiAvatarSrc,
    processing: aiAvatarProcessing,
  } = useProcessedAvatar(DEFAULT_AI_IMAGE_URL);

  const {
    src: userAvatarSrc,
    processing: userAvatarProcessing,
  } = useProcessedAvatar(DEFAULT_USER_IMAGE_URL);

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
          mcp: true,  // assume true if backend started with mcp_enabled
        });
      })
      .catch(() => setStatus("offline"));

    setMessages([{
      id: crypto.randomUUID(),
      role: "assistant",
      content: "What a wonderful day to be alive!",
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

      // Use /agent/stream - handles both tool-use and plain chat
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

            // Standard streaming token
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

            // LLM reasoning text before a tool call
            case "thinking":
              if (event.content) {
                aiThinking = event.content;
                upsertAiMessage({
                  thinking: aiThinking,
                  toolCalls: Object.values(toolCalls),
                });
              }
              break;

            // Tool invocation started
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

            // Tool execution finished
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

            // Stream finished
            case "done":
              upsertAiMessage({
                streaming: false,
                toolCalls: Object.values(toolCalls),
              });
              break;

            // Error
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

  const canSend = !streaming && !!input.trim();

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column", background: "var(--bg)", position: "relative", zIndex: 1 }}>

      {/* Top bar */}
      <div style={{
        height: 54,
        borderBottom: "1px solid var(--frame-gold)",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "0 24px",
        background: "var(--bg2)",
        flexShrink: 0,
        boxShadow: "0 2px 26px rgba(212,168,56,0.1)",
        position: "relative", zIndex: 2,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 15 }}>
          <span style={{ color: "var(--frame-gold)", fontSize: "0.95rem", opacity: 0.7 }}>🙜</span>
          <span style={{ fontFamily: "'UnifrakturMaguntia', serif", fontSize: "1.5rem", color: "var(--gold)", animation: "gold-pulse 4s ease-in-out infinite" }}>
            Crow
          </span>
          <span style={{ width: 1, height: 20, background: "var(--border)" }} />
          <span style={{ fontFamily: "'Special Elite', monospace", fontSize: "0.65rem", color: "var(--text-muted)", letterSpacing: "0.12em", fontStyle: "italic" }}>
            Session · {SESSION_ID.slice(0, 8).toUpperCase()}
          </span>
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          {(aiAvatarProcessing || userAvatarProcessing) && (
            <div style={{
              display: "flex", alignItems: "center", gap: 5,
              fontFamily: "'Special Elite', monospace",
              fontSize: "0.6rem",
              color: "var(--frame-gold)",
              letterSpacing: "0.1em",
              animation: "proc-pulse 1s ease-in-out infinite",
              border: "1px solid var(--border)",
              padding: "3px 10px",
              borderRadius: 2,
            }}>
              <div style={{
                width: 10, height: 10,
                border: "1.5px solid rgba(212,168,56,0.3)",
                borderTop: "1.5px solid var(--frame-gold)",
                borderRadius: "50%",
                animation: "spin 0.8s linear infinite",
                flexShrink: 0,
              }} />
              ✦ IMG
            </div>
          )}
          <span style={{ fontFamily: "'Special Elite', monospace", fontSize: "0.65rem", color: status === "online" ? "var(--green)" : "var(--red)", letterSpacing: "0.1em" }}>
            ● {status.toUpperCase()}
          </span>
          <button
            onClick={clearSession}
            style={{ fontFamily: "'Cinzel Decorative', serif", fontSize: "0.52rem", background: "transparent", border: "1px solid var(--border)", color: "var(--text-dim)", padding: "4px 14px", borderRadius: 2, cursor: "pointer", letterSpacing: "0.12em", transition: "all 0.25s" }}
            onMouseEnter={e => { e.target.style.borderColor="var(--frame-gold)"; e.target.style.color="var(--gold)"; e.target.style.boxShadow="0 0 12px rgba(212,168,56,0.25)"; }}
            onMouseLeave={e => { e.target.style.borderColor="var(--border)";     e.target.style.color="var(--text-dim)"; e.target.style.boxShadow="none"; }}
          >
            Erase Memory
          </button>
          <span style={{ color: "var(--frame-gold)", fontSize: "0.95rem", opacity: 0.7 }}>🙞</span>
        </div>
      </div>

      {/* Main layout */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        <SystemPanel status={status} streaming={streaming} featureFlags={featureFlags} />

        {/* Chat column */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          {/* Messages */}
          <div style={{ flex: 1, overflowY: "auto", padding: "32px 36px 14px", display: "flex", flexDirection: "column" }}>
            {messages.map(msg => (
              <MessageBubble
                key={msg.id}
                msg={msg}
                status={status}
                avatarSrc={msg.role === "user" ? userAvatarSrc : aiAvatarSrc}
                isAvatarProcessing={msg.role === "user" ? userAvatarProcessing : aiAvatarProcessing}
              />
            ))}
            {showTyping && (
              <TypingIndicator
                avatarSrc={aiAvatarSrc}
                isAvatarProcessing={aiAvatarProcessing}
              />
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input area */}
          <div style={{
            borderTop: "1px solid var(--border)",
            padding: "13px 20px",
            background: "var(--bg2)",
            display: "flex", gap: 9, alignItems: "flex-end",
            boxShadow: "0 -2px 26px rgba(10,0,5,0.6)",
          }}>

            {/* Textarea */}
            <div style={{ flex: 1, position: "relative" }}>
              <textarea
                ref={inputRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={streaming}
                placeholder={streaming ? "responding.." : "Write some sins but no tragedies ..."}
                rows={1}
                style={{
                  width: "100%", background: "var(--panel)",
                  border: "1px solid var(--border)", borderRadius: 3,
                  padding: "11px 17px", color: "var(--text)",
                  fontFamily: "'Special Elite', monospace",
                  fontSize: "1.04rem",
                  lineHeight: 1.65, resize: "none", outline: "none",
                  transition: "border-color 0.25s, box-shadow 0.25s",
                  minHeight: 44, maxHeight: 140, overflow: "auto",
                }}
                onFocus={e => { e.target.style.borderColor="var(--frame-gold)"; e.target.style.boxShadow="0 0 16px rgba(212,168,56,0.18)"; }}
                onBlur={e  => { e.target.style.borderColor="var(--border)";     e.target.style.boxShadow="none"; }}
                onInput={e => { e.target.style.height="auto"; e.target.style.height=Math.min(e.target.scrollHeight,140)+"px"; }}
              />
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 6, flexShrink: 0 }}>
              <IconBtn title="Use tools (coming soon)" disabled>&</IconBtn>
              <IconBtn title="Attach file (coming soon)" disabled>+</IconBtn>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 6, flexShrink: 0 }}>
              <IconBtn title="Voice input (coming soon)" disabled style={{ height: 38 }}>◎</IconBtn>

              {/* Send */}
              <button
                onClick={sendMessage}
                disabled={!canSend}
                title="Send message"
                style={{
                  width: 38, height: 38, flexShrink: 0,
                  background: canSend ? "var(--gold-dim)" : "transparent",
                  border: `1px solid ${canSend ? "var(--frame-gold)" : "var(--border)"}`,
                  borderRadius: 3,
                  color: canSend ? "var(--gold)" : "var(--text-muted)",
                  cursor: canSend ? "pointer" : "not-allowed",
                  fontSize: "1rem",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  transition: "all 0.25s",
                  boxShadow: canSend ? "0 0 12px rgba(212,168,56,0.25)" : "none",
                }}
                onMouseEnter={e => {
                  if (canSend) {
                    e.currentTarget.style.background  = "var(--crimson)";
                    e.currentTarget.style.boxShadow   = "0 0 26px rgba(212,168,56,0.5)";
                    e.currentTarget.style.color        = "var(--gold-bright)";
                  }
                }}
                onMouseLeave={e => {
                  if (canSend) {
                    e.currentTarget.style.background  = "var(--gold-dim)";
                    e.currentTarget.style.boxShadow   = "0 0 12px rgba(212,168,56,0.25)";
                    e.currentTarget.style.color        = "var(--gold)";
                  }
                }}
              >
                ▶
              </button>
            </div>
          </div>

          {/* Status bar */}
          <div style={{
            padding: "4px 24px", borderTop: "1px solid var(--border)", background: "var(--bg)",
            display: "flex", justifyContent: "space-between", alignItems: "center",
            fontFamily: "'Special Elite', monospace", fontSize: "0.62rem",
            color: "var(--text-muted)", letterSpacing: "0.08em",
          }}>
            <span style={{ fontStyle: "italic", display: "flex", alignItems: "center", gap: 5 }}>
              {streaming ? "✦ generating.." : "■ ready"}
              {(aiAvatarProcessing || userAvatarProcessing) && (
                <span style={{ color: "var(--frame-gold)", animation: "proc-pulse 1s ease-in-out infinite" }}>
                  · img processing..
                </span>
              )}
            </span>
            <span>crow ver. 0.1.3</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function now() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}