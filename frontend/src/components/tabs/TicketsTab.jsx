// src/components/tabs/TicketsTab.jsx
import React, { useState, useEffect } from "react";

// ── Light-mode status configs use deeper, more saturated colors ──
function StatusBadge({ status = "open", isLight }) {
  const cfgs = {
    open: {
      dot: isLight ? "#059669" : "#10b981",
      bg: isLight ? "#ecfdf5" : "rgba(16,185,129,0.08)",
      border: isLight ? "#6ee7b7" : "rgba(16,185,129,0.2)",
      text: isLight ? "#065f46" : "#34d399",
    },
    resolved: {
      dot: isLight ? "#64748b" : "#475569",
      bg: isLight ? "#f1f5f9" : "rgba(100,116,139,0.08)",
      border: isLight ? "#cbd5e1" : "rgba(100,116,139,0.2)",
      text: isLight ? "#334155" : "#94a3b8",
    },
  };
  const cfg = cfgs[status] || cfgs.open;
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 5,
      background: cfg.bg, border: `1px solid ${cfg.border}`,
      color: cfg.text, fontSize: 10, fontWeight: 700,
      letterSpacing: "0.08em", padding: "3px 9px",
      borderRadius: 4, textTransform: "uppercase",
    }}>
      <span style={{ width: 5, height: 5, borderRadius: "50%", background: cfg.dot, flexShrink: 0 }} />
      {status === "open" ? "Open" : "Resolved"}
    </span>
  );
}

function Avatar({ name = "?" }) {
  const initials = name.split(" ").map(w => w[0]).slice(0, 2).join("").toUpperCase();
  const palette = ["#0891b2","#059669","#7c3aed","#db2777","#d97706","#2563eb"];
  const hue = palette[name.charCodeAt(0) % palette.length];
  

  return (
    <div style={{
      width: 28, height: 28, borderRadius: "50%",
      background: `${hue}22`, border: `1.5px solid ${hue}55`,
      display: "flex", alignItems: "center", justifyContent: "center",
      fontSize: 10, fontWeight: 700, color: hue, flexShrink: 0,
      fontFamily: "'DM Mono', monospace",
    }}>
      {initials}
    </div>
  );
}

function TicketRow({ ticket, onView, isLight }) {
  const [hovered, setHovered] = useState(false);
  const status = (ticket.status || "open").toLowerCase();
  return (
    <tr
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        borderBottom: `1px solid ${isLight ? "#e2e8f0" : "rgba(255,255,255,0.04)"}`,
        background: hovered
          ? (isLight ? "#f0fdf8" : "rgba(6,182,212,0.04)")
          : "transparent",
        transition: "background 0.15s",
      }}
    >
      <td style={{ padding: "14px 20px", width: 110 }}>
        <span style={{
          fontFamily: "'DM Mono', monospace", fontSize: 11,
          color: isLight ? "#059669" : "#475569",
          letterSpacing: "0.02em", fontWeight: 600,
        }}>
          #TCK_{String(ticket.id).padStart(4, "0")}
        </span>
      </td>
      <td style={{ padding: "14px 16px" }}>
        <span style={{
          fontSize: 13, fontWeight: 600,
          color: isLight ? "#0f172a" : "#e2e8f0",
          display: "block", lineHeight: 1.4,
        }}>
          {ticket.title}
        </span>
      </td>
      <td style={{ padding: "14px 16px", width: 160 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <Avatar name={ticket.creator_name || "?"} />
          <span style={{ fontSize: 12, color: isLight ? "#475569" : "#94a3b8", fontWeight: 500 }}>
            {ticket.creator_name}
          </span>
        </div>
      </td>
      <td style={{ padding: "14px 16px", width: 100, textAlign: "center" }}>
        <StatusBadge status={status} isLight={isLight} />
      </td>
      <td style={{ padding: "14px 16px", width: 160 }}>
        <span style={{ fontSize: 11, color: isLight ? "#64748b" : "#475569", fontFamily: "'DM Mono', monospace" }}>
          {new Date(ticket.updated_at || ticket.created_at).toLocaleDateString("en-GB", {
            day: "2-digit", month: "short", year: "numeric"
          })}
        </span>
      </td>
      <td style={{ padding: "14px 20px", width: 100, textAlign: "right" }}>
        <button
          onClick={() => onView(ticket.id)}
          style={{
            background: isLight ? "#059669" : "transparent",
            border: isLight ? "none" : "1px solid rgba(6,182,212,0.3)",
            color: isLight ? "#fff" : "#22d3ee",
            padding: "5px 14px", borderRadius: 5,
            fontSize: 11, fontWeight: 700, letterSpacing: "0.06em",
            cursor: "pointer", transition: "all 0.15s",
            textTransform: "uppercase",
          }}
          onMouseEnter={e => {
            if (isLight) e.currentTarget.style.background = "#047857";
            else e.currentTarget.style.background = "rgba(6,182,212,0.08)";
          }}
          onMouseLeave={e => {
            if (isLight) e.currentTarget.style.background = "#059669";
            else e.currentTarget.style.background = "transparent";
          }}
        >
          View
        </button>
      </td>
    </tr>
  );
}

function ReplyBubble({ reply, isLight }) {
  const isAdmin = reply.replier_role === "admin";
  return (
    <div style={{ display: "flex", gap: 12 }}>
      <Avatar name={reply.replier_name || "?"} />
      <div style={{ flex: 1 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: isLight ? "#0f172a" : "#e2e8f0" }}>
            {reply.replier_name}
          </span>
          <span style={{
            fontSize: 9, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase",
            padding: "2px 6px", borderRadius: 3,
            background: isAdmin
              ? (isLight ? "#fff1f2" : "rgba(239,68,68,0.08)")
              : (isLight ? "#f1f5f9" : "rgba(100,116,139,0.1)"),
            color: isAdmin
              ? (isLight ? "#be123c" : "#f87171")
              : (isLight ? "#475569" : "#94a3b8"),
            border: `1px solid ${isAdmin
              ? (isLight ? "#fecdd3" : "rgba(239,68,68,0.2)")
              : (isLight ? "#e2e8f0" : "rgba(100,116,139,0.2)")}`,
          }}>
            {reply.replier_role}
          </span>
          <span style={{ marginLeft: "auto", fontSize: 10, color: isLight ? "#94a3b8" : "#475569", fontFamily: "'DM Mono', monospace" }}>
            {new Date(reply.created_at).toLocaleString("en-GB", {
              day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit"
            })}
          </span>
        </div>
        <div style={{
          padding: "10px 14px", borderRadius: 8,
          background: isAdmin
            ? (isLight ? "#f0fdf4" : "rgba(16,185,129,0.06)")
            : (isLight ? "#f8fafc" : "rgba(255,255,255,0.03)"),
          border: `1px solid ${isAdmin
            ? (isLight ? "#bbf7d0" : "rgba(16,185,129,0.12)")
            : (isLight ? "#e2e8f0" : "rgba(255,255,255,0.06)")}`,
          // Left accent stripe for admin
          ...(isAdmin && isLight ? { borderLeft: "3px solid #10b981" } : {}),
          fontSize: 13, lineHeight: 1.65,
          color: isLight ? "#1e293b" : "#cbd5e1",
          whiteSpace: "pre-wrap",
        }}>
          {reply.message}
        </div>
      </div>
    </div>
  );
}

export default function TicketsTab({ isLight = false }) {
  const [tickets, setTickets] = useState([]);
  const [selectedTicket, setSelectedTicket] = useState(null);
  const [replies, setReplies] = useState([]);
  const [replyText, setReplyText] = useState("");
  const [showForm, setShowForm] = useState(false);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [imageFiles, setImageFiles] = useState([]);
  const userId = localStorage.getItem("user_id");
  const [replyImageFiles, setReplyImageFiles] = useState([]); // <--- ADD THIS

  const loadTickets = async () => {
    try {
      const r = await fetch(`http://localhost:21081/api/tickets?user_id=${userId}`);
      const d = await r.json();
      if (d.status === "success") setTickets(d.data);
    } catch (e) { console.error(e); }
  };

  const loadTicketDetail = async (ticketId) => {
    try {
      const r = await fetch(`http://localhost:21081/api/tickets/${ticketId}?user_id=${userId}`);
      const d = await r.json();
      if (d.status === "success") { setSelectedTicket(d.data); setReplies(d.data.replies || []); }
    } catch (e) { console.error(e); }
  };

  useEffect(() => { loadTickets(); }, []);

  const handleCreateTicket = async (e) => {
    e.preventDefault();
    if (!title.trim() || !description.trim()) { alert("Please fill in both fields."); return; }
    
    const formData = new FormData();
    formData.append("title", title);
    formData.append("description", description);
    
    // Append multiple files to the same "images" key
    imageFiles.forEach((file) => {
      formData.append("images", file);
    });

    const token = localStorage.getItem("access_token");

    try {
      const r = await fetch("http://localhost:21081/api/tickets", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}` // <--- ADD AUTH HEADER TO FIX FK ERROR
        },
        body: formData,
      });
      if (r.ok) { 
        setTitle(""); setDescription(""); setImageFiles([]); setShowForm(false); loadTickets(); 
      } else {
        const err = await r.json();
        alert("Failed to create ticket: " + (err.detail || "Unknown error"));
      }
    } catch (e) { console.error(e); alert("Network error occurred."); }
  };

  const handlePostReply = async (e) => {
    e.preventDefault();
    if (!replyText.trim() && replyImageFiles.length === 0) { 
      alert("Reply cannot be empty."); return; 
    }
    
    const formData = new FormData();
    formData.append("message", replyText);
    
    replyImageFiles.forEach((file) => {
      formData.append("images", file);
    });

    const token = localStorage.getItem("access_token");

    try {
      const r = await fetch(`http://localhost:21081/api/tickets/${selectedTicket.id}/replies`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`
        },
        body: formData,
      });
      if (r.ok) { 
        setReplyText(""); 
        setReplyImageFiles([]); // Reset images
        loadTicketDetail(selectedTicket.id); 
      } else {
        const err = await r.json();
        alert("Failed to send reply: " + (err.detail || "Unknown error"));
      }
    } catch (e) { console.error(e); }
  };

  // ── Theme tokens — light mode now uses proper saturated emerald accents ──
  const t = isLight ? {
    pageBg:       "#f1f5f9",          // page-level bg (slight blue-gray tint)
    surface:      "#ffffff",
    surfaceRaised:"#f8fafc",
    border:       "#e2e8f0",
    borderStrong: "#cbd5e1",
    accent:       "#059669",
    accentDark:   "#047857",
    accentBg:     "#ecfdf5",
    accentBorder: "#6ee7b7",
    tableHead:    "#065f46",          // deep emerald thead
    tableHeadBg:  "#059669",
    textPrimary:  "#0f172a",
    textSecondary:"#475569",
    textMuted:    "#94a3b8",
    inputBg:      "#ffffff",
    inputBorder:  "#cbd5e1",
    codeBg:       "#f1f5f9",
    shadow:       "0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04)",
  } : {
    pageBg:       "transparent",
    surface:      "#0f172a",
    surfaceRaised:"#111827",
    border:       "rgba(255,255,255,0.07)",
    borderStrong: "rgba(255,255,255,0.12)",
    accent:       "#06b6d4",
    accentDark:   "#0891b2",
    accentBg:     "rgba(6,182,212,0.08)",
    accentBorder: "rgba(6,182,212,0.3)",
    tableHead:    "#e2e8f0",
    tableHeadBg:  "rgba(255,255,255,0.03)",
    textPrimary:  "#f1f5f9",
    textSecondary:"#64748b",
    textMuted:    "#475569",
    inputBg:      "#0c1524",
    inputBorder:  "rgba(255,255,255,0.08)",
    codeBg:       "rgba(255,255,255,0.02)",
    shadow:       "none",
  };

  const inputStyle = {
    width: "100%", boxSizing: "border-box",
    background: t.inputBg,
    border: `1px solid ${t.inputBorder}`,
    borderRadius: 7, padding: "9px 12px",
    fontSize: 13, color: t.textPrimary,
    outline: "none", transition: "border-color 0.15s, box-shadow 0.15s",
    fontFamily: "inherit",
  };

  return (
    <div style={{ fontFamily: "'Inter var', 'Inter', sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');
        .tk-input:focus {
          border-color: ${t.accent} !important;
          box-shadow: 0 0 0 3px ${t.accentBg} !important;
        }
        .tk-reply:focus {
          border-color: ${t.accent} !important;
          box-shadow: 0 0 0 3px ${t.accentBg} !important;
        }
        .tk-ghost:hover { background: ${isLight ? "#f1f5f9" : "rgba(255,255,255,0.05)"} !important; }
        .tk-primary:hover { background: ${t.accentDark} !important; }
      `}</style>

      {!selectedTicket ? (
        <>
          {/* ── Page Header ── */}
          <div style={{
            display: "flex", alignItems: "flex-start",
            justifyContent: "space-between", marginBottom: 24, gap: 16, flexWrap: "wrap",
          }}>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 5 }}>
                {/* Colored icon pill */}
                <div style={{
                  width: 30, height: 30, borderRadius: 8,
                  background: isLight ? "#059669" : "rgba(6,182,212,0.12)",
                  border: isLight ? "none" : "1px solid rgba(6,182,212,0.2)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                }}>
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                    <rect x="1" y="1" width="12" height="12" rx="2.5" stroke={isLight ? "#fff" : "#22d3ee"} strokeWidth="1.3"/>
                    <path d="M4 5h6M4 7.5h4" stroke={isLight ? "#fff" : "#22d3ee"} strokeWidth="1.3" strokeLinecap="round"/>
                  </svg>
                </div>
                <h2 style={{
                  margin: 0, fontSize: 18, fontWeight: 700,
                  color: t.textPrimary, letterSpacing: "-0.02em",
                }}>
                  Support Tickets
                </h2>
              </div>
              <p style={{ margin: 0, fontSize: 12, color: t.textSecondary, lineHeight: 1.5, paddingLeft: 40 }}>
                Report technical issues, server errors, or platform curriculum feedback.
              </p>
            </div>
            <button
              onClick={() => setShowForm(!showForm)}
              className="tk-primary"
              style={{
                display: "flex", alignItems: "center", gap: 7,
                background: showForm
                  ? (isLight ? "#f1f5f9" : "rgba(255,255,255,0.06)")
                  : t.accent,
                color: showForm ? t.textSecondary : "#fff",
                border: showForm ? `1px solid ${t.border}` : "none",
                borderRadius: 7, padding: "8px 16px",
                fontSize: 12, fontWeight: 600, letterSpacing: "0.02em",
                cursor: "pointer", transition: "all 0.15s",
                boxShadow: (!showForm && isLight) ? "0 1px 3px rgba(5,150,105,0.3)" : "none",
              }}
            >
              {showForm ? (
                <>
                  <svg width="11" height="11" viewBox="0 0 11 11" fill="none">
                    <path d="M1 1L10 10M10 1L1 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>
                  Cancel
                </>
              ) : (
                <>
                  <svg width="11" height="11" viewBox="0 0 11 11" fill="none">
                    <path d="M5.5 1v9M1 5.5h9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>
                  New Ticket
                </>
              )}
            </button>
          </div>

          {/* ── Create Form ── */}
          {showForm && (
            <div style={{
              background: t.surface,
              border: `1px solid ${isLight ? t.accentBorder : t.border}`,
              borderRadius: 12, padding: 24, marginBottom: 24,
              position: "relative", overflow: "hidden",
              boxShadow: isLight ? "0 4px 12px rgba(5,150,105,0.1)" : "none",
            }}>
              {/* Top stripe */}
              <div style={{
                position: "absolute", top: 0, left: 0, right: 0, height: 3,
                background: "linear-gradient(90deg, #10b981, #06b6d4, #818cf8)",
              }} />
              <h3 style={{
                margin: "0 0 20px", fontSize: 12, fontWeight: 700,
                color: isLight ? "#059669" : "#22d3ee",
                textTransform: "uppercase", letterSpacing: "0.1em",
              }}>
                New Support Request
              </h3>
              <form onSubmit={handleCreateTicket} style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <div>
                  <label style={{
                    display: "block", fontSize: 11, fontWeight: 600,
                    color: t.textSecondary, letterSpacing: "0.06em",
                    textTransform: "uppercase", marginBottom: 6,
                  }}>
                    Issue Title
                  </label>
                  <input
                    type="text" required value={title}
                    onChange={e => setTitle(e.target.value)}
                    className="tk-input"
                    placeholder="e.g. Cannot connect to DeepWiki server…"
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={{
                    display: "block", fontSize: 11, fontWeight: 600,
                    color: t.textSecondary, letterSpacing: "0.06em",
                    textTransform: "uppercase", marginBottom: 6,
                  }}>
                    Detailed Description
                  </label>
                  <textarea
                    required rows={4} value={description}
                    onChange={e => setDescription(e.target.value)}
                    className="tk-input"
                    placeholder="Include error codes, tracebacks, or steps to reproduce…"
                    style={{ ...inputStyle, resize: "vertical", lineHeight: 1.6 }}
                  />
                </div>
                <div>
                  <label style={{
                    display: "block", fontSize: 11, fontWeight: 600,
                    color: t.textSecondary, letterSpacing: "0.06em",
                    textTransform: "uppercase", marginBottom: 6,
                  }}>
                    Attach Screenshot (Optional)
                  </label>
                  <input
                    type="file"
                    accept="image/*"
                    multiple // <--- ALLOW MULTIPLE
                    onChange={e => setImageFiles(Array.from(e.target.files))} // <--- SAVE AS ARRAY
                    style={{ fontSize: 12, color: t.textSecondary }}
                  />
                  {/* Optional: Show selected file count */}
                  {imageFiles.length > 0 && (
                    <div style={{ fontSize: 11, color: t.accent, marginTop: 4 }}>
                      {imageFiles.length} file(s) selected
                    </div>
                  )}
                </div>
                <div style={{ display: "flex", justifyContent: "flex-end", gap: 10, paddingTop: 4 }}>
                  <button
                    type="button" className="tk-ghost"
                    onClick={() => { setShowForm(false); setTitle(""); setDescription(""); }}
                    style={{
                      background: "transparent",
                      border: `1px solid ${t.border}`,
                      color: t.textSecondary, borderRadius: 7,
                      padding: "8px 16px", fontSize: 12, fontWeight: 500,
                      cursor: "pointer", transition: "background 0.15s",
                    }}
                  >
                    Cancel
                  </button>
                  <button
                    type="submit" className="tk-primary"
                    style={{
                      background: t.accent, border: "none", color: "#fff",
                      borderRadius: 7, padding: "8px 20px",
                      fontSize: 12, fontWeight: 600, cursor: "pointer",
                      transition: "background 0.15s",
                      boxShadow: isLight ? "0 1px 3px rgba(5,150,105,0.35)" : "none",
                    }}
                  >
                    Submit Ticket
                  </button>
                </div>
              </form>
            </div>
          )}

          {/* ── Tickets Table ── */}
          <div style={{
            background: t.surface,
            border: `1px solid ${t.border}`,
            borderRadius: 12, overflow: "hidden",
            boxShadow: t.shadow,
          }}>
            <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 680 }}>
              <thead>
                <tr style={{
                  background: isLight ? t.tableHeadBg : t.tableHeadBg,
                  borderBottom: `1px solid ${isLight ? t.accentDark + "55" : t.border}`,
                }}>
                  {["Ticket ID", "Title", "Requester", "Status", "Last Updated", ""].map((h, i) => (
                    <th key={i} style={{
                      padding: "12px 20px",
                      textAlign: i === 3 ? "center" : i === 5 ? "right" : "left",
                      fontSize: 10, fontWeight: 700, letterSpacing: "0.09em",
                      textTransform: "uppercase",
                      color: isLight ? "#fff" : t.textSecondary,
                      fontFamily: "'DM Mono', monospace",
                    }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {tickets.length === 0 ? (
                  <tr>
                    <td colSpan={6} style={{
                      textAlign: "center", padding: "48px 20px",
                      color: t.textSecondary, fontSize: 13,
                    }}>
                      <div style={{ marginBottom: 8, opacity: 0.35, fontSize: 28 }}>📭</div>
                      No support tickets yet
                    </td>
                  </tr>
                ) : tickets.map(t_ => (
                  <TicketRow key={t_.id} ticket={t_} onView={loadTicketDetail} isLight={isLight} />
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        /* ══════════════════════════════════════════
           TICKET DETAIL VIEW — redesigned
        ══════════════════════════════════════════ */
        <>
          {/* ── Breadcrumb ── */}
          <div style={{ marginBottom: 20, display: "flex", alignItems: "center", gap: 8 }}>
            <button
              onClick={() => setSelectedTicket(null)}
              style={{
                background: "transparent", border: "none",
                color: t.accent, fontSize: 12, fontWeight: 600,
                cursor: "pointer", display: "flex", alignItems: "center", gap: 5,
                padding: 0, transition: "opacity 0.15s",
              }}
              onMouseEnter={e => e.currentTarget.style.opacity = "0.7"}
              onMouseLeave={e => e.currentTarget.style.opacity = "1"}
            >
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <path d="M9 11L5 7L9 3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              All Tickets
            </button>
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style={{ opacity: 0.25 }}>
              <path d="M5 3l4 4-4 4" stroke={t.textSecondary} strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 11, color: t.textMuted, letterSpacing: "0.03em" }}>
              #TCK_{String(selectedTicket.id).padStart(4, "0")}
            </span>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 272px", gap: 18, alignItems: "start" }}>

            {/* ════ MAIN COLUMN ════ */}
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

              {/* ── Hero ticket card ── */}
              <div style={{
                background: t.surface,
                border: `1px solid ${t.border}`,
                borderRadius: 14, overflow: "hidden",
                boxShadow: isLight
                  ? "0 2px 8px rgba(0,0,0,0.07), 0 0 0 1px rgba(0,0,0,0.03)"
                  : "0 2px 12px rgba(0,0,0,0.3)",
              }}>
                {/* Status banner */}
                <div style={{
                  padding: "10px 22px",
                  background: selectedTicket.status === "open"
                    ? (isLight ? "linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%)" : "linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(6,182,212,0.08) 100%)")
                    : (isLight ? "#f8fafc" : "rgba(255,255,255,0.02)"),
                  borderBottom: `1px solid ${isLight
                    ? (selectedTicket.status === "open" ? "#a7f3d0" : "#e2e8f0")
                    : t.border}`,
                  display: "flex", alignItems: "center", justifyContent: "space-between",
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{
                      fontFamily: "'DM Mono', monospace", fontSize: 10,
                      fontWeight: 700, letterSpacing: "0.06em",
                      color: isLight
                        ? (selectedTicket.status === "open" ? "#065f46" : "#64748b")
                        : t.textMuted,
                    }}>
                      #TCK_{String(selectedTicket.id).padStart(4, "0")}
                    </span>
                    <span style={{ color: t.border }}>·</span>
                    <StatusBadge status={selectedTicket.status} isLight={isLight} />
                  </div>
                  <span style={{ fontSize: 10, color: t.textMuted, fontFamily: "'DM Mono', monospace" }}>
                    {new Date(selectedTicket.created_at).toLocaleString("en-GB", {
                      day: "2-digit", month: "short", year: "numeric",
                      hour: "2-digit", minute: "2-digit"
                    })}
                  </span>
                </div>

                {/* Title + meta */}
                <div style={{ padding: "20px 22px 0" }}>
                  <h1 style={{
                    margin: "0 0 12px", fontSize: 19, fontWeight: 700,
                    color: t.textPrimary, lineHeight: 1.25, letterSpacing: "-0.025em",
                  }}>
                    {selectedTicket.title}
                  </h1>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <Avatar name={selectedTicket.creator_name || "?"} />
                    <span style={{ fontSize: 12, color: t.textSecondary, fontWeight: 500 }}>
                      {selectedTicket.creator_name}
                    </span>
                    <span style={{
                      fontSize: 9, fontWeight: 700, letterSpacing: "0.07em",
                      padding: "2px 6px", borderRadius: 3,
                      background: isLight ? "#f1f5f9" : "rgba(255,255,255,0.05)",
                      border: `1px solid ${t.border}`,
                      color: t.textMuted, textTransform: "uppercase",
                    }}>
                      Requester
                    </span>
                  </div>
                </div>

                {/* Description body */}
                <div style={{ padding: "16px 22px 22px" }}>
                  <div style={{
                    background: isLight ? "#f8fafc" : "rgba(255,255,255,0.025)",
                    border: `1px solid ${isLight ? "#e2e8f0" : "rgba(255,255,255,0.06)"}`,
                    borderLeft: `3px solid ${isLight ? "#059669" : "#06b6d4"}`,
                    borderRadius: "0 8px 8px 0",
                    padding: "14px 18px",
                    fontSize: 13, lineHeight: 1.75,
                    color: isLight ? "#334155" : "#94a3b8",
                    whiteSpace: "pre-wrap",
                  }}>
                    {selectedTicket.description}
                    {selectedTicket.image_url && (() => {
                    let urls = [];
                    try { 
                      urls = JSON.parse(selectedTicket.image_url); 
                    } catch { 
                      urls = [selectedTicket.image_url]; // Fallback for older tickets
                    }
                    
                    return (
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginTop: 16 }}>
                        {urls.map((url, idx) => (
                          <img 
                            key={idx}
                            src={`http://localhost:21081${url}`} 
                            alt={`Attachment ${idx + 1}`} 
                            style={{ 
                              maxWidth: "100%", 
                              maxHeight: "300px", 
                              borderRadius: 8, 
                              border: `1px solid ${t.border}`,
                              objectFit: "cover"
                            }}
                          />
                        ))}
                      </div>
                    );
                  })()}
                  </div>
                  {/* {selectedTicket.image_url && (
                    <div style={{ marginTop: 16 }}>
                      <img 
                        src={`http://localhost:21081${selectedTicket.image_url}`} 
                        alt="Ticket attachment" 
                        style={{ maxWidth: "100%", borderRadius: 8, border: `1px solid ${t.border}` }}
                      />
                    </div> */}
                  {/* )} */}
                </div>
              </div>

              {/* ── Conversation thread ── */}
              <div style={{
                background: t.surface,
                border: `1px solid ${t.border}`,
                borderRadius: 14, overflow: "hidden",
                boxShadow: isLight
                  ? "0 2px 8px rgba(0,0,0,0.07), 0 0 0 1px rgba(0,0,0,0.03)"
                  : "0 2px 12px rgba(0,0,0,0.3)",
              }}>
                {/* Thread header */}
                <div style={{
                  padding: "13px 22px",
                  background: isLight ? "#f8fafc" : "rgba(255,255,255,0.02)",
                  borderBottom: `1px solid ${t.border}`,
                  display: "flex", alignItems: "center", gap: 10,
                }}>
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style={{ opacity: 0.5 }}>
                    <path d="M2 3h10v7a1 1 0 01-1 1H3a1 1 0 01-1-1V3zM2 3l5 4.5L12 3" stroke={t.textSecondary} strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  <span style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.09em", color: t.textSecondary }}>
                    Conversation Thread
                  </span>
                  <span style={{
                    marginLeft: "auto",
                    fontSize: 10, fontWeight: 700,
                    fontFamily: "'DM Mono', monospace",
                    background: isLight ? "#e2e8f0" : "rgba(255,255,255,0.07)",
                    color: t.textSecondary,
                    padding: "2px 8px", borderRadius: 10,
                  }}>
                    {replies.length} {replies.length === 1 ? "reply" : "replies"}
                  </span>
                </div>

                {/* Messages */}
                <div style={{ padding: "20px 22px", display: "flex", flexDirection: "column", gap: 0 }}>
                  {replies.length === 0 ? (
                    <div style={{
                      textAlign: "center", padding: "32px 0",
                      display: "flex", flexDirection: "column", alignItems: "center", gap: 10,
                    }}>
                      <div style={{
                        width: 40, height: 40, borderRadius: "50%",
                        background: isLight ? "#f1f5f9" : "rgba(255,255,255,0.04)",
                        border: `1px dashed ${t.border}`,
                        display: "flex", alignItems: "center", justifyContent: "center",
                      }}>
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style={{ opacity: 0.4 }}>
                          <path d="M3 4h10v6a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 4l5 4 5-4" stroke={t.textSecondary} strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      </div>
                      <span style={{ fontSize: 12, color: t.textMuted }}>No replies yet — be the first to respond.</span>
                    </div>
                  ) : replies.map((reply, idx) => {
                    const isAdmin = reply.replier_role === "admin";
                    const isLast = idx === replies.length - 1;
                    return (
                      <div key={reply.id} style={{ display: "flex", gap: 12, marginBottom: isLast ? 0 : 16 }}>
                        {/* Avatar + connector line */}
                        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", flexShrink: 0 }}>
                          <Avatar name={reply.replier_name || "?"} />
                          {!isLast && (
                            <div style={{
                              width: 1, flex: 1, marginTop: 6,
                              background: isLight ? "#e2e8f0" : "rgba(255,255,255,0.06)",
                              minHeight: 16,
                            }} />
                          )}
                        </div>

                        {/* Bubble */}
                        <div style={{ flex: 1, paddingBottom: isLast ? 0 : 4 }}>
                          {/* Sender row */}
                          <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 7 }}>
                            <span style={{ fontSize: 13, fontWeight: 600, color: t.textPrimary }}>
                              {reply.replier_name}
                            </span>
                            <span style={{
                              fontSize: 9, fontWeight: 800, letterSpacing: "0.09em",
                              textTransform: "uppercase", padding: "2px 6px", borderRadius: 3,
                              background: isAdmin
                                ? (isLight ? "#fff1f2" : "rgba(239,68,68,0.1)")
                                : (isLight ? "#f0f9ff" : "rgba(99,102,241,0.1)"),
                              color: isAdmin
                                ? (isLight ? "#be123c" : "#f87171")
                                : (isLight ? "#0369a1" : "#a5b4fc"),
                              border: `1px solid ${isAdmin
                                ? (isLight ? "#fecdd3" : "rgba(239,68,68,0.25)")
                                : (isLight ? "#bae6fd" : "rgba(99,102,241,0.25)")}`,
                            }}>
                              {reply.replier_role}
                            </span>
                            <span style={{ marginLeft: "auto", fontSize: 10, color: t.textMuted, fontFamily: "'DM Mono', monospace" }}>
                              {new Date(reply.created_at).toLocaleString("en-GB", {
                                day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit"
                              })}
                            </span>
                          </div>

                          {/* Message bubble */}
                          <div style={{
                            padding: "11px 15px",
                            borderRadius: "4px 12px 12px 12px",
                            background: isAdmin
                              ? (isLight ? "#f0fdf4" : "rgba(16,185,129,0.07)")
                              : (isLight ? "#f8fafc" : "rgba(255,255,255,0.04)"),
                            border: `1px solid ${isAdmin
                              ? (isLight ? "#bbf7d0" : "rgba(16,185,129,0.18)")
                              : (isLight ? "#e2e8f0" : "rgba(255,255,255,0.07)")}`,
                            fontSize: 13, lineHeight: 1.7,
                            color: isLight ? "#1e293b" : "#cbd5e1",
                            whiteSpace: "pre-wrap",
                            position: "relative",
                          }}>
                            {/* Admin accent pip */}
                            {isAdmin && (
                              <div style={{
                                position: "absolute", top: 0, left: 0, bottom: 0,
                                width: 3, borderRadius: "4px 0 0 12px",
                                background: isLight ? "#10b981" : "#059669",
                              }} />
                            )}
                            <span style={{ marginLeft: isAdmin ? 8 : 0 }}>
                              {reply.message}
                            </span>
                            {/* --- NEW: RENDER REPLY IMAGES --- */}
                            {reply.image_url && (() => {
                              let urls = [];
                              try { urls = JSON.parse(reply.image_url); } 
                              catch { urls = [reply.image_url]; }
                              
                              return (
                                <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginTop: reply.message ? 12 : 0, marginLeft: isAdmin ? 8 : 0 }}>
                                  {urls.map((url, i) => (
                                    <img 
                                      key={i}
                                      src={`http://localhost:21081${url}`} 
                                      alt="Reply attachment" 
                                      style={{ maxWidth: "100%", maxHeight: "250px", borderRadius: 6, objectFit: "cover", border: `1px solid ${t.border}` }}
                                    />
                                  ))}
                                </div>
                              );
                            })()}
                            {/* --------------------------------- */}

                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* ── Compose reply ── */}
                <div style={{
                  borderTop: `1px solid ${t.border}`,
                  background: isLight ? "#fafafa" : "rgba(255,255,255,0.01)",
                  padding: "16px 22px 20px",
                }}>
                  <form onSubmit={handlePostReply}>
                    <div style={{
                      background: t.surface,
                      border: `1px solid ${isLight ? "#cbd5e1" : "rgba(255,255,255,0.1)"}`,
                      borderRadius: 10, overflow: "hidden",
                      boxShadow: isLight ? "0 1px 4px rgba(0,0,0,0.06)" : "none",
                      transition: "border-color 0.15s, box-shadow 0.15s",
                    }}
                    onFocusCapture={e => {
                      e.currentTarget.style.borderColor = t.accent;
                      e.currentTarget.style.boxShadow = `0 0 0 3px ${t.accentBg}`;
                    }}
                    onBlurCapture={e => {
                      e.currentTarget.style.borderColor = isLight ? "#cbd5e1" : "rgba(255,255,255,0.1)";
                      e.currentTarget.style.boxShadow = isLight ? "0 1px 4px rgba(0,0,0,0.06)" : "none";
                    }}
                    >
                      <textarea
                        rows={3} value={replyText}
                        onChange={e => setReplyText(e.target.value)}
                        placeholder="Write your reply here…"
                        style={{
                          width: "100%", boxSizing: "border-box",
                          background: "transparent", border: "none",
                          padding: "12px 14px", resize: "none",
                          fontSize: 13, lineHeight: 1.65, color: t.textPrimary,
                          outline: "none", fontFamily: "inherit",
                        }}
                      />
                      <div style={{
                        padding: "10px 14px",
                        borderTop: `1px solid ${t.border}`,
                        display: "flex", justifyContent: "space-between", alignItems: "center",
                        background: isLight ? "#f8fafc" : "rgba(255,255,255,0.02)",
                      }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                          <span style={{ fontSize: 11, color: t.textMuted }}>Markdown supported</span>
                          <label style={{ cursor: "pointer", fontSize: 11, color: t.textSecondary, display: "flex", alignItems: "center", gap: 5 }}>
                            <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
                              <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
                            </svg>
                            Attach
                            <input 
                              type="file" multiple accept="image/*" 
                              style={{ display: "none" }} 
                              value={replyImageFiles.length === 0 ? "" : undefined} // <-- ADD THIS LINE
                              onChange={e => setReplyImageFiles(Array.from(e.target.files))} 
                            />
                          </label>
                          {replyImageFiles.length > 0 && (
                            <span style={{ fontSize: 11, color: t.accent }}>{replyImageFiles.length} file(s)</span>
                          )}
                        </div>
                        <span style={{ fontSize: 11, color: t.textMuted }}>
                          Markdown supported
                        </span>
                        <button
                          type="submit" className="tk-primary"
                          style={{
                            background: t.accent, border: "none", color: "#fff",
                            borderRadius: 7, padding: "7px 16px",
                            fontSize: 12, fontWeight: 700, cursor: "pointer",
                            transition: "background 0.15s",
                            display: "flex", alignItems: "center", gap: 7,
                            letterSpacing: "0.02em",
                            boxShadow: isLight ? "0 1px 4px rgba(5,150,105,0.35)" : "none",
                          }}
                        >
                          <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
                            <path d="M1.5 6.5h10M7.5 2l4.5 4.5L7.5 11" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                          </svg>
                          Send Reply
                        </button>
                      </div>
                    </div>
                  </form>
                </div>
              </div>
            </div>

            {/* ════ SIDEBAR ════ */}
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

              {/* ── Status card ── */}
              <div style={{
                background: t.surface,
                border: `1px solid ${t.border}`,
                borderRadius: 14, overflow: "hidden",
                boxShadow: isLight
                  ? "0 2px 8px rgba(0,0,0,0.07), 0 0 0 1px rgba(0,0,0,0.03)"
                  : "0 2px 12px rgba(0,0,0,0.3)",
              }}>
                {/* Colored header based on status */}
                <div style={{
                  padding: "14px 18px",
                  background: selectedTicket.status === "open"
                    ? (isLight ? "linear-gradient(135deg, #059669, #0891b2)" : "linear-gradient(135deg, rgba(5,150,105,0.2), rgba(8,145,178,0.15))")
                    : (isLight ? "#f1f5f9" : "rgba(255,255,255,0.03)"),
                  borderBottom: `1px solid ${isLight
                    ? (selectedTicket.status === "open" ? "rgba(0,0,0,0.1)" : "#e2e8f0")
                    : t.border}`,
                  display: "flex", alignItems: "center", justifyContent: "space-between",
                }}>
                  <span style={{
                    fontSize: 10, fontWeight: 800, textTransform: "uppercase",
                    letterSpacing: "0.1em",
                    color: (isLight && selectedTicket.status === "open") ? "#fff" : t.textSecondary,
                  }}>
                    Ticket Status
                  </span>
                  <StatusBadge status={selectedTicket.status} isLight={isLight} />
                </div>

                <div style={{ padding: "16px 18px", display: "flex", flexDirection: "column", gap: 10 }}>
                  {/* Resolve button */}
                  <button
                    onClick={() => handleToggleStatus("resolved")}
                    disabled={selectedTicket.status === "resolved"}
                    className={selectedTicket.status !== "resolved" ? "tk-primary" : ""}
                    style={{
                      width: "100%", padding: "10px 14px",
                      background: selectedTicket.status === "resolved"
                        ? (isLight ? "#f1f5f9" : "rgba(255,255,255,0.04)")
                        : "#059669",
                      color: selectedTicket.status === "resolved" ? t.textMuted : "#fff",
                      border: selectedTicket.status === "resolved" ? `1px solid ${t.border}` : "none",
                      borderRadius: 8, fontSize: 12, fontWeight: 700,
                      cursor: selectedTicket.status === "resolved" ? "not-allowed" : "pointer",
                      transition: "background 0.15s",
                      display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
                      letterSpacing: "0.02em",
                      boxShadow: (selectedTicket.status !== "resolved" && isLight)
                        ? "0 2px 6px rgba(5,150,105,0.35)" : "none",
                    }}
                  >
                    <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
                      <path d="M2 6.5l3.5 3.5 5.5-6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                    {selectedTicket.status === "resolved" ? "Already Resolved" : "Mark as Resolved"}
                  </button>

                  {/* Re-open button */}
                  <button
                    onClick={() => handleToggleStatus("open")}
                    disabled={selectedTicket.status === "open"}
                    className="tk-ghost"
                    style={{
                      width: "100%", padding: "9px 14px",
                      background: "transparent",
                      color: selectedTicket.status === "open" ? t.textMuted : t.textPrimary,
                      border: `1px solid ${t.border}`,
                      borderRadius: 8, fontSize: 12, fontWeight: 500,
                      cursor: selectedTicket.status === "open" ? "not-allowed" : "pointer",
                      transition: "background 0.15s",
                      display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
                    }}
                  >
                    <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                      <path d="M10.5 6A4.5 4.5 0 1 1 6 1.5M10.5 1.5v4H6.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                    Re-open Ticket
                  </button>
                </div>
              </div>

              {/* ── Meta details card ── */}
              <div style={{
                background: t.surface,
                border: `1px solid ${t.border}`,
                borderRadius: 14, overflow: "hidden",
                boxShadow: isLight
                  ? "0 2px 8px rgba(0,0,0,0.07), 0 0 0 1px rgba(0,0,0,0.03)"
                  : "0 2px 12px rgba(0,0,0,0.3)",
              }}>
                <div style={{
                  padding: "13px 18px",
                  background: isLight ? "#f8fafc" : "rgba(255,255,255,0.02)",
                  borderBottom: `1px solid ${t.border}`,
                  display: "flex", alignItems: "center", gap: 8,
                }}>
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{ opacity: 0.5 }}>
                    <rect x="1" y="1" width="10" height="10" rx="2" stroke={t.textSecondary} strokeWidth="1.2"/>
                    <path d="M3.5 4.5h5M3.5 6.5h3.5" stroke={t.textSecondary} strokeWidth="1.2" strokeLinecap="round"/>
                  </svg>
                  <span style={{ fontSize: 10, fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.1em", color: t.textSecondary }}>
                    Ticket Info
                  </span>
                </div>

                <div style={{ padding: "6px 0" }}>
                  {[
                    {
                      icon: (
                        <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                          <path d="M2 2h8M2 5.5h8M2 9h5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
                        </svg>
                      ),
                      label: "Ticket ID",
                      value: `#TCK_${String(selectedTicket.id).padStart(4, "0")}`,
                      mono: true, accent: true,
                    },
                    {
                      icon: (
                        <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                          <circle cx="6" cy="4" r="2.5" stroke="currentColor" strokeWidth="1.2"/>
                          <path d="M1.5 10.5c0-2.485 2.015-4.5 4.5-4.5s4.5 2.015 4.5 4.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
                        </svg>
                      ),
                      label: "Submitted by",
                      value: selectedTicket.creator_name,
                    },
                    {
                      icon: (
                        <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                          <rect x="1" y="2" width="10" height="9" rx="1.5" stroke="currentColor" strokeWidth="1.2"/>
                          <path d="M4 1v2M8 1v2M1 5h10" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
                        </svg>
                      ),
                      label: "Created",
                      value: new Date(selectedTicket.created_at).toLocaleDateString("en-GB", {
                        day: "2-digit", month: "short", year: "numeric"
                      }), mono: true,
                    },
                    {
                      icon: (
                        <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                          <path d="M2 2h8v6a1 1 0 01-1 1H3a1 1 0 01-1-1V2zM2 2l4 3.5L10 2" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      ),
                      label: "Replies",
                      value: String(replies.length), mono: true,
                    },
                  ].map(({ icon, label, value, mono, accent: isAccent }, idx, arr) => (
                    <div key={label} style={{
                      display: "flex", alignItems: "center",
                      padding: "11px 18px",
                      borderBottom: idx < arr.length - 1 ? `1px solid ${t.border}` : "none",
                      gap: 10,
                    }}>
                      <div style={{ color: t.textMuted, flexShrink: 0, lineHeight: 0 }}>{icon}</div>
                      <span style={{ fontSize: 11, color: t.textSecondary, flex: 1 }}>{label}</span>
                      <span style={{
                        fontSize: 12, fontWeight: 700,
                        color: isAccent ? t.accent : t.textPrimary,
                        fontFamily: mono ? "'DM Mono', monospace" : "inherit",
                      }}>
                        {value}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* ── Activity timeline hint ── */}
              <div style={{
                padding: "14px 18px",
                background: isLight
                  ? "linear-gradient(135deg, #ecfdf5, #eff6ff)"
                  : "rgba(6,182,212,0.05)",
                border: `1px solid ${isLight ? "#a7f3d0" : "rgba(6,182,212,0.15)"}`,
                borderRadius: 12,
                display: "flex", alignItems: "flex-start", gap: 12,
              }}>
                <div style={{
                  width: 30, height: 30, borderRadius: "50%", flexShrink: 0,
                  background: isLight ? "#059669" : "rgba(6,182,212,0.15)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                }}>
                  <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
                    <circle cx="6.5" cy="6.5" r="5.5" stroke={isLight ? "#fff" : "#22d3ee"} strokeWidth="1.2"/>
                    <path d="M6.5 4v3l1.5 1.5" stroke={isLight ? "#fff" : "#22d3ee"} strokeWidth="1.2" strokeLinecap="round"/>
                  </svg>
                </div>
                <div>
                  <p style={{ margin: 0, fontSize: 11, fontWeight: 600, color: isLight ? "#065f46" : "#22d3ee", marginBottom: 2 }}>
                    Awaiting response
                  </p>
                  <p style={{ margin: 0, fontSize: 11, color: isLight ? "#047857" : "#5eead4", lineHeight: 1.5 }}>
                    Last activity {new Date(selectedTicket.updated_at || selectedTicket.created_at).toLocaleDateString("en-GB", { day: "2-digit", month: "short" })}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}