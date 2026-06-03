// src/components/tabs/TicketsTab.jsx
import React, { useState, useEffect } from "react";

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
        background: hovered ? (isLight ? "#f0fdf8" : "rgba(6,182,212,0.04)") : "transparent",
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

export default function TicketsTab({ isLight = false }) {
  const [tickets, setTickets] = useState([]);
  const [selectedTicket, setSelectedTicket] = useState(null);
  const [replies, setReplies] = useState([]);
  
  // Creation state
  const [showForm, setShowForm] = useState(false);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [imageFiles, setImageFiles] = useState([]);
  
  // Reply state
  const [replyText, setReplyText] = useState("");
  const [replyImageFiles, setReplyImageFiles] = useState([]);

  // Editing Ticket State
  const [isEditingTicket, setIsEditingTicket] = useState(false);
  const [editTicketTitle, setEditTicketTitle] = useState("");
  const [editTicketDesc, setEditTicketDesc] = useState("");
  const [editTicketKeptImages, setEditTicketKeptImages] = useState([]);
  const [editTicketNewImages, setEditTicketNewImages] = useState([]);

  // Editing Reply State
  const [editingReplyId, setEditingReplyId] = useState(null);
  const [editReplyText, setEditReplyText] = useState("");
  const [editReplyKeptImages, setEditReplyKeptImages] = useState([]);
  const [editReplyNewImages, setEditReplyNewImages] = useState([]);

  const userId = localStorage.getItem("user_id");
  const userRole = localStorage.getItem("user_role");
  const token = localStorage.getItem("access_token");

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
      if (d.status === "success") { 
        setSelectedTicket(d.data); 
        setReplies(d.data.replies || []); 
        setIsEditingTicket(false);
        setEditingReplyId(null);
      }
    } catch (e) { console.error(e); }
  };

  useEffect(() => { loadTickets(); }, []);

  // --- CRUD: TICKET ---
  const handleCreateTicket = async (e) => {
    e.preventDefault();
    if (!title.trim() || !description.trim()) { alert("Please fill in both fields."); return; }
    
    const formData = new FormData();
    formData.append("title", title);
    formData.append("description", description);
    imageFiles.forEach((file) => formData.append("images", file));

    try {
      const r = await fetch("http://localhost:21081/api/tickets", {
        method: "POST",
        headers: { "Authorization": `Bearer ${token}` },
        body: formData,
      });
      if (r.ok) { 
        setTitle(""); setDescription(""); setImageFiles([]); setShowForm(false); loadTickets(); 
      } else {
        const err = await r.json(); alert("Error: " + err.detail);
      }
    } catch (e) { console.error(e); }
  };

  const startEditTicket = () => {
    setEditTicketTitle(selectedTicket.title);
    setEditTicketDesc(selectedTicket.description);
    let urls = [];
    try { urls = JSON.parse(selectedTicket.image_url || "[]"); } catch { urls = selectedTicket.image_url ? [selectedTicket.image_url] : []; }
    setEditTicketKeptImages(urls);
    setEditTicketNewImages([]);
    setIsEditingTicket(true);
  };

  const handleUpdateTicket = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append("title", editTicketTitle);
    formData.append("description", editTicketDesc);
    formData.append("kept_images", JSON.stringify(editTicketKeptImages));
    editTicketNewImages.forEach(f => formData.append("images", f));

    try {
      const r = await fetch(`http://localhost:21081/api/tickets/${selectedTicket.id}`, {
        method: "PUT",
        headers: { "Authorization": `Bearer ${token}` },
        body: formData,
      });
      if (r.ok) loadTicketDetail(selectedTicket.id);
      else { const err = await r.json(); alert("Error: " + err.detail); }
    } catch (e) { console.error(e); }
  };

  const handleDeleteTicket = async () => {
    if (!window.confirm("Are you sure you want to delete this ticket?")) return;
    try {
      const r = await fetch(`http://localhost:21081/api/tickets/${selectedTicket.id}`, {
        method: "DELETE",
        headers: { "Authorization": `Bearer ${token}` }
      });
      if (r.ok) { setSelectedTicket(null); loadTickets(); }
      else { const err = await r.json(); alert("Error: " + err.detail); }
    } catch (e) { console.error(e); }
  };

  const handleToggleStatus = async (statusValue) => {
    try {
      const r = await fetch(`http://localhost:21081/api/tickets/${selectedTicket.id}/status`, {
        method: "POST", headers: { "Content-Type": "application/json", "Authorization": `Bearer ${token}` },
        body: JSON.stringify({ user_id: Number(userId), status: statusValue }),
      });
      if (r.ok) { loadTicketDetail(selectedTicket.id); loadTickets(); }
      else { const err = await r.json(); alert(err.detail); }
    } catch (e) { console.error(e); }
  };

  // --- CRUD: REPLIES ---
  const handlePostReply = async (e) => {
    e.preventDefault();
    if (!replyText.trim() && replyImageFiles.length === 0) return;
    const formData = new FormData();
    formData.append("message", replyText);
    replyImageFiles.forEach((file) => formData.append("images", file));

    try {
      const r = await fetch(`http://localhost:21081/api/tickets/${selectedTicket.id}/replies`, {
        method: "POST",
        headers: { "Authorization": `Bearer ${token}` },
        body: formData,
      });
      if (r.ok) { setReplyText(""); setReplyImageFiles([]); loadTicketDetail(selectedTicket.id); }
      else { const err = await r.json(); alert("Error: " + err.detail); }
    } catch (e) { console.error(e); }
  };

  const startEditReply = (reply) => {
    setEditReplyText(reply.message);
    let urls = [];
    try { urls = JSON.parse(reply.image_url || "[]"); } catch { urls = reply.image_url ? [reply.image_url] : []; }
    setEditReplyKeptImages(urls);
    setEditReplyNewImages([]);
    setEditingReplyId(reply.id);
  };

  const handleUpdateReply = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append("message", editReplyText);
    formData.append("kept_images", JSON.stringify(editReplyKeptImages));
    editReplyNewImages.forEach(f => formData.append("images", f));

    try {
      const r = await fetch(`http://localhost:21081/api/tickets/replies/${editingReplyId}`, {
        method: "PUT",
        headers: { "Authorization": `Bearer ${token}` },
        body: formData,
      });
      if (r.ok) { setEditingReplyId(null); loadTicketDetail(selectedTicket.id); }
      else { const err = await r.json(); alert("Error: " + err.detail); }
    } catch (e) { console.error(e); }
  };

  const handleDeleteReply = async (replyId) => {
    if (!window.confirm("Delete this reply?")) return;
    try {
      const r = await fetch(`http://localhost:21081/api/tickets/replies/${replyId}`, {
        method: "DELETE",
        headers: { "Authorization": `Bearer ${token}` }
      });
      if (r.ok) loadTicketDetail(selectedTicket.id);
      else { const err = await r.json(); alert("Error: " + err.detail); }
    } catch (e) { console.error(e); }
  };

  // ── Theme tokens
  const t = isLight ? {
    pageBg: "#f1f5f9", surface: "#ffffff", surfaceRaised: "#f8fafc",
    border: "#e2e8f0", borderStrong: "#cbd5e1", accent: "#059669",
    accentDark: "#047857", accentBg: "#ecfdf5", accentBorder: "#6ee7b7",
    tableHead: "#065f46", tableHeadBg: "#059669", textPrimary: "#0f172a",
    textSecondary: "#475569", textMuted: "#94a3b8", inputBg: "#ffffff",
    inputBorder: "#cbd5e1", codeBg: "#f1f5f9", shadow: "0 1px 3px rgba(0,0,0,0.08)",
  } : {
    pageBg: "transparent", surface: "#0f172a", surfaceRaised: "#111827",
    border: "rgba(255,255,255,0.07)", borderStrong: "rgba(255,255,255,0.12)",
    accent: "#06b6d4", accentDark: "#0891b2", accentBg: "rgba(6,182,212,0.08)",
    accentBorder: "rgba(6,182,212,0.3)", tableHead: "#e2e8f0", tableHeadBg: "rgba(255,255,255,0.03)",
    textPrimary: "#f1f5f9", textSecondary: "#64748b", textMuted: "#475569",
    inputBg: "#0c1524", inputBorder: "rgba(255,255,255,0.08)", codeBg: "rgba(255,255,255,0.02)",
    shadow: "none",
  };

  const inputStyle = {
    width: "100%", boxSizing: "border-box", background: t.inputBg, border: `1px solid ${t.inputBorder}`,
    borderRadius: 7, padding: "9px 12px", fontSize: 13, color: t.textPrimary,
    outline: "none", transition: "border-color 0.15s, box-shadow 0.15s", fontFamily: "inherit",
  };

  // Helper for rendering image grids
  const renderImageGrid = (urls, isEditing = false, removeFn = null) => {
    if (!urls || urls.length === 0) return null;
    return (
      <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginTop: 12 }}>
        {urls.map((url, idx) => (
          <div key={idx} style={{ position: "relative" }}>
            <img 
              src={`http://localhost:21081${url}`} 
              alt="Attachment" 
              style={{ maxWidth: "200px", maxHeight: "200px", borderRadius: 8, border: `1px solid ${t.border}`, objectFit: "cover" }}
            />
            {isEditing && (
              <button 
                type="button"
                onClick={() => removeFn(url)}
                style={{ 
                  position: "absolute", top: -8, right: -8, background: "#ef4444", color: "#fff", 
                  border: "none", borderRadius: "50%", width: 22, height: 22, cursor: "pointer", 
                  fontSize: 12, display: "flex", alignItems: "center", justifyContent: "center", boxShadow: "0 2px 4px rgba(0,0,0,0.2)"
                }}
              >×</button>
            )}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div style={{ fontFamily: "'Inter var', 'Inter', sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');
        .tk-input:focus { border-color: ${t.accent} !important; box-shadow: 0 0 0 3px ${t.accentBg} !important; }
        .tk-ghost:hover { background: ${isLight ? "#f1f5f9" : "rgba(255,255,255,0.05)"} !important; }
        .tk-primary:hover { background: ${t.accentDark} !important; }
        .tk-icon-btn:hover { background: ${isLight ? "#e2e8f0" : "rgba(255,255,255,0.1)"}; border-radius: 4px; }
      `}</style>

      {!selectedTicket ? (
        <>
          {/* ── Page Header ── */}
          <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 24, gap: 16, flexWrap: "wrap" }}>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 5 }}>
                <div style={{ width: 30, height: 30, borderRadius: 8, background: isLight ? "#059669" : "rgba(6,182,212,0.12)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><rect x="1" y="1" width="12" height="12" rx="2.5" stroke={isLight ? "#fff" : "#22d3ee"} strokeWidth="1.3"/><path d="M4 5h6M4 7.5h4" stroke={isLight ? "#fff" : "#22d3ee"} strokeWidth="1.3" strokeLinecap="round"/></svg>
                </div>
                <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: t.textPrimary }}>Support Tickets</h2>
              </div>
              <p style={{ margin: 0, fontSize: 12, color: t.textSecondary, paddingLeft: 40 }}>Report technical issues, server errors, or platform curriculum feedback.</p>
            </div>
            <button onClick={() => setShowForm(!showForm)} className="tk-primary" style={{ display: "flex", alignItems: "center", gap: 7, background: showForm ? (isLight ? "#f1f5f9" : "rgba(255,255,255,0.06)") : t.accent, color: showForm ? t.textSecondary : "#fff", border: showForm ? `1px solid ${t.border}` : "none", borderRadius: 7, padding: "8px 16px", fontSize: 12, fontWeight: 600, cursor: "pointer" }}>
              {showForm ? "Cancel" : "New Ticket"}
            </button>
          </div>

          {/* ── Create Form ── */}
          {showForm && (
            <div style={{ background: t.surface, border: `1px solid ${t.border}`, borderRadius: 12, padding: 24, marginBottom: 24, position: "relative" }}>
              <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 3, background: "linear-gradient(90deg, #10b981, #06b6d4, #818cf8)" }} />
              <h3 style={{ margin: "0 0 20px", fontSize: 12, fontWeight: 700, color: t.accent, textTransform: "uppercase" }}>New Support Request</h3>
              <form onSubmit={handleCreateTicket} style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <div>
                  <label style={{ display: "block", fontSize: 11, fontWeight: 600, color: t.textSecondary, textTransform: "uppercase", marginBottom: 6 }}>Issue Title</label>
                  <input type="text" required value={title} onChange={e => setTitle(e.target.value)} className="tk-input" style={inputStyle} />
                </div>
                <div>
                  <label style={{ display: "block", fontSize: 11, fontWeight: 600, color: t.textSecondary, textTransform: "uppercase", marginBottom: 6 }}>Detailed Description</label>
                  <textarea required rows={4} value={description} onChange={e => setDescription(e.target.value)} className="tk-input" style={{ ...inputStyle, resize: "vertical" }} />
                </div>
                <div>
                  <label style={{ display: "block", fontSize: 11, fontWeight: 600, color: t.textSecondary, textTransform: "uppercase", marginBottom: 6 }}>Attach Screenshots</label>
                  <input type="file" accept="image/*" multiple onChange={e => setImageFiles(Array.from(e.target.files))} style={{ fontSize: 12, color: t.textSecondary }} />
                </div>
                <div style={{ display: "flex", justifyContent: "flex-end", gap: 10 }}>
                  <button type="button" className="tk-ghost" onClick={() => setShowForm(false)} style={{ background: "transparent", border: `1px solid ${t.border}`, color: t.textSecondary, borderRadius: 7, padding: "8px 16px", fontSize: 12, cursor: "pointer" }}>Cancel</button>
                  <button type="submit" className="tk-primary" style={{ background: t.accent, border: "none", color: "#fff", borderRadius: 7, padding: "8px 20px", fontSize: 12, fontWeight: 600, cursor: "pointer" }}>Submit Ticket</button>
                </div>
              </form>
            </div>
          )}

          {/* ── Tickets Table ── */}
          <div style={{ background: t.surface, border: `1px solid ${t.border}`, borderRadius: 12, overflow: "hidden", boxShadow: t.shadow }}>
            <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 680 }}>
              <thead>
                <tr style={{ background: t.tableHeadBg, borderBottom: `1px solid ${t.border}` }}>
                  {["Ticket ID", "Title", "Requester", "Status", "Last Updated", ""].map((h, i) => (
                    <th key={i} style={{ padding: "12px 20px", textAlign: i === 3 ? "center" : i === 5 ? "right" : "left", fontSize: 10, fontWeight: 700, textTransform: "uppercase", color: isLight ? "#fff" : t.textSecondary, fontFamily: "'DM Mono', monospace" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {tickets.length === 0 ? (
                  <tr><td colSpan={6} style={{ textAlign: "center", padding: "48px 20px", color: t.textSecondary, fontSize: 13 }}>No support tickets yet</td></tr>
                ) : tickets.map(t_ => <TicketRow key={t_.id} ticket={t_} onView={loadTicketDetail} isLight={isLight} />)}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        /* ══════════════════════════════════════════
           TICKET DETAIL VIEW 
        ══════════════════════════════════════════ */
        <>
          <div style={{ marginBottom: 20, display: "flex", alignItems: "center", gap: 8 }}>
            <button onClick={() => setSelectedTicket(null)} style={{ background: "transparent", border: "none", color: t.accent, fontSize: 12, fontWeight: 600, cursor: "pointer" }}>All Tickets</button>
            <span style={{ color: t.textSecondary }}>›</span>
            <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 11, color: t.textMuted }}>#TCK_{String(selectedTicket.id).padStart(4, "0")}</span>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 272px", gap: 18, alignItems: "start" }}>
            {/* ════ MAIN COLUMN ════ */}
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              
              {/* ── Hero ticket card ── */}
              <div style={{ background: t.surface, border: `1px solid ${t.border}`, borderRadius: 14, overflow: "hidden", boxShadow: t.shadow }}>
                
                {isEditingTicket ? (
                  <form onSubmit={handleUpdateTicket} style={{ padding: 22, display: "flex", flexDirection: "column", gap: 16 }}>
                    <h3 style={{ margin: 0, fontSize: 14, color: t.textPrimary }}>Edit Ticket</h3>
                    <input type="text" required value={editTicketTitle} onChange={e => setEditTicketTitle(e.target.value)} className="tk-input" style={inputStyle} />
                    <textarea required rows={4} value={editTicketDesc} onChange={e => setEditTicketDesc(e.target.value)} className="tk-input" style={{ ...inputStyle, resize: "vertical" }} />
                    
                    <div>
                      <span style={{ fontSize: 11, color: t.textSecondary, fontWeight: 600 }}>Kept Images:</span>
                      {renderImageGrid(editTicketKeptImages, true, (url) => setEditTicketKeptImages(prev => prev.filter(u => u !== url)))}
                    </div>

                    <div>
                      <label style={{ display: "block", fontSize: 11, color: t.textSecondary, fontWeight: 600, marginBottom: 5 }}>Add New Images</label>
                      <input type="file" multiple accept="image/*" onChange={e => setEditTicketNewImages(Array.from(e.target.files))} style={{ fontSize: 12, color: t.textPrimary }} />
                    </div>

                    <div style={{ display: "flex", justifyContent: "flex-end", gap: 10 }}>
                      <button type="button" onClick={() => setIsEditingTicket(false)} style={{ background: "transparent", border: `1px solid ${t.border}`, color: t.textPrimary, padding: "6px 12px", borderRadius: 6, cursor: "pointer" }}>Cancel</button>
                      <button type="submit" style={{ background: t.accent, border: "none", color: "#fff", padding: "6px 12px", borderRadius: 6, cursor: "pointer", fontWeight: 600 }}>Save Changes</button>
                    </div>
                  </form>
                ) : (
                  <>
                    <div style={{ padding: "10px 22px", background: isLight ? "#f8fafc" : "rgba(255,255,255,0.02)", borderBottom: `1px solid ${t.border}`, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 10, fontWeight: 700, color: t.textMuted }}>#TCK_{String(selectedTicket.id).padStart(4, "0")}</span>
                        <StatusBadge status={selectedTicket.status} isLight={isLight} />
                      </div>
                      
                      {/* Ticket Action Buttons (Owner or Admin) */}
                      {(userRole === "admin" || selectedTicket.user_id === Number(userId)) && selectedTicket.status !== "resolved" && (
                        <div style={{ display: "flex", gap: 8 }}>
                          <button onClick={startEditTicket} className="tk-icon-btn" style={{ background: "transparent", border: "none", color: t.textSecondary, cursor: "pointer", padding: 4 }}>
                            ✏️
                          </button>
                          <button onClick={handleDeleteTicket} className="tk-icon-btn" style={{ background: "transparent", border: "none", color: "#ef4444", cursor: "pointer", padding: 4 }}>
                            🗑️
                          </button>
                        </div>
                      )}
                    </div>

                    <div style={{ padding: "20px 22px 0" }}>
                      <h1 style={{ margin: "0 0 12px", fontSize: 19, fontWeight: 700, color: t.textPrimary }}>{selectedTicket.title}</h1>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <Avatar name={selectedTicket.creator_name || "?"} />
                        <span style={{ fontSize: 12, color: t.textSecondary, fontWeight: 500 }}>{selectedTicket.creator_name}</span>
                      </div>
                    </div>

                    <div style={{ padding: "16px 22px 22px" }}>
                      <div style={{ background: isLight ? "#f8fafc" : "rgba(255,255,255,0.025)", border: `1px solid ${t.border}`, borderLeft: `3px solid ${t.accent}`, borderRadius: "0 8px 8px 0", padding: "14px 18px", fontSize: 13, color: isLight ? "#334155" : "#94a3b8", whiteSpace: "pre-wrap" }}>
                        {selectedTicket.description}
                        {selectedTicket.image_url && (() => {
                          let urls = [];
                          try { urls = JSON.parse(selectedTicket.image_url); } catch { urls = [selectedTicket.image_url]; }
                          return renderImageGrid(urls);
                        })()}
                      </div>
                    </div>
                  </>
                )}
              </div>

              {/* ── Conversation thread ── */}
              <div style={{ background: t.surface, border: `1px solid ${t.border}`, borderRadius: 14, overflow: "hidden", boxShadow: t.shadow }}>
                <div style={{ padding: "13px 22px", background: isLight ? "#f8fafc" : "rgba(255,255,255,0.02)", borderBottom: `1px solid ${t.border}`, display: "flex", alignItems: "center", gap: 10 }}>
                  <span style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", color: t.textSecondary }}>Conversation Thread</span>
                  <span style={{ marginLeft: "auto", fontSize: 10, fontWeight: 700, background: isLight ? "#e2e8f0" : "rgba(255,255,255,0.07)", color: t.textSecondary, padding: "2px 8px", borderRadius: 10 }}>
                    {replies.length} replies
                  </span>
                </div>

                <div style={{ padding: "20px 22px", display: "flex", flexDirection: "column", gap: 16 }}>
                  {replies.length === 0 ? (
                    <div style={{ textAlign: "center", color: t.textMuted, fontSize: 12, padding: "20px 0" }}>No replies yet.</div>
                  ) : replies.map((reply) => {
                    const isAdmin = reply.replier_role === "admin";
                    const isOwner = userRole === "admin" || reply.user_id === Number(userId);

                    if (editingReplyId === reply.id) {
                      return (
                        <form key={reply.id} onSubmit={handleUpdateReply} style={{ background: t.pageBg, padding: 16, borderRadius: 8, border: `1px solid ${t.border}` }}>
                          <textarea rows={3} value={editReplyText} onChange={e => setEditReplyText(e.target.value)} className="tk-input" style={{ ...inputStyle, marginBottom: 10 }} />
                          {renderImageGrid(editReplyKeptImages, true, (url) => setEditReplyKeptImages(prev => prev.filter(u => u !== url)))}
                          <div style={{ marginTop: 10 }}>
                            <input type="file" multiple accept="image/*" onChange={e => setEditReplyNewImages(Array.from(e.target.files))} style={{ fontSize: 11, color: t.textPrimary }} />
                          </div>
                          <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 10 }}>
                            <button type="button" onClick={() => setEditingReplyId(null)} style={{ background: "transparent", border: `1px solid ${t.border}`, color: t.textPrimary, padding: "4px 10px", borderRadius: 4, cursor: "pointer", fontSize: 12 }}>Cancel</button>
                            <button type="submit" style={{ background: t.accent, border: "none", color: "#fff", padding: "4px 10px", borderRadius: 4, cursor: "pointer", fontSize: 12, fontWeight: 600 }}>Save</button>
                          </div>
                        </form>
                      );
                    }

                    return (
                      <div key={reply.id} style={{ display: "flex", gap: 12 }}>
                        <Avatar name={reply.replier_name || "?"} />
                        <div style={{ flex: 1 }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 4 }}>
                            <span style={{ fontSize: 13, fontWeight: 600, color: t.textPrimary }}>{reply.replier_name}</span>
                            <span style={{ fontSize: 9, fontWeight: 800, textTransform: "uppercase", padding: "2px 6px", borderRadius: 3, background: isAdmin ? (isLight ? "#fff1f2" : "rgba(239,68,68,0.1)") : (isLight ? "#f0f9ff" : "rgba(99,102,241,0.1)"), color: isAdmin ? (isLight ? "#be123c" : "#f87171") : (isLight ? "#0369a1" : "#a5b4fc") }}>
                              {reply.replier_role}
                            </span>
                            
                            {/* Reply Actions (Edit/Delete) */}
                            {isOwner && selectedTicket.status !== "resolved" && (
                              <div style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
                                <button onClick={() => startEditReply(reply)} className="tk-icon-btn" style={{ background: "none", border: "none", cursor: "pointer", fontSize: 11, color: t.textMuted }}>Edit</button>
                                <button onClick={() => handleDeleteReply(reply.id)} className="tk-icon-btn" style={{ background: "none", border: "none", cursor: "pointer", fontSize: 11, color: "#ef4444" }}>Delete</button>
                              </div>
                            )}
                          </div>

                          <div style={{ padding: "11px 15px", borderRadius: "4px 12px 12px 12px", background: isAdmin ? (isLight ? "#f0fdf4" : "rgba(16,185,129,0.07)") : (isLight ? "#f8fafc" : "rgba(255,255,255,0.04)"), border: `1px solid ${isAdmin ? (isLight ? "#bbf7d0" : "rgba(16,185,129,0.18)") : (isLight ? "#e2e8f0" : "rgba(255,255,255,0.07)")}`, fontSize: 13, color: isLight ? "#1e293b" : "#cbd5e1", whiteSpace: "pre-wrap" }}>
                            {reply.message}
                            {reply.image_url && (() => {
                              let urls = [];
                              try { urls = JSON.parse(reply.image_url); } catch { urls = [reply.image_url]; }
                              return renderImageGrid(urls);
                            })()}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* ── Compose reply ── */}
                {selectedTicket.status === "resolved" ? ( 
                  <div style={{ borderTop: `1px solid ${t.border}`, background: isLight ? "#fafafa" : "rgba(255,255,255,0.01)", padding: "20px 22px", textAlign: "center" }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: t.textMuted, marginBottom: 4 }}>Ticket Closed</div>
                    <div style={{ fontSize: 12, color: t.textSecondary }}>This ticket has been marked as resolved. No further edits or replies can be made.</div>
                  </div>
                ) : (
                <div style={{ borderTop: `1px solid ${t.border}`, background: isLight ? "#fafafa" : "rgba(255,255,255,0.01)", padding: "16px 22px 20px" }}>
                  <form onSubmit={handlePostReply}>
                    <div style={{ background: t.surface, border: `1px solid ${t.border}`, borderRadius: 10, overflow: "hidden" }}>
                      <textarea rows={3} value={replyText} onChange={e => setReplyText(e.target.value)} placeholder="Write your reply here…" style={{ width: "100%", boxSizing: "border-box", background: "transparent", border: "none", padding: "12px 14px", resize: "none", fontSize: 13, color: t.textPrimary, outline: "none", fontFamily: "inherit" }} />
                      <div style={{ padding: "10px 14px", borderTop: `1px solid ${t.border}`, display: "flex", justifyContent: "space-between", alignItems: "center", background: isLight ? "#f8fafc" : "rgba(255,255,255,0.02)" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                          <label style={{ cursor: "pointer", fontSize: 11, color: t.textSecondary, display: "flex", alignItems: "center", gap: 5 }}>
                            📎 Attach
                            <input type="file" multiple accept="image/*" style={{ display: "none" }} value={replyImageFiles.length === 0 ? "" : undefined} onChange={e => setReplyImageFiles(Array.from(e.target.files))} />
                          </label>
                          {replyImageFiles.length > 0 && <span style={{ fontSize: 11, color: t.accent }}>{replyImageFiles.length} file(s)</span>}
                        </div>
                        <button type="submit" className="tk-primary" style={{ background: t.accent, border: "none", color: "#fff", borderRadius: 7, padding: "7px 16px", fontSize: 12, fontWeight: 700, cursor: "pointer" }}>Send Reply</button>
                      </div>
                    </div>
                  </form>
                </div>
                )}
              </div>
            </div>

            {/* ════ SIDEBAR ════ */}
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

              {/* ── Status card ── */}
              <div style={{ background: t.surface, border: `1px solid ${t.border}`, borderRadius: 14, overflow: "hidden", boxShadow: t.shadow }}>
                <div style={{ padding: "14px 18px", background: selectedTicket.status === "open" ? (isLight ? "linear-gradient(135deg, #059669, #0891b2)" : "linear-gradient(135deg, rgba(5,150,105,0.2), rgba(8,145,178,0.15))") : (isLight ? "#f1f5f9" : "rgba(255,255,255,0.03)"), borderBottom: `1px solid ${t.border}`, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <span style={{ fontSize: 10, fontWeight: 800, textTransform: "uppercase", color: (isLight && selectedTicket.status === "open") ? "#fff" : t.textSecondary }}>Ticket Status</span>
                  <StatusBadge status={selectedTicket.status} isLight={isLight} />
                </div>

                {userRole === "admin" ? (
                  <div style={{ padding: "16px 18px", display: "flex", flexDirection: "column", gap: 10 }}>
                    <button onClick={() => handleToggleStatus("resolved")} disabled={selectedTicket.status === "resolved"} style={{ width: "100%", padding: "10px 14px", background: selectedTicket.status === "resolved" ? (isLight ? "#f1f5f9" : "rgba(255,255,255,0.04)") : "#059669", color: selectedTicket.status === "resolved" ? t.textMuted : "#fff", border: selectedTicket.status === "resolved" ? `1px solid ${t.border}` : "none", borderRadius: 8, fontSize: 12, fontWeight: 700, cursor: selectedTicket.status === "resolved" ? "not-allowed" : "pointer" }}>
                      {selectedTicket.status === "resolved" ? "Already Resolved" : "Mark as Resolved"}
                    </button>
                    <button onClick={() => handleToggleStatus("open")} disabled={selectedTicket.status === "open"} style={{ width: "100%", padding: "9px 14px", background: "transparent", color: selectedTicket.status === "open" ? t.textMuted : t.textPrimary, border: `1px solid ${t.border}`, borderRadius: 8, fontSize: 12, fontWeight: 500, cursor: selectedTicket.status === "open" ? "not-allowed" : "pointer" }}>
                      Re-open Ticket
                    </button>
                  </div>
                ) : (
                  <div style={{ padding: "16px 18px", fontSize: 12, color: t.textSecondary, textAlign: 'center' }}>
                    Only administrators can resolve or reopen tickets.
                  </div>
                )}
              </div>
            </div>

          </div>
        </>
      )}
    </div>
  );
}