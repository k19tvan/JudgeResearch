// src/components/tabs/UsersTab.jsx
import React, { useEffect, useState } from "react";

export default function UsersTab({ isLight = false }) {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [actionLoading, setActionLoading] = useState(null);

  const currentUserId = localStorage.getItem("user_id");

  const loadUsersList = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`http://localhost:21081/api/admin/users?admin_id=${currentUserId}`);
      const result = await response.json();
      if (response.ok) {
        setUsers(result.data || []);
      } else {
        setError(result.detail || "Failed to load users list");
      }
    } catch (err) {
      setError("Failed to fetch users: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadUsersList();
  }, []);

  const handleUpdateUserRole = async (targetUserId, newRole) => {
    setActionLoading(targetUserId);
    try {
      const response = await fetch(`http://localhost:21081/api/admin/users/${targetUserId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          admin_id: Number(currentUserId),
          role: newRole,
        }),
      });
      if (response.ok) {
        await loadUsersList();
      } else {
        const result = await response.json();
        alert(result.detail || "Failed to update role");
      }
    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      setActionLoading(null);
    }
  };

  const handleToggleUserStatus = async (targetUserId, currentStatus) => {
    const newStatus = currentStatus === "active" ? "disabled" : "active";
    setActionLoading(targetUserId);
    try {
      const response = await fetch(`http://localhost:21081/api/admin/users/${targetUserId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          admin_id: Number(currentUserId),
          status: newStatus,
        }),
      });
      if (response.ok) {
        await loadUsersList();
      } else {
        const result = await response.json();
        alert(result.detail || "Failed to update status");
      }
    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      setActionLoading(null);
    }
  };

  const t = isLight ? {
    pageBg:       "#f1f5f9",
    surface:      "#ffffff",
    surfaceRaised:"#f8fafc",
    border:       "#e2e8f0",
    borderStrong: "#cbd5e1",
    accent:       "#059669",
    accentDark:   "#047857",
    accentBg:     "#ecfdf5",
    accentBorder: "#6ee7b7",
    tableHead:    "#065f46",
    tableHeadBg:  "#059669",
    textPrimary:  "#0f172a",
    textSecondary:"#475569",
    textMuted:    "#94a3b8",
    inputBg:      "#ffffff",
    inputBorder:  "#cbd5e1",
    shadow:       "0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04)",
  } : {
    pageBg:       "transparent",
    surface:      "#131b2e",
    surfaceRaised:"#182239",
    border:       "rgba(255,255,255,0.12)",
    borderStrong: "rgba(255,255,255,0.2)",
    accent:       "#06b6d4",
    accentDark:   "#0891b2",
    accentBg:     "rgba(6,182,212,0.08)",
    accentBorder: "rgba(6,182,212,0.3)",
    tableHead:    "#e2e8f0",
    tableHeadBg:  "rgba(255,255,255,0.04)",
    textPrimary:  "#f1f5f9",
    textSecondary:"#94a3b8",
    textMuted:    "#64748b",
    inputBg:      "#0d1322",
    inputBorder:  "rgba(255,255,255,0.15)",
    shadow:       "0 4px 20px rgba(0,0,0,0.4)",
  };

  function StatusBadge({ status = "active" }) {
    const isActive = status === "active";
    return (
      <span style={{
        display: "inline-flex", alignItems: "center", gap: 5,
        background: isActive ? (isLight ? "#ecfdf5" : "rgba(16,185,129,0.08)") : (isLight ? "#fff1f2" : "rgba(244,63,94,0.08)"),
        border: `1px solid ${isActive ? (isLight ? "#6ee7b7" : "rgba(16,185,129,0.2)") : (isLight ? "#fecdd3" : "rgba(244,63,94,0.2)")}`,
        color: isActive ? (isLight ? "#065f46" : "#34d399") : (isLight ? "#9f1239" : "#fb7185"),
        fontSize: 10, fontWeight: 700,
        letterSpacing: "0.08em", padding: "3px 9px",
        borderRadius: 4, textTransform: "uppercase",
      }}>
        <span style={{
          width: 5, height: 5, borderRadius: "50%",
          background: isActive ? (isLight ? "#059669" : "#10b981") : (isLight ? "#e11d48" : "#f43f5e"),
          flexShrink: 0
        }} />
        {status}
      </span>
    );
  }

  return (
    <div style={{ fontFamily: "'Inter var', 'Inter', sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');
        .tk-select {
          outline: none;
          transition: border-color 0.15s, box-shadow 0.15s;
        }
        .tk-select:focus {
          border-color: ${t.accent} !important;
          box-shadow: 0 0 0 3px ${t.accentBg} !important;
        }
      `}</style>

      {/* ── Page Header ── */}
      <div style={{
        display: "flex", alignItems: "flex-start",
        justifyContent: "space-between", marginBottom: 24, gap: 16, flexWrap: "wrap",
      }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 5 }}>
            <div style={{
              width: 30, height: 30, borderRadius: 8,
              background: isLight ? "#059669" : "rgba(6,182,212,0.12)",
              border: isLight ? "none" : "1px solid rgba(6,182,212,0.2)",
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={isLight ? "#fff" : "#22d3ee"} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z"/>
              </svg>
            </div>
            <h2 style={{
              margin: 0, fontSize: 18, fontWeight: 700,
              color: t.textPrimary, letterSpacing: "-0.02em",
            }}>
              Manage System Users
            </h2>
          </div>
          <p style={{ margin: 0, fontSize: 12, color: t.textSecondary, lineHeight: 1.5, paddingLeft: 40 }}>
            Modify member roles or temporarily disable / activate user accounts.
          </p>
        </div>
      </div>

      {error && (
        <div style={{
          marginBottom: 20, padding: "10px 14px", borderRadius: 8,
          background: isLight ? "#fff1f2" : "rgba(239,68,68,0.08)",
          border: `1px solid ${isLight ? "#fecdd3" : "rgba(239,68,68,0.2)"}`,
          color: isLight ? "#be123c" : "#f87171", fontSize: 12,
        }}>
          {error}
        </div>
      )}

      {loading ? (
        <div style={{
          background: t.surface, border: `1px solid ${t.border}`, borderRadius: 12, padding: "48px 24px", textAlign: "center", color: t.textSecondary, fontSize: 13,
        }}>
          <p className="animate-pulse">Loading users workspace...</p>
        </div>
      ) : (
        <div style={{
          background: t.surface,
          border: `1px solid ${t.border}`,
          borderRadius: 12, overflow: "hidden",
          boxShadow: t.shadow,
        }}>
          <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 800 }}>
            <thead>
              <tr style={{
                background: isLight ? t.tableHeadBg : t.tableHeadBg,
                borderBottom: `1px solid ${isLight ? t.accentDark + "55" : t.border}`,
              }}>
                {["ID", "Username", "Display Name", "Email Address", "Role", "Status", "Actions"].map((h, i) => (
                  <th key={i} style={{
                    padding: "12px 20px",
                    textAlign: i >= 4 ? "center" : "left",
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
              {users.map((u, idx) => {
                const isSelf = Number(u.id) === Number(currentUserId);
                return (
                  <tr
                    key={u.id}
                    style={{
                      borderBottom: `1px solid ${isLight ? "#e2e8f0" : "rgba(255,255,255,0.04)"}`,
                      background: idx % 2 === 0
                        ? "transparent"
                        : (isLight ? "#f8fafc" : "rgba(255,255,255,0.01)"),
                    }}
                  >
                    <td style={{ padding: "14px 20px", width: 90 }}>
                      <span style={{
                        fontFamily: "'DM Mono', monospace", fontSize: 11,
                        color: isLight ? "#059669" : "#475569",
                        letterSpacing: "0.02em", fontWeight: 600,
                      }}>
                        #{u.id}
                      </span>
                    </td>
                    <td style={{ padding: "14px 20px" }}>
                      <span style={{
                        fontSize: 13, fontWeight: 600,
                        color: isLight ? "#0f172a" : "#e2e8f0",
                        display: "block", lineHeight: 1.4,
                      }}>
                        @{u.username}
                      </span>
                    </td>
                    <td style={{ padding: "14px 20px" }}>
                      <span style={{ fontSize: 12, color: t.textSecondary, fontWeight: 500 }}>
                        {u.display_name}
                      </span>
                    </td>
                    <td style={{ padding: "14px 20px" }}>
                      <span style={{ fontSize: 11, color: t.textMuted, fontFamily: "'DM Mono', monospace" }}>
                        {u.email}
                      </span>
                    </td>
                    <td style={{ padding: "14px 20px", width: 140, textAlign: "center" }}>
                      <select
                        value={u.role}
                        disabled={actionLoading === u.id || isSelf}
                        onChange={(e) => handleUpdateUserRole(u.id, e.target.value)}
                        className="tk-select"
                        style={{
                          background: t.inputBg,
                          border: `1px solid ${t.inputBorder}`,
                          borderRadius: 7, padding: "5px 10px",
                          fontSize: 12, color: t.textPrimary,
                          fontFamily: "inherit", cursor: "pointer",
                        }}
                      >
                        <option value="user">User</option>
                        <option value="contributor">Contributor</option>
                        <option value="admin">Admin</option>
                      </select>
                    </td>
                    <td style={{ padding: "14px 20px", width: 130, textAlign: "center" }}>
                      <StatusBadge status={u.status} />
                    </td>
                    <td style={{ padding: "14px 20px", width: 150, textAlign: "center" }}>
                      <button
                        type="button"
                        disabled={actionLoading === u.id || isSelf}
                        onClick={() => handleToggleUserStatus(u.id, u.status)}
                        style={{
                          background: u.status === "active"
                            ? (isLight ? "#e11d48" : "transparent")
                            : (isLight ? "#059669" : "transparent"),
                          border: isLight
                            ? "none"
                            : `1px solid ${u.status === "active" ? "rgba(244,63,94,0.3)" : "rgba(16,185,129,0.3)"}`,
                          color: isLight
                            ? "#fff"
                            : (u.status === "active" ? "#fb7185" : "#34d399"),
                          padding: "5px 14px", borderRadius: 5,
                          fontSize: 11, fontWeight: 700, letterSpacing: "0.06em",
                          cursor: "pointer", transition: "all 0.15s",
                          textTransform: "uppercase",
                          opacity: (actionLoading === u.id || isSelf) ? 0.6 : 1,
                        }}
                      >
                        {u.status === "active" ? "DISABLE" : "ACTIVE"}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}