// src/components/tabs/MyRequestsTab.jsx
import React, { useEffect, useState } from "react";

export default function MyRequestsTab({ isLight = false }) {
  const [requests, setRequests] = useState([]);
  const [loading, setLoading] = useState(true);
  const userId = localStorage.getItem("user_id");

  const loadMyRequests = async () => {
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:21081/api/problems/my-requests?user_id=${userId}`);
      const result = await response.json();
      setRequests(result?.data || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadMyRequests();
  }, []);

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
    shadow:       "0 4px 20px rgba(0,0,0,0.4)",
  };

  const statusColors = {
    PENDING: {
      dot: isLight ? "#2563eb" : "#3b82f6",
      bg: isLight ? "#eff6ff" : "rgba(59,130,246,0.08)",
      border: isLight ? "#93c5fd" : "rgba(59,130,246,0.2)",
      text: isLight ? "#1e40af" : "#60a5fa",
    },
    APPROVED: {
      dot: isLight ? "#059669" : "#10b981",
      bg: isLight ? "#ecfdf5" : "rgba(16,185,129,0.08)",
      border: isLight ? "#6ee7b7" : "rgba(16,185,129,0.2)",
      text: isLight ? "#065f46" : "#34d399",
    },
    REJECTED: {
      dot: isLight ? "#e11d48" : "#f43f5e",
      bg: isLight ? "#fff1f2" : "rgba(244,63,94,0.08)",
      border: isLight ? "#fecdd3" : "rgba(244,63,94,0.2)",
      text: isLight ? "#9f1239" : "#fb7185",
    },
  };

  function StatusBadge({ status = "PENDING" }) {
    const key = status.toUpperCase();
    const cfg = statusColors[key] || statusColors.PENDING;
    return (
      <span style={{
        display: "inline-flex", alignItems: "center", gap: 5,
        background: cfg.bg, border: `1px solid ${cfg.border}`,
        color: cfg.text, fontSize: 10, fontWeight: 700,
        letterSpacing: "0.08em", padding: "3px 9px",
        borderRadius: 4, textTransform: "uppercase",
      }}>
        <span style={{ width: 5, height: 5, borderRadius: "50%", background: cfg.dot, flexShrink: 0 }} />
        {status}
      </span>
    );
  }

  return (
    <div style={{ fontFamily: "'Inter var', 'Inter', sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');
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
                <path d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            </div>
            <h2 style={{
              margin: 0, fontSize: 18, fontWeight: 700,
              color: t.textPrimary, letterSpacing: "-0.02em",
            }}>
              My Public Requests
            </h2>
          </div>
          <p style={{ margin: 0, fontSize: 12, color: t.textSecondary, lineHeight: 1.5, paddingLeft: 40 }}>
            Track status of your private problems requested to be public.
          </p>
        </div>
      </div>

      {loading ? (
        <div style={{
          background: t.surface, border: `1px solid ${t.border}`, borderRadius: 12, padding: "48px 24px", textAlign: "center", color: t.textSecondary, fontSize: 13,
        }}>
          <p className="animate-pulse">Loading requests...</p>
        </div>
      ) : requests.length === 0 ? (
        <div style={{
          background: t.surface, border: `1px solid ${t.border}`, borderRadius: 12, padding: "48px 24px", textAlign: "center", color: t.textSecondary, fontSize: 13,
          boxShadow: t.shadow,
        }}>
          <div style={{ marginBottom: 8, opacity: 0.35, fontSize: 28 }}>📭</div>
          You haven't requested any problems to be public yet.
        </div>
      ) : (
        <div style={{
          background: t.surface,
          border: `1px solid ${t.border}`,
          borderRadius: 12, overflow: "hidden",
          boxShadow: t.shadow,
        }}>
          <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 600 }}>
            <thead>
              <tr style={{
                background: isLight ? t.tableHeadBg : t.tableHeadBg,
                borderBottom: `1px solid ${isLight ? t.accentDark + "55" : t.border}`,
              }}>
                {["Problem ID", "Problem Name", "Date Requested", "Status"].map((h, i) => (
                  <th key={i} style={{
                    padding: "12px 20px",
                    textAlign: i === 3 ? "center" : "left",
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
              {requests.map((req, idx) => (
                <tr
                  key={req.id}
                  style={{
                    borderBottom: `1px solid ${isLight ? "#e2e8f0" : "rgba(255,255,255,0.04)"}`,
                    background: idx % 2 === 0
                      ? "transparent"
                      : (isLight ? "#f8fafc" : "rgba(255,255,255,0.01)"),
                  }}
                >
                  <td style={{ padding: "14px 20px", width: 150 }}>
                    <span style={{
                      fontFamily: "'DM Mono', monospace", fontSize: 11,
                      color: isLight ? "#059669" : "#475569",
                      letterSpacing: "0.02em", fontWeight: 600,
                    }}>
                      #ml_{req.id}
                    </span>
                  </td>
                  <td style={{ padding: "14px 20px" }}>
                    <span style={{
                      fontSize: 13, fontWeight: 600,
                      color: isLight ? "#0f172a" : "#e2e8f0",
                      display: "block", lineHeight: 1.4,
                    }}>
                      {req.name}
                    </span>
                  </td>
                  <td style={{ padding: "14px 20px", width: 180 }}>
                    <span style={{ fontSize: 11, color: isLight ? "#64748b" : "#475569", fontFamily: "'DM Mono', monospace" }}>
                      {new Date(req.created_at).toLocaleDateString("en-GB", {
                        day: "2-digit", month: "short", year: "numeric"
                      })}
                    </span>
                  </td>
                  <td style={{ padding: "14px 20px", width: 140, textAlign: "center" }}>
                    <StatusBadge status={req.request_status} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}