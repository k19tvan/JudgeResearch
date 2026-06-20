import { API_BASE_URL } from "../../api";
import React, { useEffect, useState } from "react";

function Avatar({ name = "?", url = null }) {
  if (url) {
    const fullUrl = url.startsWith("/") ? `${API_BASE_URL}${url}` : url;
    return (
      <img
        src={fullUrl}
        alt={name}
        style={{ width: 28, height: 28, borderRadius: "50%", objectFit: "cover" }}
      />
    );
  }
  const initials = name.split(" ").map(w => w[0]).slice(0, 2).join("").toUpperCase();
  const palette = ["#0891b2", "#059669", "#7c3aed", "#db2777", "#d97706", "#2563eb"];
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

export default function SubmissionsTab({ isLight = false }) {
  const [submissions, setSubmissions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [expandedSubId, setExpandedSubId] = useState(null);

  // Bộ lọc tìm kiếm
  const [filterUsername, setFilterUsername] = useState("");
  const [filterProblem, setFilterProblem] = useState("");
  const [filterStatus, setFilterStatus] = useState("all");

  const userId = localStorage.getItem("user_id");
  const role = localStorage.getItem("user_role") || "user";
  const isAdmin = role === "admin";

  const loadSubmissions = async () => {
    setLoading(true);
    setError("");
    try {
      // SỬA: Nếu là Admin thì gọi API tổng (không truyền user_id của chính mình), ngược lại mới truyền lọc theo user_id cá nhân
      const url = isAdmin 
        ? "http://localhost:21081/api/submissions" 
        : `http://localhost:21081/api/submissions?user_id=${userId}`;

      const response = await fetch(url, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("access_token")}`
        }
      });
      const result = await response.json();
      if (response.ok && result.status === "success") {
        setSubmissions(result.data || []);
      } else {
        setError(result.detail || "Failed to load submissions.");
      }
    } catch (err) {
      setError("Giao tiếp máy chủ thất bại: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSubmissions();
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

  const getStatusBadgeStyle = (status) => {
    const lowerStatus = status.toLowerCase();
    if (lowerStatus === "accepted") {
      return {
        background: isLight ? "#ecfdf5" : "rgba(16,185,129,0.08)",
        color: isLight ? "#065f46" : "#34d399",
        border: `1px solid ${isLight ? "#6ee7b7" : "rgba(16,185,129,0.2)"}`,
      };
    }
    if (lowerStatus === "wrong_answer" || lowerStatus === "wrong answer") {
      return {
        background: isLight ? "#fff1f2" : "rgba(244,63,94,0.08)",
        color: isLight ? "#9f1239" : "#fb7185",
        border: `1px solid ${isLight ? "#fecdd3" : "rgba(244,63,94,0.2)"}`,
      };
    }
    return {
      background: isLight ? "#fffbeb" : "rgba(245,158,11,0.08)",
      color: isLight ? "#d97706" : "#fbbf24",
      border: `1px solid ${isLight ? "#fde68a" : "rgba(245,158,11,0.2)"}`,
    };
  };

  // Logic lọc dữ liệu trên Frontend
  const filteredSubmissions = submissions.filter((sub) => {
    const matchUser = isAdmin
      ? (sub.username || "").toLowerCase().includes(filterUsername.toLowerCase()) ||
        (sub.display_name || "").toLowerCase().includes(filterUsername.toLowerCase())
      : true;

    const matchProblem = (sub.problem_name || "").toLowerCase().includes(filterProblem.toLowerCase());

    const matchStatus = filterStatus === "all"
      ? true
      : (sub.status || "").toLowerCase() === filterStatus.toLowerCase();

    return matchUser && matchProblem && matchStatus;
  });

  const inputStyle = {
    background: t.inputBg,
    border: `1px solid ${t.inputBorder}`,
    borderRadius: 7,
    padding: "8px 12px",
    fontSize: 13,
    color: t.textPrimary,
    outline: "none",
  };

  return (
    <div style={{ fontFamily: "'Inter var', 'Inter', sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');
        .filter-input:focus {
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
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
              </svg>
            </div>
            <h2 style={{
              margin: 0, fontSize: 18, fontWeight: 700,
              color: t.textPrimary, letterSpacing: "-0.02em",
            }}>
              {isAdmin ? "System Submissions Queue" : "My Submissions History"}
            </h2>
          </div>
          <p style={{ margin: 0, fontSize: 12, color: t.textSecondary, lineHeight: 1.5, paddingLeft: 40 }}>
            {isAdmin
              ? "Giám sát toàn bộ hoạt động nộp bài thực hành của tất cả các tài khoản trên hệ thống."
              : "Theo dõi lại lịch sử điểm số và chi tiết các lần nộp bài giải thuật trước đây của bạn."}
          </p>
        </div>
      </div>

      {/* ── Filter Row ── */}
      <div style={{
        background: t.surface,
        border: `1px solid ${t.border}`,
        borderRadius: 12, padding: "16px 20px",
        marginBottom: 20, display: "flex", gap: 14, flexWrap: "wrap",
        boxShadow: t.shadow,
      }}>
        {isAdmin && (
          <div style={{ display: "flex", flexDirection: "column", gap: 6, flex: 1, minWidth: 150 }}>
            <span style={{ fontSize: 10, fontWeight: 700, textTransform: "uppercase", color: t.textSecondary }}>Filter User</span>
            <input
              type="text"
              placeholder="Username or display name..."
              value={filterUsername}
              onChange={(e) => setFilterUsername(e.target.value)}
              className="filter-input"
              style={inputStyle}
            />
          </div>
        )}

        <div style={{ display: "flex", flexDirection: "column", gap: 6, flex: 2, minWidth: 200 }}>
          <span style={{ fontSize: 10, fontWeight: 700, textTransform: "uppercase", color: t.textSecondary }}>Filter Problem</span>
          <input
            type="text"
            placeholder="Search problem name..."
            value={filterProblem}
            onChange={(e) => setFilterProblem(e.target.value)}
            className="filter-input"
            style={inputStyle}
          />
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 6, width: 180 }}>
          <span style={{ fontSize: 10, fontWeight: 700, textTransform: "uppercase", color: t.textSecondary }}>Filter Status</span>
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="filter-input"
            style={{ ...inputStyle, cursor: "pointer" }}
          >
            <option value="all">All Statuses</option>
            <option value="accepted">Accepted</option>
            <option value="wrong_answer">Wrong Answer</option>
            <option value="runtime_error">Runtime Error</option>
            <option value="time_limit_exceeded">Time Limit Exceeded</option>
          </select>
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

      {/* ── Submissions Queue List ── */}
      {loading ? (
        <div style={{
          background: t.surface, border: `1px solid ${t.border}`, borderRadius: 12, padding: "48px 24px", textAlign: "center", color: t.textSecondary, fontSize: 13,
        }}>
          <p className="animate-pulse">Loading execution queue...</p>
        </div>
      ) : filteredSubmissions.length === 0 ? (
        <div style={{
          background: t.surface, border: `1px solid ${t.border}`, borderRadius: 12, padding: "48px 24px", textAlign: "center", color: t.textSecondary, fontSize: 13,
          boxShadow: t.shadow,
        }}>
          <div style={{ marginBottom: 8, opacity: 0.35, fontSize: 28 }}>📭</div>
          {isAdmin ? "Không tìm thấy lượt nộp bài nào khớp với bộ lọc." : "You have no submissions yet."}
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {filteredSubmissions.map((sub) => {
            const isExpanded = expandedSubId === sub.id;
            const badgeStyle = getStatusBadgeStyle(sub.status);
            
            return (
              <div
                key={sub.id}
                style={{
                  background: t.surface,
                  border: `1px solid ${isExpanded ? t.accentBorder : t.border}`,
                  borderRadius: 12, overflow: "hidden",
                  boxShadow: t.shadow,
                  transition: "all 0.15s",
                }}
              >
                {/* Header Row */}
                <div
                  onClick={() => setExpandedSubId(isExpanded ? null : sub.id)}
                  style={{
                    padding: "16px 20px", display: "flex", alignItems: "center", justifyContent: "space-between",
                    cursor: "pointer", userSelect: "none"
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 16, flex: 1, minWidth: 0 }}>
                    {/* ID */}
                    <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 11, color: t.textMuted, width: 70 }}>
                      #SUB_{String(sub.id).padStart(4, "0")}
                    </span>

                    {/* Problem Name */}
                    <div style={{ minWidth: 0, flex: 2 }}>
                      <span style={{ fontSize: 13, fontWeight: 700, color: t.textPrimary, block: "block", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {sub.problem_name}
                      </span>
                    </div>

                    {/* User profile (only for Admin) */}
                    {isAdmin && (
                      <div style={{ display: "flex", alignItems: "center", gap: 8, width: 180, flexShrink: 0 }}>
                        <Avatar name={sub.display_name || sub.username} url={sub.avatar_url} />
                        <span style={{ fontSize: 12, color: t.textSecondary, fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {sub.display_name || `@${sub.username}`}
                        </span>
                      </div>
                    )}

                    {/* Status badge */}
                    <div style={{ width: 140, textAlign: "center", flexShrink: 0 }}>
                      <span style={{
                        display: "inline-flex", padding: "3px 8px", borderRadius: 5, fontSize: 9, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.04em",
                        ...badgeStyle
                      }}>
                        {sub.status.replace("_", " ")}
                      </span>
                    </div>

                    {/* Score Bar */}
                    <div style={{ width: 110, display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
                      <span style={{ fontSize: 12, fontWeight: 700, color: t.textPrimary, fontFamily: "'DM Mono', monospace" }}>
                        {sub.score} pts
                      </span>
                      <div style={{ flex: 1, height: 4, borderRadius: 2, background: isLight ? "#e2e8f0" : "rgba(255,255,255,0.06)", overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${sub.score}%`, background: sub.score === 100 ? "#10b981" : "#f59e0b" }} />
                      </div>
                    </div>
                  </div>

                  {/* Right side Metadata */}
                  <div style={{ display: "flex", alignItems: "center", gap: 14, marginLeft: 16 }}>
                    <span style={{ fontSize: 11, color: t.textMuted, fontFamily: "'DM Mono', monospace" }}>
                      {new Date(sub.created_at).toLocaleDateString("en-GB", {
                        day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit"
                      })}
                    </span>
                    <span style={{ fontSize: 12, color: t.textSecondary }}>
                      {isExpanded ? "▲" : "▼"}
                    </span>
                  </div>
                </div>

                {/* Expanded Details Pane */}
                {isExpanded && (
                  <div style={{
                    borderTop: `1px solid ${t.border}`,
                    background: isLight ? "#fafafa" : "rgba(0,0,0,0.15)",
                    padding: 20,
                  }}>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 280px", gap: 20 }}>
                      {/* Code block preview */}
                      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                        <span style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", color: t.textSecondary }}>Submitted Code</span>
                        <pre style={{
                          margin: 0, padding: 14, borderRadius: 8,
                          background: isLight ? "#f1f5f9" : "#050b14",
                          border: `1px solid ${t.border}`,
                          color: isLight ? "#1e293b" : "#e2e8f0",
                          fontFamily: "'DM Mono', monospace", fontSize: 12,
                          lineHeight: 1.6, overflowX: "auto", maxHeight: 280,
                        }}>
                          <code>{sub.submitted_code}</code>
                        </pre>
                      </div>

                      {/* Execution Details & Testcases */}
                      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                        <span style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", color: t.textSecondary }}>Testcases Details</span>
                        
                        {sub.test_results && sub.test_results.length > 0 ? (
                          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                            {sub.test_results.map((result, idx) => (
                              <div
                                key={idx}
                                style={{
                                  display: "flex", alignItems: "center", justifyContent: "space-between",
                                  padding: "8px 12px", borderRadius: 6,
                                  background: isLight ? "#ffffff" : "rgba(255,255,255,0.02)",
                                  border: `1px solid ${t.border}`,
                                }}
                              >
                                <span style={{ fontSize: 11, fontWeight: 600, color: t.textPrimary }}>
                                  {result.testcase}
                                </span>
                                <span style={{
                                  fontSize: 10, fontWeight: 700,
                                  color: result.status === "Accepted" ? "#10b981" : "#ef4444"
                                }}>
                                  {result.status}
                                </span>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div style={{
                            padding: "16px", borderRadius: 6, textAlign: "center",
                            background: isLight ? "#ffffff" : "rgba(255,255,255,0.02)",
                            border: `1px dashed ${t.border}`, fontSize: 12, color: t.textMuted
                          }}>
                            No individual testcase results recorded.
                          </div>
                        )}

                        {/* Additional Info Box */}
                        <div style={{
                          marginTop: "auto", padding: "10px 14px", borderRadius: 6,
                          background: t.accentBg, border: `1px solid ${t.accentBorder}`,
                          fontSize: 11, color: isLight ? t.accentDark : t.accent, lineHeight: 1.5
                        }}>
                          🚀 <strong>Sandbox Feedback:</strong> Evaluation finished. Code compiled and run within secure container isolated boundary limits.
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
