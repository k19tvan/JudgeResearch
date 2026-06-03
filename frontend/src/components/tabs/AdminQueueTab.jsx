// src/components/tabs/AdminQueueTab.jsx
import React, { useEffect, useState } from "react";

export default function AdminQueueTab({ isLight = false }) {
  const [activeQueueTab, setActiveQueueTab] = useState("problems"); // 'problems', 'solutions', 'roadmaps'
  const [requests, setRequests] = useState([]);
  const [solutionProposals, setSolutionProposals] = useState([]);
  const [roadmaps, setRoadmaps] = useState([]);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(null);
  const [expandedProposalId, setExpandedProposalId] = useState(null);

  const adminId = localStorage.getItem("user_id");

  // Load danh sách bài tập công khai chờ duyệt
  const loadPendingRequests = async (showLoading = true) => {
    if (showLoading) setLoading(true);
    try {
      const response = await fetch(`http://localhost:21081/api/problems/pending-requests?admin_id=${adminId}`);
      const result = await response.json();
      setRequests(result?.data || []);
    } catch (err) {
      console.error(err);
    } finally {
      if (showLoading) setLoading(false);
    }
  };

  // Load danh sách đề xuất lời giải từ Contributor chờ kiểm duyệt
  const loadSolutionProposals = async (showLoading = true) => {
    if (showLoading) setLoading(true);
    try {
      const response = await fetch(`http://localhost:21081/api/admin/solution-proposals?admin_id=${adminId}`);
      const result = await response.json();
      setSolutionProposals(result?.data || []);
    } catch (err) {
      console.error(err);
    } finally {
      if (showLoading) setLoading(false);
    }
  };

  // Load danh sách Roadmap gửi duyệt (status = 'pending')
  const loadPendingRoadmaps = async (showLoading = true) => {
    if (showLoading) setLoading(true);
    try {
      const response = await fetch(`http://localhost:21081/api/roadmaps`, {
        headers: {
          "Authorization": `Bearer ${localStorage.getItem("access_token")}`
        }
      });
      const result = await response.json();
      // Lọc local lấy ra những Roadmap có trạng thái chờ duyệt 'pending'
      const pendingRoadmaps = (result?.data || []).filter(r => r.status === "pending");
      setRoadmaps(pendingRoadmaps);
    } catch (err) {
      console.error(err);
    } finally {
      if (showLoading) setLoading(false);
    }
  };

  // Tải đồng bộ số lượng hàng đợi của cả 3 tab khi mount lần đầu tiên
  useEffect(() => {
    const loadAllQueues = async () => {
      setLoading(true);
      await Promise.all([
        loadPendingRequests(false),
        loadSolutionProposals(false),
        loadPendingRoadmaps(false)
      ]);
      setLoading(false);
    };
    loadAllQueues();
  }, []);

  // Thực hiện tải mới khi chuyển tab thủ công
  useEffect(() => {
    if (activeQueueTab === "problems") {
      loadPendingRequests(true);
    } else if (activeQueueTab === "solutions") {
      loadSolutionProposals(true);
    } else if (activeQueueTab === "roadmaps") {
      loadPendingRoadmaps(true);
    }
  }, [activeQueueTab]);

  // Hành động kiểm duyệt bài tập công khai
  const handleProblemAction = async (problemId, action) => {
    setActionLoading(problemId);
    try {
      const response = await fetch(`http://localhost:21081/api/problems/${problemId}/${action}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ admin_id: Number(adminId) }),
      });
      if (response.ok) {
        await loadPendingRequests(true);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setActionLoading(null);
    }
  };

  // Hành động kiểm duyệt đề xuất lời giải
  const handleProposalAction = async (proposalId, action) => {
    setActionLoading(proposalId);
    try {
      const response = await fetch(`http://localhost:21081/api/admin/solution-proposals/${proposalId}/action`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ admin_id: Number(adminId), action }),
      });
      if (response.ok) {
        alert(action === "approve" ? "Đã duyệt giải pháp này thành công!" : "Đã bác bỏ đề xuất thành công!");
        await loadSolutionProposals(true);
      } else {
        const res = await response.json();
        alert(res.detail || "Lỗi xử lý đề xuất");
      }
    } catch (err) {
      console.error(err);
    } finally {
      setActionLoading(null);
    }
  };

  // Hành động kiểm duyệt Roadmap công khai
  const handleRoadmapAction = async (roadmapId, action) => {
    setActionLoading(roadmapId);
    try {
      const response = await fetch(`http://localhost:21081/api/roadmaps/${roadmapId}/${action}`, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Authorization": `Bearer ${localStorage.getItem("access_token")}`
        }
      });
      if (response.ok) {
        alert(action === "approve" ? "Đã duyệt lộ trình thành công!" : "Đã bác bỏ yêu cầu duyệt lộ trình!");
        await loadPendingRoadmaps(true);
      } else {
        const res = await response.json();
        alert(res.detail || "Lỗi xử lý lộ trình");
      }
    } catch (err) {
      console.error(err);
    } finally {
      setActionLoading(null);
    }
  };

  const t = isLight ? {
    pageBg: "#f1f5f9",
    surface: "#ffffff",
    surfaceRaised: "#f8fafc",
    border: "#e2e8f0",
    borderStrong: "#cbd5e1",
    accent: "#059669",
    accentDark: "#047857",
    accentBg: "#ecfdf5",
    accentBorder: "#6ee7b7",
    tableHead: "#065f46",
    tableHeadBg: "#059669",
    textPrimary: "#0f172a",
    textSecondary: "#475569",
    textMuted: "#94a3b8",
    shadow: "0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04)",
  } : {
    pageBg: "transparent",
    surface: "#0f172a",
    surfaceRaised: "#111827",
    border: "rgba(255,255,255,0.07)",
    borderStrong: "rgba(255,255,255,0.12)",
    accent: "#06b6d4",
    accentDark: "#0891b2",
    accentBg: "rgba(6,182,212,0.08)",
    accentBorder: "rgba(6,182,212,0.3)",
    tableHead: "#e2e8f0",
    tableHeadBg: "rgba(255,255,255,0.03)",
    textPrimary: "#f1f5f9",
    textSecondary: "#64748b",
    textMuted: "#475569",
    shadow: "none",
  };

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
                <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 002 2h2a2 2 0 002-2" />
              </svg>
            </div>
            <h2 style={{
              margin: 0, fontSize: 18, fontWeight: 700,
              color: t.textPrimary, letterSpacing: "-0.02em",
            }}>
              Admin Approval Queue
            </h2>
          </div>
          <p style={{ margin: 0, fontSize: 12, color: t.textSecondary, lineHeight: 1.5, paddingLeft: 40 }}>
            Approve or reject requests from the community and contributors.
          </p>
        </div>
      </div>

      {/* ── Sub-navigation Tabs (Điều hướng song song 3 Tab) ── */}
      <div style={{ display: "flex", gap: 8, marginBottom: 20 }}>
        <button
          type="button"
          onClick={() => { setActiveQueueTab("problems"); setExpandedProposalId(null); }}
          style={{
            padding: "8px 16px", borderRadius: 8, fontSize: 12, fontWeight: 600, cursor: "pointer",
            background: activeQueueTab === "problems" ? t.accent : (isLight ? "#e2e8f0" : "rgba(255,255,255,0.04)"),
            color: activeQueueTab === "problems" ? "#fff" : t.textSecondary,
            border: "none", transition: "all 0.15s"
          }}
        >
          Problem Requests ({requests.length})
        </button>
        <button
          type="button"
          onClick={() => { setActiveQueueTab("solutions"); setExpandedProposalId(null); }}
          style={{
            padding: "8px 16px", borderRadius: 8, fontSize: 12, fontWeight: 600, cursor: "pointer",
            background: activeQueueTab === "solutions" ? t.accent : (isLight ? "#e2e8f0" : "rgba(255,255,255,0.04)"),
            color: activeQueueTab === "solutions" ? "#fff" : t.textSecondary,
            border: "none", transition: "all 0.15s"
          }}
        >
          Solution Proposals ({solutionProposals.length})
        </button>
        <button
          type="button"
          onClick={() => { setActiveQueueTab("roadmaps"); setExpandedProposalId(null); }}
          style={{
            padding: "8px 16px", borderRadius: 8, fontSize: 12, fontWeight: 600, cursor: "pointer",
            background: activeQueueTab === "roadmaps" ? t.accent : (isLight ? "#e2e8f0" : "rgba(255,255,255,0.04)"),
            color: activeQueueTab === "roadmaps" ? "#fff" : t.textSecondary,
            border: "none", transition: "all 0.15s"
          }}
        >
          Roadmap Requests ({roadmaps.length})
        </button>
      </div>

      {/* ── KHỐI CHUNG: LOADING GENERAL SPINNER ── */}
      {loading && (
        <div style={{
          background: t.surface, border: `1px solid ${t.border}`, borderRadius: 12, padding: "48px 24px", textAlign: "center", color: t.textSecondary, fontSize: 13,
        }}>
          <p className="animate-pulse">Loading requests queue...</p>
        </div>
      )}

      {/* ── TAB 1: HIỂN THỊ HÀNG CHỜ YÊU CẦU CÔNG KHAI BÀI TẬP ── */}
      {!loading && activeQueueTab === "problems" && (
        requests.length === 0 ? (
          <div style={{
            background: t.surface, border: `1px solid ${t.border}`, borderRadius: 12, padding: "48px 24px", textAlign: "center", color: t.textSecondary, fontSize: 13,
            boxShadow: t.shadow,
          }}>
            <div style={{ marginBottom: 8, opacity: 0.35, fontSize: 28 }}>✅</div>
            All caught up! No pending problem requests.
          </div>
        ) : (
          <div style={{
            background: t.surface,
            border: `1px solid ${t.border}`,
            borderRadius: 12, overflow: "hidden",
            boxShadow: t.shadow,
          }}>
            <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 700 }}>
              <thead>
                <tr style={{
                  background: isLight ? t.tableHeadBg : t.tableHeadBg,
                  borderBottom: `1px solid ${isLight ? t.accentDark + "55" : t.border}`,
                }}>
                  {["ID", "Problem Name", "Author", "Requested Date", "Actions"].map((h, i) => (
                    <th key={i} style={{
                      padding: "12px 20px",
                      textAlign: i === 4 ? "center" : "left",
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
                    <td style={{ padding: "14px 20px", width: 120 }}>
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
                    <td style={{ padding: "14px 20px", width: 160 }}>
                      <span style={{ fontSize: 12, color: t.textSecondary, fontWeight: 500 }}>
                        {req.author_name}
                      </span>
                    </td>
                    <td style={{ padding: "14px 20px", width: 160 }}>
                      <span style={{ fontSize: 11, color: isLight ? "#64748b" : "#475569", fontFamily: "'DM Mono', monospace" }}>
                        {new Date(req.created_at).toLocaleDateString("en-GB", {
                          day: "2-digit", month: "short", year: "numeric"
                        })}
                      </span>
                    </td>
                    <td style={{ padding: "14px 20px", width: 220, textAlign: "center" }}>
                      <div style={{ display: "flex", justifyContent: "center", gap: 8 }}>
                        <button
                          type="button"
                          disabled={actionLoading === req.id}
                          onClick={() => handleProblemAction(req.id, "approve")}
                          style={{
                            background: isLight ? "#059669" : "transparent",
                            border: isLight ? "none" : "1px solid rgba(16,185,129,0.3)",
                            color: isLight ? "#fff" : "#34d399",
                            padding: "5px 12px", borderRadius: 5,
                            fontSize: 11, fontWeight: 700, letterSpacing: "0.06em",
                            cursor: "pointer", transition: "all 0.15s",
                            textTransform: "uppercase",
                            opacity: actionLoading === req.id ? 0.6 : 1,
                          }}
                        >
                          APPROVE
                        </button>
                        <button
                          type="button"
                          disabled={actionLoading === req.id}
                          onClick={() => handleProblemAction(req.id, "reject")}
                          style={{
                            background: isLight ? "#e11d48" : "transparent",
                            border: isLight ? "none" : "1px solid rgba(244,63,94,0.3)",
                            color: isLight ? "#fff" : "#fb7185",
                            padding: "5px 12px", borderRadius: 5,
                            fontSize: 11, fontWeight: 700, letterSpacing: "0.06em",
                            cursor: "pointer", transition: "all 0.15s",
                            textTransform: "uppercase",
                            opacity: actionLoading === req.id ? 0.6 : 1,
                          }}
                        >
                          REJECT
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )
      )}

      {/* ── TAB 2: HIỂN THỊ HÀNG CHỜ YÊU CẦU DUYỆT ĐỀ XUẤT LỜI GIẢI MẪU ── */}
      {!loading && activeQueueTab === "solutions" && (
        solutionProposals.length === 0 ? (
          <div style={{
            background: t.surface, border: `1px solid ${t.border}`, borderRadius: 12, padding: "48px 24px", textAlign: "center", color: t.textSecondary, fontSize: 13,
            boxShadow: t.shadow,
          }}>
            <div style={{ marginBottom: 8, opacity: 0.35, fontSize: 28 }}>✅</div>
            All caught up! No pending solution proposals.
          </div>
        ) : (
          <div style={{
            background: t.surface,
            border: `1px solid ${t.border}`,
            borderRadius: 12, overflow: "hidden",
            boxShadow: t.shadow,
          }}>
            <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 700 }}>
              <thead>
                <tr style={{
                  background: isLight ? t.tableHeadBg : t.tableHeadBg,
                  borderBottom: `1px solid ${isLight ? t.accentDark + "55" : t.border}`,
                }}>
                  {["ID", "Problem Name", "Contributor", "Date Proposed", "Proposed Code", "Actions"].map((h, i) => (
                    <th key={i} style={{
                      padding: "12px 20px",
                      textAlign: i === 5 ? "center" : "left",
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
                {solutionProposals.map((prop, idx) => {
                  const isCodeExpanded = expandedProposalId === prop.id;
                  return (
                    <React.Fragment key={prop.id}>
                      <tr
                        style={{
                          borderBottom: `1px solid ${isLight ? "#e2e8f0" : "rgba(255,255,255,0.04)"}`,
                          background: idx % 2 === 0
                            ? "transparent"
                            : (isLight ? "#f8fafc" : "rgba(255,255,255,0.01)"),
                        }}
                      >
                        <td style={{ padding: "14px 20px", width: 110 }}>
                          <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 11, color: t.accent, fontWeight: 600 }}>
                            #PROP_{prop.id}
                          </span>
                        </td>
                        <td style={{ padding: "14px 20px" }}>
                          <span style={{ fontSize: 13, fontWeight: 600, color: t.textPrimary }}>
                            {prop.problem_name}
                          </span>
                        </td>
                        <td style={{ padding: "14px 20px", width: 160 }}>
                          <span style={{ fontSize: 12, color: t.textSecondary, fontWeight: 500 }}>
                            {prop.contributor_name}
                          </span>
                        </td>
                        <td style={{ padding: "14px 20px", width: 160 }}>
                          <span style={{ fontSize: 11, color: isLight ? "#64748b" : "#475569", fontFamily: "'DM Mono', monospace" }}>
                            {new Date(prop.created_at).toLocaleDateString()}
                          </span>
                        </td>
                        <td style={{ padding: "14px 20px", width: 140 }}>
                          <button
                            type="button"
                            onClick={() => setExpandedProposalId(isCodeExpanded ? null : prop.id)}
                            style={{
                              background: "transparent", border: "none", color: t.accent,
                              fontSize: 11, fontWeight: 600, cursor: "pointer"
                            }}
                          >
                            {isCodeExpanded ? "Hide Code ▲" : "View Code ▼"}
                          </button>
                        </td>
                        <td style={{ padding: "14px 20px", width: 220, textAlign: "center" }}>
                          <div style={{ display: "flex", justifyContent: "center", gap: 8 }}>
                            <button
                              type="button"
                              disabled={actionLoading === prop.id}
                              onClick={() => handleProposalAction(prop.id, "approve")}
                              style={{
                                background: isLight ? "#059669" : "transparent",
                                border: isLight ? "none" : "1px solid rgba(16,185,129,0.3)",
                                color: isLight ? "#fff" : "#34d399",
                                padding: "5px 12px", borderRadius: 5,
                                fontSize: 11, fontWeight: 700, letterSpacing: "0.06em",
                                cursor: "pointer", transition: "all 0.15s",
                                textTransform: "uppercase",
                                opacity: actionLoading === prop.id ? 0.6 : 1,
                              }}
                            >
                              Approve
                            </button>
                            <button
                              type="button"
                              disabled={actionLoading === prop.id}
                              onClick={() => handleProposalAction(prop.id, "reject")}
                              style={{
                                background: isLight ? "#e11d48" : "transparent",
                                border: isLight ? "none" : "1px solid rgba(244,63,94,0.3)",
                                color: isLight ? "#fff" : "#fb7185",
                                padding: "5px 12px", borderRadius: 5,
                                fontSize: 11, fontWeight: 700, letterSpacing: "0.06em",
                                cursor: "pointer", transition: "all 0.15s",
                                textTransform: "uppercase",
                                opacity: actionLoading === prop.id ? 0.6 : 1,
                              }}
                            >
                              Reject
                            </button>
                          </div>
                        </td>
                      </tr>
                      {isCodeExpanded && (
                        <tr>
                          <td colSpan={6} style={{ padding: "16px 20px", background: isLight ? "#f8fafc" : "rgba(0,0,0,0.2)" }}>
                            <pre style={{
                              margin: 0, padding: 12, borderRadius: 6,
                              background: isLight ? "#f1f5f9" : "#050b14",
                              border: `1px solid ${t.border}`,
                              fontFamily: "'Fira Code', 'DM Mono', monospace", fontSize: 12,
                              lineHeight: 1.5, overflowX: "auto"
                            }}>
                              <code>{prop.proposed_code}</code>
                            </pre>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        )
      )}

      {/* ── TAB 3: HIỂN THỊ HÀNG CHỜ YÊU CẦU DUYỆT ROADMAP ── */}
      {!loading && activeQueueTab === "roadmaps" && (
        roadmaps.length === 0 ? (
          <div style={{
            background: t.surface, border: `1px solid ${t.border}`, borderRadius: 12, padding: "48px 24px", textAlign: "center", color: t.textSecondary, fontSize: 13,
            boxShadow: t.shadow,
          }}>
            <div style={{ marginBottom: 8, opacity: 0.35, fontSize: 28 }}>✅</div>
            All caught up! No pending roadmap requests.
          </div>
        ) : (
          <div style={{
            background: t.surface,
            border: `1px solid ${t.border}`,
            borderRadius: 12, overflow: "hidden",
            boxShadow: t.shadow,
          }}>
            <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 700 }}>
              <thead>
                <tr style={{
                  background: isLight ? t.tableHeadBg : t.tableHeadBg,
                  borderBottom: `1px solid ${isLight ? t.accentDark + "55" : t.border}`,
                }}>
                  {["ID", "Roadmap Name", "Repository URL", "Level", "Steps", "Actions"].map((h, i) => (
                    <th key={i} style={{
                      padding: "12px 20px",
                      textAlign: i === 5 ? "center" : "left",
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
                {roadmaps.map((road, idx) => (
                  <tr
                    key={road.id}
                    style={{
                      borderBottom: `1px solid ${isLight ? "#e2e8f0" : "rgba(255,255,255,0.04)"}`,
                      background: idx % 2 === 0
                        ? "transparent"
                        : (isLight ? "#f8fafc" : "rgba(255,255,255,0.01)"),
                    }}
                  >
                    <td style={{ padding: "14px 20px", width: 110 }}>
                      <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 11, color: t.accent, fontWeight: 600 }}>
                        #ROAD_{road.id}
                      </span>
                    </td>
                    <td style={{ padding: "14px 20px" }}>
                      <span style={{ fontSize: 13, fontWeight: 600, color: t.textPrimary }}>
                        {road.name}
                      </span>
                    </td>
                    <td style={{ padding: "14px 20px" }}>
                      <span style={{ fontSize: 12, color: t.textSecondary, fontFamily: "'DM Mono', monospace" }}>
                        {road.repository_url}
                      </span>
                    </td>
                    <td style={{ padding: "14px 20px", width: 120 }}>
                      <span style={{ fontSize: 11, color: t.textSecondary, textTransform: "capitalize" }}>
                        {road.level}
                      </span>
                    </td>
                    <td style={{ padding: "14px 20px", width: 100 }}>
                      <span style={{ fontSize: 11, color: t.textSecondary, fontFamily: "'DM Mono', monospace" }}>
                        {road.problem_count || 0} steps
                      </span>
                    </td>
                    <td style={{ padding: "14px 20px", width: 220, textAlign: "center" }}>
                      <div style={{ display: "flex", justifyContent: "center", gap: 8 }}>
                        <button
                          type="button"
                          disabled={actionLoading === road.id}
                          onClick={() => handleRoadmapAction(road.id, "approve")}
                          style={{
                            background: isLight ? "#059669" : "transparent",
                            border: isLight ? "none" : "1px solid rgba(16,185,129,0.3)",
                            color: isLight ? "#fff" : "#34d399",
                            padding: "5px 12px", borderRadius: 5,
                            fontSize: 11, fontWeight: 700, letterSpacing: "0.06em",
                            cursor: "pointer", transition: "all 0.15s",
                            textTransform: "uppercase",
                            opacity: actionLoading === road.id ? 0.6 : 1,
                          }}
                        >
                          Approve
                        </button>
                        <button
                          type="button"
                          disabled={actionLoading === road.id}
                          onClick={() => handleRoadmapAction(road.id, "reject")}
                          style={{
                            background: isLight ? "#e11d48" : "transparent",
                            border: isLight ? "none" : "1px solid rgba(244,63,94,0.3)",
                            color: isLight ? "#fff" : "#fb7185",
                            padding: "5px 12px", borderRadius: 5,
                            fontSize: 11, fontWeight: 700, letterSpacing: "0.06em",
                            cursor: "pointer", transition: "all 0.15s",
                            textTransform: "uppercase",
                            opacity: actionLoading === road.id ? 0.6 : 1,
                          }}
                        >
                          Reject
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )
      )}
    </div>
  );
}