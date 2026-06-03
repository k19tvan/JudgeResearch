// src/components/tabs/ProblemsTab.jsx
import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createPortal } from "react-dom";
import { createManualProblem, filterProblems } from "../../api";

export default function ProblemsTab({ isLight = false }) {
  const navigate = useNavigate();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [problems, setProblems] = useState([]);
  const [filterMode, setFilterMode] = useState("all");
  const [editingProblemId, setEditingProblemId] = useState(null);
  const [deletingProblem, setDeletingProblem] = useState(null);

  const [formData, setFormData] = useState({
    name: "",
    source: "",
    statement_markdown: "",
    theory_markdown: "",
    tutorial_markdown: "",
    solution_markdown: "",
    coding_markdown: "",
    checker_markdown: "",
  });
  const [inputZipFile, setInputZipFile] = useState(null);
  const [outputZipFile, setOutputZipFile] = useState(null);
  const [testcases, setTestcases] = useState([]);

  const role = localStorage.getItem("user_role") || "user";
  const currentUserId = Number(localStorage.getItem("user_id") || "1");

  const handleChange = (e) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setEditingProblemId(null);
    setTestcases([]);
    resetForm();
    setError("");
  };

  const handleAddTestcase = () => {
    setTestcases((prev) => [
      ...prev,
      { id: Date.now(), input: "{}", output: "{}" },
    ]);
  };

  const handleDeleteTestcase = (indexToDelete) => {
    setTestcases((prev) => prev.filter((_, idx) => idx !== indexToDelete));
  };

  const handleTestcaseChange = (index, field, value) => {
    setTestcases((prev) =>
      prev.map((tc, idx) => (idx === index ? { ...tc, [field]: value } : tc))
    );
  };

  const resetForm = () => {
    setFormData({
      name: "",
      source: "",
      statement_markdown: "",
      theory_markdown: "",
      tutorial_markdown: "",
      solution_markdown: "",
      coding_markdown: "",
      checker_markdown: "",
    });
    setInputZipFile(null);
    setOutputZipFile(null);
  };

  const loadProblems = async () => {
    setIsLoading(true);
    try {
      const userId = localStorage.getItem("user_id");
      const uid = userId ? Number(userId) : null;
      const result = await filterProblems(filterMode === "all" ? (uid ? "all" : "public") : filterMode, uid);
      setProblems(result?.data || []);
    } catch (err) {
      setError(err.message || "Fetch problems failed");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadProblems();
  }, [filterMode]);

  const handleEditProblemInit = async (e, problem) => {
    e.stopPropagation();
    setError("");
    try {
      const response = await fetch(`http://localhost:21081/api/problems/${problem.id}/content?user_id=${currentUserId}`);
      const result = await response.json();
      if (response.ok && result.data) {
        setFormData({
          name: result.data.name || "",
          source: problem.source || "",
          statement_markdown: result.data.statement_markdown || "",
          theory_markdown: result.data.theory_markdown || "",
          tutorial_markdown: result.data.tutorial_markdown || "",
          solution_markdown: result.data.solution_markdown || "",
          coding_markdown: result.data.coding_markdown || "",
          checker_markdown: result.data.checker_markdown || "",
        });
        setEditingProblemId(problem.id);
        
        try {
          const tcResponse = await fetch(`http://localhost:21081/api/problems/${problem.id}/testcases`);
          const tcResult = await tcResponse.json();
          if (tcResponse.ok && tcResult.data) {
            setTestcases(tcResult.data);
          } else {
            setTestcases([]);
          }
        } catch (tcErr) {
          console.error("Failed to load test cases:", tcErr);
          setTestcases([]);
        }

        setIsModalOpen(true);
      } else {
        alert(result.detail || "Failed to load problem content");
      }
    } catch (err) {
      alert("Error loading problem content: " + err.message);
    }
  };

  const handleDeleteProblemDirect = (e, problem) => {
    e.stopPropagation();
    setDeletingProblem(problem);
  };

  const confirmDeleteProblem = async () => {
    if (!deletingProblem) return;
    const problemId = deletingProblem.id;
    setDeletingProblem(null);
    try {
      const response = await fetch(`http://localhost:21081/api/problems/${problemId}`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: currentUserId })
      });
      if (response.ok) {
        alert("Problem deleted successfully!");
        await loadProblems();
      } else {
        const result = await response.json();
        alert(`Error: ${result.detail || "Delete failed"}`);
      }
    } catch (err) {
      alert("Failed to delete problem: " + err.message);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setIsSubmitting(true);

    try {
      const userId = localStorage.getItem("user_id");
      if (!userId) {
        throw new Error("Không tìm thấy thông tin đăng nhập. Vui lòng đăng nhập lại.");
      }

      if (editingProblemId) {
        const formPayload = new FormData();
        formPayload.append("user_id", currentUserId);
        formPayload.append("name", formData.name);
        formPayload.append("source", formData.source || "");
        formPayload.append("statement_markdown", formData.statement_markdown);
        formPayload.append("theory_markdown", formData.theory_markdown || "");
        formPayload.append("tutorial_markdown", formData.tutorial_markdown || "");
        formPayload.append("solution_markdown", formData.solution_markdown || "");
        formPayload.append("coding_markdown", formData.coding_markdown || "");
        formPayload.append("checker_markdown", formData.checker_markdown || "");

        if (inputZipFile) {
          formPayload.append("input_zip", inputZipFile);
        }
        if (outputZipFile) {
          formPayload.append("output_zip", outputZipFile);
        }

        // Only validate and append testcases if no zip file is uploaded
        if (!inputZipFile && !outputZipFile) {
          // Validate testcases JSON
          for (let i = 0; i < testcases.length; i++) {
            const tc = testcases[i];
            try {
              JSON.parse(tc.input);
            } catch (e) {
              throw new Error(`Test Case #${i + 1} Input is not valid JSON: ${e.message}`);
            }
            try {
              JSON.parse(tc.output);
            } catch (e) {
              throw new Error(`Test Case #${i + 1} Output is not valid JSON: ${e.message}`);
            }
          }
          formPayload.append("testcases", JSON.stringify(testcases));
        }

        const response = await fetch(`http://localhost:21081/api/problems/${editingProblemId}`, {
          method: "PUT",
          body: formPayload
        });
        const result = await response.json();
        if (!response.ok) {
          throw new Error(result.detail || "Edit problem failed");
        }
      } else {
        const formPayload = new FormData();
        formPayload.append("name", formData.name);
        formPayload.append("source", formData.source || "");
        formPayload.append("statement_markdown", formData.statement_markdown);
        formPayload.append("theory_markdown", formData.theory_markdown || "");
        formPayload.append("tutorial_markdown", formData.tutorial_markdown || "");
        formPayload.append("solution_markdown", formData.solution_markdown || "");
        formPayload.append("coding_markdown", formData.coding_markdown || "");
        formPayload.append("checker_markdown", formData.checker_markdown || "");
        formPayload.append("author_id", Number(userId));

        if (inputZipFile) {
          formPayload.append("input_zip", inputZipFile);
        }
        if (outputZipFile) {
          formPayload.append("output_zip", outputZipFile);
        }

        const result = await createManualProblem(formPayload);
        if (!result || !result.data) {
          throw new Error("Create problem failed");
        }
      }

      closeModal();
      await loadProblems();
    } catch (err) {
      setError(err.message || "Action failed");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleRequestPublic = async (e, problemId) => {
    e.stopPropagation();
    const confirmReq = window.confirm("Do you want to request the Admin to make this problem Public?");
    if (confirmReq) {
      try {
        const response = await fetch(`http://localhost:21081/api/problems/${problemId}/request-approval`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user_id: currentUserId })
        });
        const result = await response.json();
        if (response.ok) {
          alert("Public request submitted successfully!");
          await loadProblems();
        } else {
          alert(`Error: ${result.detail || "Request failed"}`);
        }
      } catch (err) {
        console.error(err);
        alert("Failed to submit request.");
      }
    }
  };

  const handleAdminApproveDirectly = async (e, problemId, action) => {
    e.stopPropagation();
    const confirmAction = window.confirm(`Are you sure you want to ${action} this problem?`);
    if (confirmAction) {
      try {
        const response = await fetch(`http://localhost:21081/api/problems/${problemId}/${action}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ admin_id: currentUserId })
        });
        const result = await response.json();
        if (response.ok) {
          alert(`Problem successfully ${action}ed!`);
          await loadProblems();
        } else {
          alert(`Error: ${result.detail || "Action failed"}`);
        }
      } catch (err) {
        console.error(err);
        alert("Failed to complete action.");
      }
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
    inputBg: "#ffffff",
    inputBorder: "#cbd5e1",
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
    inputBg: "#0c1524",
    inputBorder: "rgba(255,255,255,0.08)",
    shadow: "none",
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

  const labelStyle = {
    display: "block", fontSize: 11, fontWeight: 600,
    color: t.textSecondary, letterSpacing: "0.06em",
    textTransform: "uppercase", marginBottom: 6,
  };

  const renderStatusIndicator = (problem) => {
    const score = problem.best_score;
    const status = problem.best_status;

    if (score === null || score === undefined) {
      return (
        <span style={{
          width: 6, height: 6, borderRadius: "50%",
          background: isLight ? "#cbd5e1" : "#475569",
          display: "inline-block"
        }} title="Not attempted" />
      );
    }

    if (score === 100 || status === "accepted") {
      return (
        <span style={{
          width: 6, height: 6, borderRadius: "50%",
          background: "#10b981", display: "inline-block",
          boxShadow: "0 0 8px rgba(16,185,129,0.5)"
        }} title="Solved (100 pts)" />
      );
    }

    if (score > 0 && score < 100) {
      return (
        <span style={{
          width: 6, height: 6, borderRadius: "50%",
          background: "#f59e0b", display: "inline-block",
          boxShadow: "0 0 8px rgba(245,158,11,0.5)"
        }} title={`Partial Solved (${score} pts)`} />
      );
    }

    return (
      <span style={{
        width: 6, height: 6, borderRadius: "50%",
        background: "#ef4444", display: "inline-block",
        boxShadow: "0 0 8px rgba(239,68,68,0.5)"
      }} title="Attempted / Failed" />
    );
  };

  return (
    <div style={{ fontFamily: "'Inter var', 'Inter', sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');
        .tk-input:focus, .tk-select:focus {
          border-color: ${t.accent} !important;
          box-shadow: 0 0 0 3px ${t.accentBg} !important;
        }
        .tk-primary:hover { background: ${t.accentDark} !important; }
        .tk-ghost:hover { background: ${isLight ? "#f1f5f9" : "rgba(255,255,255,0.05)"} !important; }

        /* Custom scrollbar style */
        ::-webkit-scrollbar {
          width: 6px;
          height: 6px;
        }
        ::-webkit-scrollbar-track {
          background: rgba(15, 23, 42, 0.05);
        }
        ::-webkit-scrollbar-thumb {
          background: ${isLight ? "rgba(15,23,42,0.12)" : "rgba(255, 255, 255, 0.08)"};
          border-radius: 4px;
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
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                <line x1="9" y1="9" x2="15" y2="9" />
                <line x1="9" y1="13" x2="15" y2="13" />
                <line x1="9" y1="17" x2="11" y2="17" />
              </svg>
            </div>
            <h2 style={{
              margin: 0, fontSize: 18, fontWeight: 700,
              color: t.textPrimary, letterSpacing: "-0.02em",
            }}>
              Coding Problems
            </h2>
            <select
              value={filterMode}
              onChange={(e) => setFilterMode(e.target.value)}
              className="tk-select"
              style={{
                borderRadius: 8, border: `1px solid ${t.border}`,
                padding: "4px 10px", fontSize: 11, fontWeight: 700,
                background: t.inputBg, color: t.textPrimary,
                cursor: "pointer", outline: "none", marginLeft: 8
              }}
            >
              <option value="public">Public</option>
              <option value="private">My Problems</option>
              <option value="all">All</option>
            </select>
          </div>
          <p style={{ margin: 0, fontSize: 12, color: t.textSecondary, lineHeight: 1.5, paddingLeft: 40 }}>
            Browse through active programming challenges, test datasets, and machine learning models.
          </p>
        </div>

        {(role === "admin" || role === "contributor") && (
          <button
            type="button"
            onClick={() => { resetForm(); setIsModalOpen(true); }}
            className="tk-primary"
            style={{
              display: "flex", alignItems: "center", gap: 7,
              background: t.accent, color: "#fff", border: "none",
              borderRadius: 7, padding: "8px 16px",
              fontSize: 12, fontWeight: 600, letterSpacing: "0.02em",
              cursor: "pointer", transition: "all 0.15s",
              boxShadow: isLight ? "0 1px 3px rgba(5,150,105,0.3)" : "none",
            }}
          >
            <svg width="11" height="11" viewBox="0 0 11 11" fill="none">
              <path d="M5.5 1v9M1 5.5h9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
            Create Problem
          </button>
        )}
      </div>

      {error && !isModalOpen && (
        <div style={{
          marginBottom: 20, padding: "10px 14px", borderRadius: 8,
          background: isLight ? "#fff1f2" : "rgba(239,68,68,0.08)",
          border: `1px solid ${isLight ? "#fecdd3" : "rgba(239,68,68,0.2)"}`,
          color: isLight ? "#be123c" : "#f87171", fontSize: 12,
        }}>
          {error}
        </div>
      )}

      {isLoading ? (
        <div style={{
          background: t.surface, border: `1px solid ${t.border}`, borderRadius: 12, padding: "48px 24px", textAlign: "center", color: t.textSecondary, fontSize: 13,
        }}>
          <p className="animate-pulse">Loading problems workspace...</p>
        </div>
      ) : problems.length === 0 ? (
        <div style={{
          background: t.surface, border: `1px solid ${t.border}`, borderRadius: 12, padding: "48px 24px", textAlign: "center", color: t.textSecondary, fontSize: 13,
          boxShadow: t.shadow,
        }}>
          <div style={{ marginBottom: 8, opacity: 0.35, fontSize: 28 }}>📭</div>
          No created problems yet.
        </div>
      ) : (
        <div style={{
          background: t.surface,
          border: `1px solid ${t.border}`,
          borderRadius: 12, overflow: "hidden",
          boxShadow: t.shadow,
        }}>
          <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 950 }}>
            <thead>
              <tr style={{
                background: isLight ? t.tableHeadBg : t.tableHeadBg,
                borderBottom: `1px solid ${isLight ? t.accentDark + "55" : t.border}`,
              }}>
                {["✔", "ID", "Problem Name", "Category", "Points", "Visibility", "Request", "Actions", "Status"].map((h, i) => (
                  <th key={i} style={{
                    padding: "12px 14px",
                    textAlign: i === 0 || i === 4 || i === 5 || i === 6 || i === 7 || i === 8 ? "center" : "left",
                    fontSize: 10, fontWeight: 700, letterSpacing: "0.08em",
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
              {problems.map((problem, index) => {
                const isPrivate = problem.is_public === 0;
                const reqStatus = problem.request_status ? problem.request_status.toUpperCase() : "NONE";
                const bestStatus = problem.best_status ? problem.best_status.toLowerCase() : null;

                const cleanSlug = problem.name
                  .toLowerCase()
                  .replace(/[^\w\s-]/g, '')
                  .replace(/[-\s]+/g, '_')
                  .substring(0, 16);
                const displayId = `ml_${cleanSlug}_${problem.id}`;

                return (
                  <tr
                    key={problem.id}
                    onClick={() => navigate(`/livecoding/${problem.id}`, { state: { problem } })}
                    style={{
                      borderBottom: `1px solid ${isLight ? "#e2e8f0" : "rgba(255,255,255,0.04)"}`,
                      cursor: "pointer",
                      background: index % 2 === 0
                        ? "transparent"
                        : (isLight ? "#f8fafc" : "rgba(255,255,255,0.01)"),
                    }}
                    onMouseEnter={e => e.currentTarget.style.background = isLight ? "#f0fdf8" : "rgba(6,182,212,0.04)"}
                    onMouseLeave={e => e.currentTarget.style.background = index % 2 === 0 ? "transparent" : (isLight ? "#f8fafc" : "rgba(255,255,255,0.01)")}
                  >
                    {/* Status Dot */}
                    <td style={{ padding: "14px 10px", textAlign: "center", width: 40 }}>
                      {renderStatusIndicator(problem)}
                    </td>

                    {/* ID */}
                    <td style={{ padding: "14px 14px", width: 140 }}>
                      <span style={{
                        fontFamily: "'DM Mono', monospace", fontSize: 11,
                        color: isLight ? "#059669" : "#475569",
                        letterSpacing: "0.02em", fontWeight: 600,
                      }}>
                        {displayId}
                      </span>
                    </td>

                    {/* Name */}
                    <td style={{ padding: "14px 14px" }}>
                      <span style={{
                        fontSize: 13, fontWeight: 600,
                        color: isLight ? "#0f172a" : "#e2e8f0",
                        display: "block", lineHeight: 1.4,
                      }}>
                        {problem.name}
                      </span>
                    </td>

                    {/* Category */}
                    <td style={{ padding: "14px 14px", width: 150 }}>
                      <span style={{ fontSize: 12, color: t.textSecondary }}>
                        Machine Learning
                      </span>
                    </td>

                    {/* Points */}
                    <td style={{ padding: "14px 14px", width: 80, textAlign: "center" }}>
                      <span style={{ fontSize: 11, color: t.textSecondary, fontFamily: "'DM Mono', monospace" }}>
                        {problem.best_score !== null ? Number(problem.best_score).toFixed(2) : "0.00"}
                      </span>
                    </td>

                    {/* Visibility */}
                    <td style={{ padding: "14px 10px", width: 100, textAlign: "center" }}>
                      <span style={{
                        display: "inline-flex", padding: "2px 6px", borderRadius: 4, fontSize: 9, fontWeight: 700, textTransform: "uppercase",
                        background: isPrivate ? (isLight ? "#fffbeb" : "rgba(245,158,11,0.08)") : (isLight ? "#ecfdf5" : "rgba(16,185,129,0.08)"),
                        color: isPrivate ? (isLight ? "#d97706" : "#fbbf24") : (isLight ? "#065f46" : "#34d399"),
                        border: `1px solid ${isPrivate ? (isLight ? "#fde68a" : "rgba(245,158,11,0.2)") : (isLight ? "#6ee7b7" : "rgba(16,185,129,0.2)")}`,
                      }}>
                        {isPrivate ? "Private" : "Public"}
                      </span>
                    </td>

                    {/* Request Status */}
                    <td style={{ padding: "14px 10px", width: 150, textAlign: "center" }} onClick={(e) => e.stopPropagation()}>
                      {isPrivate ? (
                        role === "admin" ? (
                          <div style={{ display: "flex", gap: 6, justifyContent: "center" }}>
                            <button
                              type="button"
                              onClick={(e) => handleAdminApproveDirectly(e, problem.id, "approve")}
                              style={{
                                background: isLight ? "#059669" : "transparent",
                                border: isLight ? "none" : "1px solid rgba(16,185,129,0.3)",
                                color: isLight ? "#fff" : "#34d399",
                                padding: "4px 8px", borderRadius: 4,
                                fontSize: 10, fontWeight: 700, cursor: "pointer",
                              }}
                            >
                              Approve
                            </button>
                            <button
                              type="button"
                              onClick={(e) => handleAdminApproveDirectly(e, problem.id, "reject")}
                              style={{
                                background: isLight ? "#e11d48" : "transparent",
                                border: isLight ? "none" : "1px solid rgba(244,63,94,0.3)",
                                color: isLight ? "#fff" : "#fb7185",
                                padding: "4px 8px", borderRadius: 4,
                                fontSize: 10, fontWeight: 700, cursor: "pointer",
                              }}
                            >
                              Reject
                            </button>
                          </div>
                        ) : (
                          problem.author_id === currentUserId ? (
                            reqStatus === "PENDING" ? (
                              <span style={{ fontSize: 11, color: t.textMuted }}>Pending Review</span>
                            ) : (
                              <button
                                type="button"
                                onClick={(e) => handleRequestPublic(e, problem.id)}
                                style={{
                                  background: isLight ? "#d97706" : "transparent",
                                  border: isLight ? "none" : "1px solid rgba(245,158,11,0.3)",
                                  color: isLight ? "#fff" : "#fbbf24",
                                  padding: "4px 8px", borderRadius: 4,
                                  fontSize: 10, fontWeight: 700, cursor: "pointer",
                                }}
                              >
                                Request Public
                              </button>
                            )
                          ) : (
                            <span style={{ fontSize: 11, color: t.textMuted }}>-</span>
                          )
                        )
                      ) : (
                        <span style={{ fontSize: 10, fontWeight: 700, color: t.accent, textTransform: "uppercase" }}>Published</span>
                      )}
                    </td>

                    {/* Sửa và Xóa */}
                    <td style={{ padding: "14px 10px", width: 140, textAlign: "center" }} onClick={(e) => e.stopPropagation()}>
                      {role === "admin" || (role === "contributor" && problem.author_id === currentUserId) ? (
                        <div style={{ display: "flex", gap: 6, justifyContent: "center" }}>
                          <button
                            type="button"
                            onClick={(e) => handleEditProblemInit(e, problem)}
                            style={{
                              background: isLight ? "#4f46e5" : "transparent",
                              border: isLight ? "none" : "1px solid rgba(99,102,241,0.3)",
                              color: isLight ? "#fff" : "#a5b4fc",
                              padding: "4px 8px", borderRadius: 4,
                              fontSize: 10, fontWeight: 700, cursor: "pointer",
                            }}
                          >
                            Edit
                          </button>
                          <button
                            type="button"
                            onClick={(e) => handleDeleteProblemDirect(e, problem)}
                            style={{
                              background: isLight ? "#e11d48" : "transparent",
                              border: isLight ? "none" : "1px solid rgba(244,63,94,0.3)",
                              color: isLight ? "#fff" : "#fb7185",
                              padding: "4px 8px", borderRadius: 4,
                              fontSize: 10, fontWeight: 700, cursor: "pointer",
                            }}
                          >
                            Delete
                          </button>
                        </div>
                      ) : (
                        <span style={{ fontSize: 11, color: t.textMuted }}>-</span>
                      )}
                    </td>

                    {/* Status Badge */}
                    <td style={{ padding: "14px 14px", width: 130, textAlign: "center" }}>
                      {!bestStatus ? (
                        <span style={{
                          display: "inline-flex", padding: "2px 6px", borderRadius: 4, fontSize: 9, fontWeight: 700, textTransform: "uppercase",
                          background: isLight ? "#f1f5f9" : "rgba(255,255,255,0.04)",
                          color: t.textMuted, border: `1px solid ${t.border}`,
                        }}>
                          Unattempted
                        </span>
                      ) : bestStatus === "accepted" ? (
                        <span style={{
                          display: "inline-flex", padding: "2px 6px", borderRadius: 4, fontSize: 9, fontWeight: 700, textTransform: "uppercase",
                          background: isLight ? "#ecfdf5" : "rgba(16,185,129,0.08)",
                          color: isLight ? "#065f46" : "#34d399",
                          border: `1px solid ${isLight ? "#6ee7b7" : "rgba(16,185,129,0.2)"}`,
                        }}>
                          Accepted
                        </span>
                      ) : bestStatus === "wrong_answer" || bestStatus === "wrong answer" ? (
                        <span style={{
                          display: "inline-flex", padding: "2px 6px", borderRadius: 4, fontSize: 9, fontWeight: 700, textTransform: "uppercase",
                          background: isLight ? "#fff1f2" : "rgba(244,63,94,0.08)",
                          color: isLight ? "#9f1239" : "#fb7185",
                          border: `1px solid ${isLight ? "#fecdd3" : "rgba(244,63,94,0.2)"}`,
                        }}>
                          Wrong Answer
                        </span>
                      ) : bestStatus === "runtime_error" || bestStatus === "runtime error" ? (
                        <span style={{
                          display: "inline-flex", padding: "2px 6px", borderRadius: 4, fontSize: 9, fontWeight: 700, textTransform: "uppercase",
                          background: isLight ? "#fffbeb" : "rgba(245,158,11,0.08)",
                          color: isLight ? "#d97706" : "#fbbf24",
                          border: `1px solid ${isLight ? "#fde68a" : "rgba(245,158,11,0.2)"}`,
                        }}>
                          Runtime Error
                        </span>
                      ) : (
                        <span style={{
                          display: "inline-flex", padding: "2px 6px", borderRadius: 4, fontSize: 9, fontWeight: 700, textTransform: "uppercase",
                          background: isLight ? "#f0f9ff" : "rgba(6,182,212,0.08)",
                          color: isLight ? "#0369a1" : "#22d3ee",
                          border: `1px solid ${isLight ? "#bae6fd" : "rgba(6,182,212,0.2)"}`,
                        }}>
                          TLE
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* DIALOG PORTAL */}
      {isModalOpen && createPortal(
        <div style={{
          position: "fixed", inset: 0, zIndex: 1200, display: "flex",
          alignItems: "flex-start", justifyContent: "center", overflowY: "auto",
          background: "rgba(0,0,0,0.65)", padding: "40px 16px"
        }}>
          <div style={{
            margin: "auto", width: "100%", maxWidth: 768,
            background: t.surface,
            border: `1px solid ${isLight ? t.accentBorder : t.border}`,
            borderRadius: 14, padding: 24,
            boxShadow: isLight ? "0 10px 25px rgba(0,0,0,0.15)" : "0 10px 25px rgba(0,0,0,0.5)",
            position: "relative", overflow: "hidden",
          }}>
            {/* Top stripe */}
            <div style={{
              position: "absolute", top: 0, left: 0, right: 0, height: 3,
              background: "linear-gradient(90deg, #10b981, #06b6d4, #818cf8)",
            }} />

            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
              <h3 style={{
                margin: 0, fontSize: 14, fontWeight: 700,
                color: isLight ? "#059669" : "#22d3ee",
                textTransform: "uppercase", letterSpacing: "0.1em",
              }}>
                {editingProblemId ? "Edit Problem" : "Create Problem"}
              </h3>
              <button
                type="button"
                onClick={closeModal}
                className="tk-ghost"
                style={{
                  background: "transparent", border: `1px solid ${t.border}`,
                  color: t.textSecondary, borderRadius: 5, padding: "4px 10px",
                  fontSize: 11, fontWeight: 600, cursor: "pointer",
                }}
              >
                Close
              </button>
            </div>

            {error && (
              <div style={{
                marginBottom: 16, padding: "10px 14px", borderRadius: 8,
                background: isLight ? "#fff1f2" : "rgba(239,68,68,0.08)",
                border: `1px solid ${isLight ? "#fecdd3" : "rgba(239,68,68,0.2)"}`,
                color: isLight ? "#be123c" : "#f87171", fontSize: 12,
              }}>
                {error}
              </div>
            )}

            <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div>
                  <label style={labelStyle}>Name *</label>
                  <input
                    name="name"
                    required
                    value={formData.name}
                    onChange={handleChange}
                    className="tk-input"
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={labelStyle}>Source</label>
                  <input
                    name="source"
                    value={formData.source}
                    onChange={handleChange}
                    className="tk-input"
                    style={inputStyle}
                  />
                </div>
              </div>

              <div>
                <label style={labelStyle}>Statement Markdown *</label>
                <textarea
                  name="statement_markdown"
                  required
                  rows={4}
                  value={formData.statement_markdown}
                  onChange={handleChange}
                  className="tk-input"
                  style={{ ...inputStyle, resize: "vertical", lineHeight: 1.6 }}
                />
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div>
                  <label style={labelStyle}>Theory Markdown</label>
                  <textarea
                    name="theory_markdown"
                    rows={3}
                    value={formData.theory_markdown}
                    onChange={handleChange}
                    className="tk-input"
                    style={{ ...inputStyle, resize: "vertical", lineHeight: 1.6 }}
                  />
                </div>
                <div>
                  <label style={labelStyle}>Tutorial Markdown</label>
                  <textarea
                    name="tutorial_markdown"
                    rows={3}
                    value={formData.tutorial_markdown}
                    onChange={handleChange}
                    className="tk-input"
                    style={{ ...inputStyle, resize: "vertical", lineHeight: 1.6 }}
                  />
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div>
                  <label style={labelStyle}>Solution Markdown</label>
                  <textarea
                    name="solution_markdown"
                    rows={3}
                    value={formData.solution_markdown}
                    onChange={handleChange}
                    className="tk-input"
                    style={{ ...inputStyle, resize: "vertical", lineHeight: 1.6 }}
                  />
                </div>
                <div>
                  <label style={labelStyle}>Coding Markdown</label>
                  <textarea
                    name="coding_markdown"
                    rows={3}
                    value={formData.coding_markdown}
                    onChange={handleChange}
                    className="tk-input"
                    style={{ ...inputStyle, resize: "vertical", lineHeight: 1.6 }}
                  />
                </div>
              </div>

              <div>
                <label style={labelStyle}>Checker Markdown</label>
                <textarea
                  name="checker_markdown"
                  rows={3}
                  value={formData.checker_markdown}
                  onChange={handleChange}
                  className="tk-input"
                  style={{ ...inputStyle, resize: "vertical", lineHeight: 1.6 }}
                />
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div>
                  <label style={labelStyle}>
                    Input Zip {editingProblemId ? "(Optional, overwrites manager)" : "File (.zip)"}
                  </label>
                  <input
                    type="file"
                    accept=".zip"
                    onChange={(e) => setInputZipFile(e.target.files[0])}
                    style={{ fontSize: 12, color: t.textSecondary }}
                  />
                </div>
                <div>
                  <label style={labelStyle}>
                    Output Zip {editingProblemId ? "(Optional, overwrites manager)" : "File (.zip)"}
                  </label>
                  <input
                    type="file"
                    accept=".zip"
                    onChange={(e) => setOutputZipFile(e.target.files[0])}
                    style={{ fontSize: 12, color: t.textSecondary }}
                  />
                </div>
              </div>

              {editingProblemId && (
                <div style={{ display: "flex", flexDirection: "column", gap: 12, borderTop: `1px solid ${t.border}`, paddingTop: 14 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <label style={{ ...labelStyle, marginBottom: 0 }}>Test Cases Manager ({testcases.length})</label>
                    <button
                      type="button"
                      onClick={handleAddTestcase}
                      style={{
                        background: t.accentBg,
                        border: `1px solid ${t.accentBorder}`,
                        color: t.accent,
                        borderRadius: 6,
                        padding: "4px 10px",
                        fontSize: 11,
                        fontWeight: 600,
                        cursor: "pointer",
                      }}
                    >
                      + Add Test Case
                    </button>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 12, maxHeight: 220, overflowY: "auto", paddingRight: 4 }}>
                    {testcases.map((tc, idx) => (
                      <div
                        key={tc.id || idx}
                        style={{
                          border: `1px solid ${t.border}`,
                          borderRadius: 8,
                          padding: 12,
                          background: isLight ? "#f8fafc" : "rgba(255,255,255,0.02)",
                          position: "relative"
                        }}
                      >
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                          <span style={{ fontSize: 11, fontWeight: 700, color: t.accent }}>Test Case #{idx + 1}</span>
                          <button
                            type="button"
                            onClick={() => handleDeleteTestcase(idx)}
                            style={{
                              background: "rgba(239, 68, 68, 0.1)",
                              border: "1px solid rgba(239, 68, 68, 0.3)",
                              color: "#ef4444",
                              borderRadius: 4,
                              padding: "2px 6px",
                              fontSize: 10,
                              fontWeight: 600,
                              cursor: "pointer",
                            }}
                          >
                            Delete
                          </button>
                        </div>
                        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                          <div>
                            <span style={{ display: "block", fontSize: 10, color: t.textSecondary, marginBottom: 4 }}>Input JSON</span>
                            <textarea
                              value={tc.input}
                              onChange={(e) => handleTestcaseChange(idx, "input", e.target.value)}
                              rows={3}
                              className="tk-input"
                              style={{
                                ...inputStyle,
                                fontFamily: "'DM Mono', monospace",
                                fontSize: 11,
                                resize: "vertical"
                              }}
                            />
                          </div>
                          <div>
                            <span style={{ display: "block", fontSize: 10, color: t.textSecondary, marginBottom: 4 }}>Output JSON</span>
                            <textarea
                              value={tc.output}
                              onChange={(e) => handleTestcaseChange(idx, "output", e.target.value)}
                              rows={3}
                              className="tk-input"
                              style={{
                                ...inputStyle,
                                fontFamily: "'DM Mono', monospace",
                                fontSize: 11,
                                resize: "vertical"
                              }}
                            />
                          </div>
                        </div>
                      </div>
                    ))}
                    {testcases.length === 0 && (
                      <div style={{ textAlign: "center", padding: 20, color: t.textSecondary, fontSize: 12 }}>
                        No test cases defined. Click "+ Add Test Case" to create one.
                      </div>
                    )}
                  </div>
                </div>
              )}

              <div style={{ display: "flex", justifyContent: "flex-end", gap: 10, paddingTop: 10 }}>
                <button
                  type="button"
                  onClick={closeModal}
                  className="tk-ghost"
                  style={{
                    background: "transparent", border: `1px solid ${t.border}`, color: t.textSecondary,
                    borderRadius: 7, padding: "8px 16px", fontSize: 12, fontWeight: 500,
                    cursor: "pointer", transition: "background 0.15s",
                  }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="tk-primary"
                  style={{
                    background: t.accent, border: "none", color: "#fff",
                    borderRadius: 7, padding: "8px 20px",
                    fontSize: 12, fontWeight: 600, cursor: "pointer",
                    transition: "background 0.15s",
                    boxShadow: isLight ? "0 1px 3px rgba(5,150,105,0.35)" : "none",
                    opacity: isSubmitting ? 0.6 : 1,
                  }}
                >
                  {isSubmitting ? "Processing..." : editingProblemId ? "Save Changes" : "Create"}
                </button>
              </div>
            </form>
          </div>
        </div>
        , document.body)}

      {/* DELETE CONFIRMATION PORTAL */}
      {deletingProblem && createPortal(
        <div style={{
          position: "fixed", inset: 0, zIndex: 1300, display: "flex",
          alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.65)",
          padding: 16
        }}>
          <div style={{
            width: "100%", maxWidth: 400, background: t.surface,
            border: `1px solid ${isLight ? t.accentBorder : t.border}`,
            borderRadius: 12, padding: 20, boxShadow: "0 10px 25px rgba(0,0,0,0.3)",
            fontFamily: "'Inter var', 'Inter', sans-serif", color: t.textPrimary,
            position: "relative"
          }}>
            {/* red top stripe */}
            <div style={{
              position: "absolute", top: 0, left: 0, right: 0, height: 3,
              background: "#ef4444"
            }} />
            <h4 style={{ margin: "0 0 10px 0", fontSize: 15, fontWeight: 700, color: isLight ? "#be123c" : "#f87171" }}>
              Delete Problem
            </h4>
            <p style={{ margin: "0 0 20px 0", fontSize: 12, color: t.textSecondary, lineHeight: 1.5 }}>
              Are you sure you want to delete the problem <strong>{deletingProblem.name}</strong> permanently? This action cannot be undone.
            </p>
            <div style={{ display: "flex", justifyContent: "flex-end", gap: 10 }}>
              <button
                type="button"
                onClick={() => setDeletingProblem(null)}
                className="tk-ghost"
                style={{
                  background: "transparent", border: `1px solid ${t.border}`, color: t.textSecondary,
                  borderRadius: 6, padding: "6px 12px", fontSize: 11, fontWeight: 600, cursor: "pointer"
                }}
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={confirmDeleteProblem}
                style={{
                  background: "#ef4444", border: "none", color: "#fff",
                  borderRadius: 6, padding: "6px 14px", fontSize: 11, fontWeight: 700, cursor: "pointer"
                }}
              >
                Delete
              </button>
            </div>
          </div>
        </div>,
        document.body
      )}
    </div>
  );
}