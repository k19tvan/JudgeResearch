import React, { useEffect, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { createPortal } from "react-dom";
import ReactMarkdown from "react-markdown";
import Editor from "@monaco-editor/react";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { fetchProblemContent, runProblem, submitProblem, fetchProblemSubmissions } from "../api";
import DiscussionTab from "./tabs/DiscussionTab";

const CONTENT_TABS = [
  { key: "statement", label: "Description" },
  { key: "theory", label: "Theory" },
  { key: "tutorial", label: "Editorial" },
  { key: "submissions", label: "Submissions" },
  { key: "discussion", label: "Discussion" },
];

function normalizeEditorCode(raw) {
  const text = raw || "";
  const trimmed = text.trim();
  const fenceMatch = trimmed.match(/^```[a-zA-Z0-9_-]*\n([\s\S]*?)\n```$/);
  if (fenceMatch) {
    return fenceMatch[1];
  }
  return text;
}

export default function LiveCodingPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { problemId } = useParams();
  const baseProblem = location.state?.problem;

  // Đọc chế độ theme (sáng/tối) đồng bộ trực tiếp từ trang chủ
  const [theme] = useState(() => localStorage.getItem("home_theme") || "dark");
  const isLight = theme === "light";

  // Đồng bộ cấu trúc bảng màu của trang Home
  const t = {
    pageBg: isLight ? "#eaf2f0" : "#080C14",
    surface: isLight ? "#ffffff" : "#0D1117",
    surfaceAlt: isLight ? "#f0faf6" : "#111827",
    border: isLight ? "rgba(15,23,42,0.10)" : "rgba(255,255,255,0.06)",
    accent: isLight ? "#059669" : "#10B981",
    accentDim: isLight ? "rgba(5,150,105,0.12)" : "rgba(16,185,129,0.10)",
    accentBorder: isLight ? "rgba(5,150,105,0.30)" : "rgba(16,185,129,0.28)",
    textPrimary: isLight ? "#0f172a" : "#F1F5F9",
    textSecondary: isLight ? "#475569" : "#64748B",
    textMuted: isLight ? "#94a3b8" : "#334155",
    shadow: isLight ? "0 1px 10px rgba(15,23,42,0.08)" : "0 1px 10px rgba(0,0,0,0.35)",
  };

  const [activeContentTab, setActiveContentTab] = useState("statement");
  const [problem, setProblem] = useState(baseProblem || null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");

  const [code, setCode] = useState(() => {
    const savedDraft = localStorage.getItem(`draft_code_${problemId}`);
    if (savedDraft && savedDraft.trim() !== "" && savedDraft !== "undefined" && savedDraft !== "null") {
      return savedDraft;
    }
    return normalizeEditorCode(baseProblem?.coding_markdown || "");
  });

  const [consoleOutput, setConsoleOutput] = useState("Run your code to see results here...");
  const [isConsoleRunning, setIsConsoleRunning] = useState(false);

  const [submissions, setSubmissions] = useState([]);
  const [isSubmissionsLoading, setIsSubmissionsLoading] = useState(false);
  const [expandedSubId, setExpandedSubId] = useState(null);

  const currentUserId = Number(localStorage.getItem("user_id") || "1");
  const role = localStorage.getItem("user_role") || "user";

  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [editError, setEditError] = useState("");
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
  const [isDeleteConfirmOpen, setIsDeleteConfirmOpen] = useState(false);

  const closeEditModal = () => {
    setIsEditModalOpen(false);
    setInputZipFile(null);
    setOutputZipFile(null);
    setTestcases([]);
    setEditError("");
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

  const loadProblemContent = async () => {
    setIsLoading(true);
    setError("");
    try {
      const result = await fetchProblemContent(problemId, currentUserId);
      const contentProblem = result?.data;
      if (contentProblem) {
        setProblem((prev) => ({
          ...(prev || {}),
          ...contentProblem,
        }));

        const savedDraft = localStorage.getItem(`draft_code_${problemId}`);
        if (savedDraft && savedDraft.trim() !== "" && savedDraft !== "undefined" && savedDraft !== "null") {
          setCode(savedDraft);
        } else {
          setCode(normalizeEditorCode(contentProblem.coding_markdown || ""));
        }
      }
    } catch (err) {
      setError(err.message || "Failed to load problem content");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadProblemContent();
  }, [problemId, currentUserId]);

  const handleEditProblemInit = async () => {
    if (!problem) return;
    setFormData({
      name: problem.name || "",
      source: problem.source || "",
      statement_markdown: problem.statement_markdown || "",
      theory_markdown: problem.theory_markdown || "",
      tutorial_markdown: problem.tutorial_markdown || "",
      solution_markdown: problem.solution_markdown || "",
      coding_markdown: problem.coding_markdown || "",
      checker_markdown: problem.checker_markdown || "",
    });
    setEditError("");

    try {
      const tcResponse = await fetch(`http://localhost:21081/api/problems/${problemId}/testcases`);
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

    setIsEditModalOpen(true);
  };

  const handleEditSubmit = async (e) => {
    e.preventDefault();
    setEditError("");
    setIsSubmitting(true);

    try {
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

      const response = await fetch(`http://localhost:21081/api/problems/${problemId}`, {
        method: "PUT",
        body: formPayload
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail || "Edit problem failed");
      }
      closeEditModal();
      await loadProblemContent();
    } catch (err) {
      setEditError(err.message || "Failed to update problem");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDeleteProblemDirect = () => {
    setIsDeleteConfirmOpen(true);
  };

  const confirmDeleteProblem = async () => {
    setIsDeleteConfirmOpen(false);
    try {
      const response = await fetch(`http://localhost:21081/api/problems/${problemId}`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: currentUserId })
      });
      if (response.ok) {
        alert("Problem deleted successfully!");
        navigate("/", { state: { activeTab: "PROBLEMS" } });
      } else {
        const result = await response.json();
        alert(`Error: ${result.detail || "Delete failed"}`);
      }
    } catch (err) {
      alert("Failed to delete problem: " + err.message);
    }
  };

  const handleFormChange = (e) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

  const loadSubmissionsList = async () => {
    setIsSubmissionsLoading(true);
    try {
      const resp = await fetchProblemSubmissions(problemId, currentUserId);
      setSubmissions(resp?.data || []);
    } catch (err) {
      console.error("Failed to load submissions:", err);
    } finally {
      setIsSubmissionsLoading(false);
    }
  };

  useEffect(() => {
    if (activeContentTab === "submissions") {
      loadSubmissionsList();
    }
  }, [activeContentTab, problemId]);

  useEffect(() => {
    if (!problemId || !problem || isLoading) return;

    const delayDebounceFn = setTimeout(() => {
      localStorage.setItem(`draft_code_${problemId}`, code);
    }, 500);

    return () => clearTimeout(delayDebounceFn);
  }, [code, problemId, problem, isLoading]);

  const handleResetCode = () => {
    const confirmReset = window.confirm("Are you sure you want to reset your code to the default template?");
    if (confirmReset && problem) {
      const defaultCode = normalizeEditorCode(problem.coding_markdown || "");
      setCode(defaultCode);
      localStorage.setItem(`draft_code_${problemId}`, defaultCode);
    }
  };

  const handleRunCode = () => {
    (async () => {
      setIsConsoleRunning(true);
      setConsoleOutput("Running first test case...\n");
      try {
        const resp = await runProblem(problemId, code);
        const parts = [];
        parts.push(`Status: ${resp.status}`);
        if (resp.message) parts.push(`Message: ${resp.message}`);
        parts.push(`Output: ${JSON.stringify(resp.output)}`);
        parts.push(`Expected: ${JSON.stringify(resp.expected)}`);
        if (resp.elapsed_ms) parts.push(`Time: ${resp.elapsed_ms} ms`);
        setConsoleOutput(parts.join('\n'));
      } catch (err) {
        setConsoleOutput(`Error: ${err.message}`);
      } finally {
        setIsConsoleRunning(false);
      }
    })();
  };

  const handleSubmitCode = () => {
    (async () => {
      setIsConsoleRunning(true);
      setConsoleOutput("Submitting... Evaluating on hidden dataset...\n");
      try {
        const resp = await submitProblem(problemId, currentUserId, code);
        const parts = [];
        parts.push(`Final status: ${resp.status}`);
        parts.push(`Score: ${resp.score}`);
        parts.push(`Submission ID: ${resp.submission_id}`);
        parts.push('Test results:');
        resp.results.forEach((r) => {
          parts.push(`  TC ${r.testcase}: ${r.status} (${JSON.stringify(r.user_output)})`);
        });
        setConsoleOutput(parts.join('\n'));

        if (activeContentTab === "submissions") {
          loadSubmissionsList();
        }
      } catch (err) {
        setConsoleOutput(`Error: ${err.message}`);
      } finally {
        setIsConsoleRunning(false);
      }
    })();
  };

  const handleLoadSubmittedCode = (submittedCode) => {
    const confirmLoad = window.confirm("Do you want to load this submitted code into your codespace? This will overwrite your current workspace.");
    if (confirmLoad) {
      setCode(submittedCode);
      localStorage.setItem(`draft_code_${problemId}`, submittedCode);
    }
  };

  const toggleExpandSubmission = (id) => {
    setExpandedSubId(expandedSubId === id ? null : id);
  };

  const getStatusBadgeClass = (status) => {
    const lowerStatus = status.toLowerCase();
    if (lowerStatus === "accepted") {
      return isLight
        ? "bg-emerald-50 text-emerald-600 border border-emerald-200"
        : "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20";
    }
    if (lowerStatus === "wrong_answer" || lowerStatus === "wrong answer") {
      return isLight
        ? "bg-rose-50 text-rose-600 border border-rose-200"
        : "bg-rose-500/10 text-red-400 border border-red-500/20";
    }
    if (lowerStatus === "runtime_error" || lowerStatus === "runtime error") {
      return isLight
        ? "bg-amber-50 text-amber-600 border border-amber-200"
        : "bg-amber-500/10 text-amber-400 border border-amber-500/20";
    }
    return isLight
      ? "bg-cyan-50 text-cyan-600 border border-cyan-200"
      : "bg-cyan-500/10 text-cyan-400 border border-cyan-500/20";
  };

  const contentMap = {
    statement: problem?.statement_markdown || "No statement content available for this problem.",
    theory: problem?.theory_markdown || "No theory content available for this problem.",
    tutorial: problem?.tutorial_markdown || "No tutorial content available for this problem.",
  };

  const inputStyle = {
    width: "100%", boxSizing: "border-box",
    background: isLight ? "#ffffff" : "#0c1524",
    border: `1px solid ${isLight ? "#cbd5e1" : "rgba(255,255,255,0.08)"}`,
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

  return (
    <div
      className="flex h-screen w-screen flex-col overflow-hidden select-none transition-colors duration-150"
      style={{ background: t.pageBg, color: t.textPrimary, fontFamily: "'Sora', sans-serif" }}
    >
      <style>{`
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

        /* Markdown Typography */
        .markdown-preview h1, .markdown-preview h2, .markdown-preview h3, .markdown-preview h4 {
          color: ${isLight ? "#0f172a" : "#f8fafc"};
          font-weight: 700;
          margin-top: 1.25rem;
          margin-bottom: 0.5rem;
        }
        .markdown-preview h1 { font-size: 1.4rem; border-bottom: 1px solid ${t.border}; padding-bottom: 0.5rem;}
        .markdown-preview h2 { font-size: 1.2rem; }
        .markdown-preview h3 { font-size: 1.05rem; }
        .markdown-preview p { margin-bottom: 0.75rem; color: ${isLight ? "#334155" : "#cbd5e1"}; line-height: 1.6; }
        .markdown-preview ul, .markdown-preview ol {
          padding-left: 1.25rem;
          margin-bottom: 0.75rem;
          color: ${isLight ? "#334155" : "#cbd5e1"};
        }
        .markdown-preview ul { list-style-type: disc; }
        .markdown-preview ol { list-style-type: decimal; }
        .markdown-preview li { margin-bottom: 0.25rem; }
        .markdown-preview code {
          background: ${isLight ? "rgba(15,23,42,0.04)" : "rgba(15, 23, 42, 0.8)"};
          border: 1px solid ${t.border};
          border-radius: 0.25rem;
          padding: 0.1rem 0.3rem;
          color: ${isLight ? "#059669" : "#10B981"};
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.85em;
        }
        .markdown-preview pre {
          background: ${isLight ? "rgba(15,23,42,0.02)" : "rgba(2, 6, 23, 0.8)"};
          border: 1px solid ${t.border};
          border-radius: 0.5rem;
          padding: 0.75rem;
          margin: 0.75rem 0;
          overflow-x: auto;
        }
        .markdown-preview pre code {
          border: none;
          padding: 0;
          background: transparent;
          color: ${isLight ? "#0f172a" : "#e2e8f0"};
          font-size: 0.9em;
        }
        .markdown-preview blockquote {
          border-left: 4px solid ${t.accent};
          background: ${t.accentDim};
          padding: 0.5rem 0.75rem;
          margin: 0.75rem 0;
          border-radius: 0 0.375rem 0.375rem 0;
        }
        .markdown-preview a {
          color: ${t.accent};
          text-decoration: none;
        }
        .markdown-preview a:hover {
          text-decoration: underline;
        }
        .tk-ghost-btn:hover {
          background: ${isLight ? "rgba(15,23,42,0.04)" : "rgba(255,255,255,0.05)"} !important;
        }
      `}</style>

      {/* HEADER BAR */}
      <header
        className="flex h-12 w-full items-center justify-between px-5 transition-colors duration-150"
        style={{ background: isLight ? "#f8fffc" : "#0D1117", borderBottom: `1px solid ${t.border}` }}
      >
        <div className="flex items-center gap-3">
          <span className="text-xl">🤖</span>
          <div className="h-4 w-[1px]" style={{ background: t.border }} />
          <h1 className="text-sm font-bold tracking-wide" style={{ color: t.textPrimary }}>
            {problem?.title || `Problem #${problemId}`}
          </h1>
          <span
            className="rounded px-2 py-0.5 text-[10px] font-semibold"
            style={{ background: t.accentDim, color: t.accent, border: `1px solid ${t.accentBorder}` }}
          >
            ML JUDGE
          </span>
        </div>
        <div className="flex items-center gap-2">
          {(role === "admin" || (role === "contributor" && problem && problem.author_id === currentUserId)) && (
            <>
              <button
                type="button"
                onClick={handleEditProblemInit}
                className="tk-ghost-btn"
                style={{
                  background: t.accentDim,
                  border: `1px solid ${t.accentBorder}`,
                  color: t.accent,
                  borderRadius: 8,
                  padding: "6px 14px",
                  fontSize: 12,
                  fontWeight: 600,
                  cursor: "pointer",
                  transition: "all 0.15s",
                }}
              >
                EDIT
              </button>
              <button
                type="button"
                onClick={handleDeleteProblemDirect}
                style={{
                  background: isLight ? "#fff1f2" : "rgba(244,63,94,0.08)",
                  border: `1px solid ${isLight ? "#fecdd3" : "rgba(244,63,94,0.28)"}`,
                  color: isLight ? "#be123c" : "#fb7185",
                  borderRadius: 8,
                  padding: "6px 14px",
                  fontSize: 12,
                  fontWeight: 600,
                  cursor: "pointer",
                  transition: "all 0.15s",
                }}
              >
                DELETE
              </button>
            </>
          )}
          <button
            type="button"
            onClick={() => navigate("/", { state: { activeTab: "PROBLEMS" } })}
            className="tk-ghost-btn"
            style={{
              background: "transparent",
              border: `1px solid ${t.border}`,
              color: t.textPrimary,
              borderRadius: 8,
              padding: "6px 14px",
              fontSize: 12,
              fontWeight: 600,
              cursor: "pointer",
              transition: "all 0.15s",
            }}
          >
            BACK
          </button>
        </div>
      </header>

      {/* WORKSPACE AREA */}
      <main className="flex flex-1 w-full overflow-hidden p-2 gap-2" style={{ background: isLight ? "#f1f5f9" : "#080711" }}>

        {/* LEFT COLUMN: Problem Info Panel, Submissions & Discussions */}
        <section
          className="flex w-1/2 flex-col h-full rounded-xl transition-colors duration-150"
          style={{ background: t.surface, border: `1px solid ${t.border}`, boxShadow: t.shadow, overflow: "hidden" }}
        >
          <nav
            className="flex"
            style={{ background: isLight ? "rgba(5,150,105,0.04)" : "rgba(255,255,255,0.02)", borderBottom: `1px solid ${t.border}` }}
          >
            {CONTENT_TABS.map((tab) => (
              <button
                key={tab.key}
                type="button"
                onClick={() => setActiveContentTab(tab.key)}
                style={{
                  padding: "12px 16px",
                  fontSize: 12,
                  fontWeight: 600,
                  letterSpacing: "0.02em",
                  background: activeContentTab === tab.key ? t.accentDim : "transparent",
                  color: activeContentTab === tab.key ? t.accent : t.textSecondary,
                  borderBottom: `2px solid ${activeContentTab === tab.key ? t.accent : "transparent"}`,
                  transition: "all 0.15s",
                  cursor: "pointer",
                }}
              >
                {tab.label}
              </button>
            ))}
          </nav>

          {/* Internal Scrollable Content Box */}
          <div className="flex-1 overflow-y-auto p-5" style={{ background: isLight ? "#ffffff" : "rgba(0,0,0,0.08)" }}>
            {activeContentTab === "discussion" ? (
              /* PANEL HIỂN THỊ THẢO LUẬN BÀI TẬP (DISCUSSION) */
              <DiscussionTab problemId={problemId} isLight={isLight} />
            ) : activeContentTab === "submissions" ? (
              /* PANEL HIỂN THỊ DANH SÁCH SUBMISSIONS */
              <div className="space-y-3">
                <h3 className="text-xs font-bold uppercase tracking-wider mb-3" style={{ color: t.textPrimary, fontFamily: "'JetBrains Mono', monospace" }}>Submission History</h3>

                {isSubmissionsLoading ? (
                  <p className="text-xs animate-pulse" style={{ color: t.textSecondary }}>Loading submissions...</p>
                ) : submissions.length === 0 ? (
                  <p className="text-xs" style={{ color: t.textSecondary }}>You have no submissions for this problem yet.</p>
                ) : (
                  submissions.map((sub, idx) => (
                    <div
                      key={sub.id}
                      className="rounded-lg border overflow-hidden transition"
                      style={{
                        background: isLight ? "rgba(15,23,42,0.01)" : "rgba(255,255,255,0.01)",
                        borderColor: t.border,
                      }}
                    >
                      <div
                        onClick={() => toggleExpandSubmission(sub.id)}
                        className="flex cursor-pointer items-center justify-between p-3 select-none hover:opacity-90"
                      >
                        <div className="flex items-center gap-3">
                          <span className={`text-[10px] font-semibold px-2 py-1 rounded-md ${getStatusBadgeClass(sub.status)}`}>
                            {sub.status.replace("_", " ").toUpperCase()}
                          </span>
                          <span className="text-xs font-semibold" style={{ color: t.textPrimary }}>
                            Score: <span style={{ color: t.accent }}>{sub.score}</span>
                          </span>
                        </div>

                        <div className="flex items-center gap-3 text-[10px]" style={{ color: t.textSecondary }}>
                          <span>{new Date(sub.created_at).toLocaleString()}</span>
                          <span className="text-xs">{expandedSubId === sub.id ? "▲" : "▼"}</span>
                        </div>
                      </div>

                      {expandedSubId === sub.id && (
                        <div
                          className="border-t p-3 space-y-3"
                          style={{
                            background: isLight ? "#ffffff" : "rgba(0, 0, 0, 0.2)",
                            borderColor: t.border,
                          }}
                        >
                          <div className="flex justify-between items-center">
                            <span className="text-[10px] font-mono" style={{ color: t.textSecondary }}>Submission #{sub.id} Details</span>
                            <button
                              type="button"
                              onClick={() => handleLoadSubmittedCode(sub.submitted_code)}
                              style={{
                                background: t.accentDim,
                                border: `1px solid ${t.accentBorder}`,
                                color: t.accent,
                                borderRadius: 6,
                                padding: "4px 10px",
                                fontSize: 10,
                                fontWeight: 600,
                                cursor: "pointer",
                                transition: "all 0.15s",
                              }}
                            >
                              LOAD CODE TO EDITOR
                            </button>
                          </div>

                          {sub.test_results && sub.test_results.length > 0 ? (
                            <div
                              className="space-y-1.5 font-mono text-[11px] p-2 rounded"
                              style={{ background: isLight ? "#f1f5f9" : "rgba(0,0,0,0.3)" }}
                            >
                              {sub.test_results.map((tc, idx) => (
                                <div key={idx} className="flex justify-between items-start py-0.5">
                                  <span style={{ color: t.textSecondary }}>{tc.testcase}:</span>
                                  <span style={{ color: tc.status === "Accepted" ? "#10b981" : "#ef4444", fontWeight: 600 }}>
                                    {tc.status}
                                  </span>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <p className="text-[10px]" style={{ color: t.textSecondary }}>No test case details recorded.</p>
                          )}
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            ) : (
              /* PANEL HIỂN THỊ CÁC TAB KHÁC */
              <div className="markdown-preview text-sm">
                <ReactMarkdown
                  remarkPlugins={[remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                >
                  {contentMap[activeContentTab] || ""}
                </ReactMarkdown>
              </div>
            )}
          </div>
        </section>

        {/* RIGHT COLUMN: Code Editor & Output Console */}
        <section className="flex w-1/2 flex-col h-full gap-2 overflow-hidden">

          {/* EDITOR PANEL */}
          <div
            className="flex flex-1 flex-col rounded-xl overflow-hidden"
            style={{ background: t.surface, border: `1px solid ${t.border}`, boxShadow: t.shadow }}
          >
            <div
              className="flex items-center justify-between px-4 py-2"
              style={{ background: isLight ? "rgba(5,150,105,0.04)" : "rgba(255,255,255,0.02)", borderBottom: `1px solid ${t.border}` }}
            >
              <div className="flex items-center gap-2">
                <span
                  className="text-[11px] font-bold px-1.5 py-0.5 rounded border"
                  style={{ background: t.accentDim, color: t.accent, borderColor: t.accentBorder }}
                >
                  Python 3
                </span>
                <span className="text-xs font-mono" style={{ color: t.textSecondary }}>Codespace</span>
              </div>

              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={handleResetCode}
                  className="tk-ghost-btn"
                  style={{
                    background: "transparent",
                    border: `1px solid ${t.border}`,
                    color: t.textSecondary,
                    borderRadius: 6,
                    padding: "5px 12px",
                    fontSize: 11,
                    fontWeight: 600,
                    cursor: "pointer",
                    transition: "all 0.15s",
                  }}
                >
                  RESET
                </button>
                <button
                  type="button"
                  onClick={handleRunCode}
                  disabled={isConsoleRunning}
                  className="tk-ghost-btn"
                  style={{
                    background: "transparent",
                    border: `1px solid ${t.border}`,
                    color: t.textPrimary,
                    borderRadius: 6,
                    padding: "5px 12px",
                    fontSize: 11,
                    fontWeight: 600,
                    cursor: "pointer",
                    transition: "all 0.15s",
                    opacity: isConsoleRunning ? 0.5 : 1,
                  }}
                >
                  RUN
                </button>
                <button
                  type="button"
                  onClick={handleSubmitCode}
                  disabled={isConsoleRunning}
                  style={{
                    background: `linear-gradient(135deg, ${t.accent}, ${isLight ? "#047857" : "#059669"})`,
                    color: "#fff",
                    border: "none",
                    borderRadius: 6,
                    padding: "5px 14px",
                    fontSize: 11,
                    fontWeight: 700,
                    cursor: "pointer",
                    boxShadow: isLight ? "0 2px 8px rgba(5,150,105,0.25)" : "0 2px 8px rgba(16,185,129,0.2)",
                    transition: "all 0.15s",
                    opacity: isConsoleRunning ? 0.5 : 1,
                  }}
                >
                  SUBMIT
                </button>
              </div>
            </div>

            {/* Monaco Editor Frame */}
            <div className="flex-1 h-full w-full">
              <Editor
                height="100%"
                defaultLanguage="python"
                language="python"
                value={code}
                onChange={(value) => setCode(value || "")}
                theme={isLight ? "light" : "vs-dark"}
                options={{
                  minimap: { enabled: false },
                  fontSize: 13,
                  fontFamily: "'Fira Code', 'JetBrains Mono', Menlo, Monaco, Consolas, monospace",
                  automaticLayout: true,
                  scrollBeyondLastLine: false,
                  lineNumbers: "on",
                  wordWrap: "on",
                  tabSize: 4,
                  insertSpaces: true,
                }}
              />
            </div>
          </div>

          {/* CONSOLE / TEST RESULTS PANEL */}
          <div
            className="h-48 flex flex-col rounded-xl overflow-hidden"
            style={{ background: t.surface, border: `1px solid ${t.border}`, boxShadow: t.shadow }}
          >
            <div
              className="flex items-center px-4 py-2"
              style={{ background: isLight ? "rgba(5,150,105,0.04)" : "rgba(255,255,255,0.02)", borderBottom: `1px solid ${t.border}` }}
            >
              <span className="text-xs font-bold uppercase tracking-wider flex items-center gap-1.5" style={{ color: t.textSecondary }}>
                <span className="h-2 w-2 rounded-full animate-pulse" style={{ background: t.accent }} />
                Execution Results
              </span>
            </div>

            <div
              className="flex-1 overflow-y-auto p-4 font-mono text-xs leading-relaxed"
              style={{ background: isLight ? "#f8fafc" : "#0c1524" }}
            >
              <pre className="whitespace-pre-wrap font-sans" style={{ color: t.textPrimary }}>
                {consoleOutput}
              </pre>
            </div>
          </div>

        </section>
      </main>

      {/* FOOTER ERROR BAR */}
      {error && (
        <footer className="bg-red-955 border-t border-red-500/20 px-5 py-2 text-xs text-red-400">
          ⚠️ System error: {error}
        </footer>
      )}
      {/* DIALOG PORTAL */}
      {isEditModalOpen && createPortal(
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
            color: t.textPrimary,
            fontFamily: "'Sora', sans-serif"
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
                Edit Problem
              </h3>
              <button
                type="button"
                onClick={closeEditModal}
                className="tk-ghost-btn"
                style={{
                  background: "transparent", border: `1px solid ${t.border}`,
                  color: t.textSecondary, borderRadius: 5, padding: "4px 10px",
                  fontSize: 11, fontWeight: 600, cursor: "pointer",
                }}
              >
                Close
              </button>
            </div>

            {editError && (
              <div style={{
                marginBottom: 16, padding: "10px 14px", borderRadius: 8,
                background: isLight ? "#fff1f2" : "rgba(239,68,68,0.08)",
                border: `1px solid ${isLight ? "#fecdd3" : "rgba(239,68,68,0.2)"}`,
                color: isLight ? "#be123c" : "#f87171", fontSize: 12,
              }}>
                {editError}
              </div>
            )}

            <form onSubmit={handleEditSubmit} style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div>
                  <label style={labelStyle}>Name *</label>
                  <input
                    name="name"
                    required
                    value={formData.name}
                    onChange={handleFormChange}
                    className="tk-input"
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={labelStyle}>Source</label>
                  <input
                    name="source"
                    value={formData.source}
                    onChange={handleFormChange}
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
                  onChange={handleFormChange}
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
                    onChange={handleFormChange}
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
                    onChange={handleFormChange}
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
                    onChange={handleFormChange}
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
                    onChange={handleFormChange}
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
                  onChange={handleFormChange}
                  className="tk-input"
                  style={{ ...inputStyle, resize: "vertical", lineHeight: 1.6 }}
                />
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div>
                  <label style={labelStyle}>Input Zip File (.zip) (optional, overwrites manager)</label>
                  <input
                    type="file"
                    accept=".zip"
                    onChange={(e) => setInputZipFile(e.target.files[0])}
                    style={{ fontSize: 12, color: t.textSecondary }}
                  />
                </div>
                <div>
                  <label style={labelStyle}>Output Zip File (.zip) (optional, overwrites manager)</label>
                  <input
                    type="file"
                    accept=".zip"
                    onChange={(e) => setOutputZipFile(e.target.files[0])}
                    style={{ fontSize: 12, color: t.textSecondary }}
                  />
                </div>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 12, borderTop: `1px solid ${t.border}`, paddingTop: 14 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <label style={{ ...labelStyle, marginBottom: 0 }}>Test Cases Manager ({testcases.length})</label>
                  <button
                    type="button"
                    onClick={handleAddTestcase}
                    style={{
                      background: t.accentDim,
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

              <div style={{ display: "flex", justifyContent: "flex-end", gap: 10, paddingTop: 10 }}>
                <button
                  type="button"
                  onClick={closeEditModal}
                  className="tk-ghost-btn"
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
                    opacity: isSubmitting ? 0.6 : 1,
                  }}
                >
                  {isSubmitting ? "Processing..." : "Save Changes"}
                </button>
              </div>
            </form>
          </div>
        </div>,
        document.body
      )}

      {/* DELETE CONFIRMATION PORTAL */}
      {isDeleteConfirmOpen && createPortal(
        <div style={{
          position: "fixed", inset: 0, zIndex: 1300, display: "flex",
          alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.65)",
          padding: 16
        }}>
          <div style={{
            width: "100%", maxWidth: 400, background: t.surface,
            border: `1px solid ${isLight ? t.accentBorder : t.border}`,
            borderRadius: 12, padding: 20, boxShadow: "0 10px 25px rgba(0,0,0,0.3)",
            fontFamily: "'Sora', sans-serif", color: t.textPrimary,
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
              Are you sure you want to delete the problem <strong>{problem?.name}</strong> permanently? This action cannot be undone.
            </p>
            <div style={{ display: "flex", justifyContent: "flex-end", gap: 10 }}>
              <button
                type="button"
                onClick={() => setIsDeleteConfirmOpen(false)}
                className="tk-ghost-btn"
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