import React, { useEffect, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
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

  useEffect(() => {
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

    loadProblemContent();
  }, [problemId, currentUserId]);

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
    </div>
  );
}