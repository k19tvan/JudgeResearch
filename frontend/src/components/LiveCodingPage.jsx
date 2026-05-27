// src/components/LiveCodingPage.jsx
import React, { useEffect, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import Editor from "@monaco-editor/react";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { fetchProblemContent } from "../api";

const CONTENT_TABS = [
  { key: "statement", label: "Description" },
  { key: "theory", label: "Theory" },
  { key: "tutorial", label: "Editorial" },
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

  const [activeContentTab, setActiveContentTab] = useState("statement");
  const [problem, setProblem] = useState(baseProblem || null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [code, setCode] = useState(normalizeEditorCode(baseProblem?.coding_markdown || ""));
  
  // Trạng thái Console giống LeetCode
  const [consoleOutput, setConsoleOutput] = useState("Run your code to see results here...");
  const [isConsoleRunning, setIsConsoleRunning] = useState(false);

  useEffect(() => {
    const loadProblemContent = async () => {
      setIsLoading(true);
      setError("");
      try {
        const result = await fetchProblemContent(problemId);
        const contentProblem = result?.data;
        if (contentProblem) {
          setProblem((prev) => ({
            ...(prev || {}),
            ...contentProblem,
          }));
          setCode(normalizeEditorCode(contentProblem.coding_markdown || ""));
        }
      } catch (err) {
        setError(err.message || "Failed to load problem content");
      } finally {
        setIsLoading(false);
      }
    };

    loadProblemContent();
  }, [problemId]);

  const handleRunCode = () => {
    setIsConsoleRunning(true);
    setConsoleOutput("Initializing Docker Sandbox...\nCompiling code...\nExecuting test cases...");
    setTimeout(() => {
      setConsoleOutput(
        "Status: SUCCESS\n---\nRunning test cases:\nTest Case 1: Passed\nTest Case 2: Passed\nAll checks completed successfully."
      );
      setIsConsoleRunning(false);
    }, 1500);
  };

  const handleSubmitCode = () => {
    setIsConsoleRunning(true);
    setConsoleOutput("Submitting code to ML Online Judge...\nEvaluating on hidden dataset...");
    setTimeout(() => {
      setConsoleOutput(
        "Submission Result: ACCEPTED\nTime: 124ms\nMemory: 24.5 MB\nScore: 100/100"
      );
      setIsConsoleRunning(false);
    }, 2000);
  };

  const contentMap = {
    statement: problem?.statement_markdown || "No statement content available for this problem.",
    theory: problem?.theory_markdown || "No theory content available for this problem.",
    tutorial: problem?.tutorial_markdown || "No tutorial content available for this problem.",
  };

  return (
    <div className="flex h-screen w-screen flex-col bg-[#0a0a14] text-slate-100 overflow-hidden select-none">
      <style>{`
        /* Custom scrollbar style */
        ::-webkit-scrollbar {
          width: 6px;
          height: 6px;
        }
        ::-webkit-scrollbar-track {
          background: rgba(15, 23, 42, 0.1);
        }
        ::-webkit-scrollbar-thumb {
          background: rgba(148, 163, 184, 0.2);
          border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
          background: rgba(148, 163, 184, 0.4);
        }

        /* Markdown Typography */
        .markdown-preview h1, .markdown-preview h2, .markdown-preview h3, .markdown-preview h4 {
          color: #f8fafc;
          font-weight: 700;
          margin-top: 1.25rem;
          margin-bottom: 0.5rem;
        }
        .markdown-preview h1 { font-size: 1.4rem; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 0.5rem;}
        .markdown-preview h2 { font-size: 1.2rem; }
        .markdown-preview h3 { font-size: 1.05rem; }
        .markdown-preview p { margin-bottom: 0.75rem; color: #cbd5e1; line-height: 1.6; }
        .markdown-preview ul, .markdown-preview ol {
          padding-left: 1.25rem;
          margin-bottom: 0.75rem;
          color: #cbd5e1;
        }
        .markdown-preview ul { list-style-type: disc; }
        .markdown-preview ol { list-style-type: decimal; }
        .markdown-preview li { margin-bottom: 0.25rem; }
        .markdown-preview code {
          background: rgba(15, 23, 42, 0.8);
          border: 1px solid rgba(148, 163, 184, 0.15);
          border-radius: 0.25rem;
          padding: 0.1rem 0.3rem;
          color: #38bdf8;
          font-family: monospace;
          font-size: 0.85em;
        }
        .markdown-preview pre {
          background: rgba(2, 6, 23, 0.8);
          border: 1px solid rgba(148, 163, 184, 0.1);
          border-radius: 0.5rem;
          padding: 0.75rem;
          margin: 0.75rem 0;
          overflow-x: auto;
        }
        .markdown-preview pre code {
          border: none;
          padding: 0;
          background: transparent;
          color: #e2e8f0;
          font-size: 0.9em;
        }
        .markdown-preview blockquote {
          border-left: 4px solid #0284c7;
          background: rgba(2, 132, 199, 0.05);
          padding: 0.5rem 0.75rem;
          margin: 0.75rem 0;
          border-radius: 0 0.375rem 0.375rem 0;
        }
        .markdown-preview a {
          color: #38bdf8;
          text-decoration: none;
        }
        .markdown-preview a:hover {
          text-decoration: underline;
        }
      `}</style>

      {/* HEADER BAR (LeetCode Slim Nav Style) */}
      <header className="flex h-12 w-full items-center justify-between border-b border-white/5 bg-[#0f0f1b] px-5">
        <div className="flex items-center gap-3">
          <span className="text-xl">🤖</span>
          <div className="h-4 w-[1px] bg-white/10" />
          <h1 className="text-sm font-bold text-slate-100 tracking-wide">
            {problem?.title || `Problem #${problemId}`}
          </h1>
          <span className="rounded bg-slate-800/80 px-2 py-0.5 text-[10px] font-semibold text-cyan-400">
            ML JUDGE
          </span>
        </div>
        <button
          type="button"
          onClick={() => navigate("/")}
          className="rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-1.5 text-xs font-semibold text-slate-300 hover:border-slate-600 transition"
        >
          BACK
        </button>
      </header>

      {/* WORKSPACE AREA (Grid 50/50, No Scroll on Main Body) */}
      <main className="flex flex-1 w-full overflow-hidden p-2 gap-2 bg-[#080711]">
        
        {/* LEFT COLUMN: Problem Info Panel (Description, Theory, Editorial) */}
        <section className="flex w-1/2 flex-col h-full rounded-xl border border-white/5 bg-slate-950/20 backdrop-blur-xl overflow-hidden">
          {/* Flat Tabs bar */}
          <nav className="flex border-b border-white/5 bg-slate-950/40">
            {CONTENT_TABS.map((tab) => (
              <button
                key={tab.key}
                type="button"
                onClick={() => setActiveContentTab(tab.key)}
                className={`px-4 py-3 text-xs font-semibold tracking-wide transition-all border-b-2 ${
                  activeContentTab === tab.key
                    ? "border-cyan-400 text-cyan-400 bg-cyan-500/5"
                    : "border-transparent text-slate-400 hover:text-slate-200"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>

          {/* Internal Scrollable Content Box */}
          <div className="flex-1 overflow-y-auto p-5 bg-slate-950/10">
            {isLoading ? (
              <p className="text-xs text-slate-400 animate-pulse">Loading workspace content...</p>
            ) : (
              <div className="markdown-preview text-sm text-slate-300">
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

        {/* RIGHT COLUMN: Code Editor (Top) & Output Console (Bottom) */}
        <section className="flex w-1/2 flex-col h-full gap-2 overflow-hidden">
          
          {/* EDITOR PANEL */}
          <div className="flex flex-1 flex-col rounded-xl border border-white/5 bg-slate-950/20 backdrop-blur-xl overflow-hidden">
            <div className="flex items-center justify-between border-b border-white/5 bg-slate-950/40 px-4 py-2">
              <div className="flex items-center gap-2">
                <span className="text-[11px] text-emerald-400 font-bold bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20">
                  Python 3
                </span>
                <span className="text-xs text-slate-400 font-mono">solution.py</span>
              </div>
              
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={handleRunCode}
                  disabled={isConsoleRunning}
                  className="rounded bg-slate-800 hover:bg-slate-700 px-3 py-1.5 text-xs font-semibold text-slate-300 border border-slate-700 hover:border-slate-600 transition disabled:opacity-50"
                >
                  RUN
                </button>
                <button
                  type="button"
                  onClick={handleSubmitCode}
                  disabled={isConsoleRunning}
                  className="rounded bg-cyan-600 hover:bg-cyan-500 px-3 py-1.5 text-xs font-semibold text-white shadow-lg shadow-cyan-500/10 transition disabled:opacity-50"
                >
                  SUBMIT
                </button>
              </div>
            </div>

            {/* Monaco Editor Frame */}
            <div className="flex-1 h-full w-full bg-[#0b1220]">
              <Editor
                height="100%"
                defaultLanguage="python"
                language="python"
                value={code}
                onChange={(value) => setCode(value || "")}
                theme="vs-dark"
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

          {/* CONSOLE / TEST RESULTS PANEL (LeetCode Bottom Drawer) */}
          <div className="h-48 flex flex-col rounded-xl border border-white/5 bg-slate-950/30 backdrop-blur-xl overflow-hidden">
            <div className="flex items-center border-b border-white/5 bg-slate-950/40 px-4 py-2">
              <span className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-cyan-400 animate-pulse" />
                Execution Results
              </span>
            </div>
            
            <div className="flex-1 overflow-y-auto bg-slate-950/60 p-4 font-mono text-xs leading-relaxed text-slate-300">
              <pre className="whitespace-pre-wrap font-sans text-slate-300">
                {consoleOutput}
              </pre>
            </div>
          </div>

        </section>
      </main>

      {/* FOOTER ERROR BAR */}
      {error && (
        <footer className="bg-red-950/80 border-t border-red-500/20 px-5 py-2 text-xs text-red-400">
          ⚠️ System error: {error}
        </footer>
      )}
    </div>
  );
}