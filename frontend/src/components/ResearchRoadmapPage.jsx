import React, { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams, useLocation } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import {
  fetchDraftSessionDetail,
  fetchRoadmap,
  finalizeDraftSession,
  updateDraftSessionFeedback,
  createProblemDetailedly,
} from "../api";

export default function ResearchRoadmapPage({ mode }) {
  const navigate = useNavigate();
  const location = useLocation();
  const { sessionId, roadmapId } = useParams();
  const previousPath = location.state?.previousPath || "/";
  const [draft, setDraft] = useState(null);
  const [roadmap, setRoadmap] = useState(null);
  const [proposedProblems, setProposedProblems] = useState([]);
  const [feedbackText, setFeedbackText] = useState("");
  const [roadmapTitle, setRoadmapTitle] = useState("");
  const [loading, setLoading] = useState("");
  const [error, setError] = useState("");
  const [creatingProblemId, setCreatingProblemId] = useState(null);
  const theme = localStorage.getItem("home_theme") || "dark";
  const isLight = theme === "light";

  const tone = useMemo(() => ({
    page: isLight ? "bg-[#eaf2f0] text-slate-900" : "bg-[#080C14] text-slate-100",
    panel: isLight ? "border-slate-300 bg-white/90" : "border-white/10 bg-slate-900/50",
    card: isLight ? "border-slate-300 bg-white" : "border-white/10 bg-slate-950/55",
    title: isLight ? "text-slate-900" : "text-white",
    body: isLight ? "text-slate-700" : "text-slate-300",
    muted: isLight ? "text-slate-600" : "text-slate-400",
    input: isLight
      ? "border-slate-300 bg-white text-slate-900 placeholder-slate-400"
      : "border-slate-800 bg-slate-950/70 text-slate-100 placeholder-slate-600",
  }), [isLight]);

  useEffect(() => {
    const load = async () => {
      setError("");
      setLoading("load");
      try {
        if (mode === "draft") {
          const result = await fetchDraftSessionDetail(sessionId);
          const data = result?.data;
          setDraft(data);
          setProposedProblems(data?.proposed_problems || []);
          setRoadmapTitle(data?.roadmap_name || "");
        } else {
          const result = await fetchRoadmap(roadmapId);
          setRoadmap(result?.data || null);
        }
      } catch (err) {
        setError(err.message || "Failed to load research roadmap");
      } finally {
        setLoading("");
      }
    };
    load();
  }, [mode, roadmapId, sessionId]);

  const handleFeedback = async () => {
    if (!draft?.session_id || !feedbackText.trim()) return;
    setError("");
    setLoading("feedback");
    try {
      const result = await updateDraftSessionFeedback({
        session_id: draft.session_id,
        feedback_text: feedbackText.trim(),
      });
      setProposedProblems(result?.data?.proposed_problems || []);
      setFeedbackText("");
    } catch (err) {
      setError(err.message || "Update proposed list failed");
    } finally {
      setLoading("");
    }
  };

  const handleSave = async () => {
    if (!draft?.session_id || !roadmapTitle.trim()) return;
    setError("");
    setLoading("save");
    try {
      const result = await finalizeDraftSession({
        session_id: draft.session_id,
        roadmap_title: roadmapTitle.trim(),
      });
      const saved = result?.data;
      if (saved?.roadmap_id) {
        navigate(`/research/roadmap/${saved.roadmap_id}`, { state: { previousPath: "/research" } });
      }
    } catch (err) {
      setError(err.message || "Save roadmap failed");
    } finally {
      setLoading("");
    }
  };

  const handleCreateDetailedly = async (problemId) => {
    setError("");
    setCreatingProblemId(problemId);
    try {
      const result = await createProblemDetailedly(problemId);
      if (result?.status === "success") {
        // Refresh roadmap data to update has_materials flag
        const refreshedRoadmap = await fetchRoadmap(roadmapId);
        setRoadmap(refreshedRoadmap?.data || null);
      }
    } catch (err) {
      setError(err.message || "Create problem materials failed");
    } finally {
      setCreatingProblemId(null);
    }
  };

  const handleGoToCoding = (problemId) => {
    navigate(`/livecoding/${problemId}`);
  };

  const currentTitle = mode === "draft"
    ? (draft?.roadmap_name || roadmapTitle || "Draft roadmap")
    : (roadmap?.name || "Roadmap");

  return (
    <div className={`min-h-screen w-screen ${tone.page}`}>
      <div className="mx-auto w-full max-w-[1500px] px-4 py-6 md:px-8">
        <header className={`rounded-lg border p-4 ${tone.panel}`}>
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div>
              <h1 className={`text-2xl font-bold ${tone.title}`}>{currentTitle}</h1>
              <p className={`mt-1 text-sm ${tone.muted}`}>
                {mode === "draft" ? draft?.repository_url : roadmap?.repository_url}
              </p>
            </div>
            <button
              type="button"
              onClick={() => navigate(previousPath)}
              className="rounded-lg border border-emerald-500/40 px-4 py-2 text-sm font-semibold text-emerald-500 hover:bg-emerald-500/10"
            >
              Back
            </button>
          </div>
        </header>

        {error && (
          <div className="mt-4 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-300">
            {error}
          </div>
        )}

        {loading === "load" ? (
          <p className={`mt-6 text-sm ${tone.muted}`}>Loading research roadmap...</p>
        ) : mode === "draft" ? (
          <div className="mt-5 grid gap-5 xl:grid-cols-[1fr_380px]">
            <section className={`rounded-lg border p-5 ${tone.panel}`}>
              <div className="flex items-center justify-between">
                <h2 className={`text-base font-semibold ${tone.title}`}>Proposed problems</h2>
                <span className={`text-xs ${tone.muted}`}>{proposedProblems.length} items</span>
              </div>
              <div className="mt-4 space-y-3">
                {proposedProblems.map((problem, index) => (
                  <article key={`${problem.title}-${index}`} className={`rounded-lg border p-4 ${tone.card}`}>
                    <div className="flex gap-3">
                      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-emerald-600 text-sm font-bold text-white">
                        {index + 1}
                      </div>
                  <div className="min-w-0">
                        <h3 className={`font-semibold ${tone.title}`}>{problem.title}</h3>
                        <div className={`mt-2 text-sm leading-6 ${tone.body} prose prose-invert max-w-none`}>
                          <ReactMarkdown>{problem.description}</ReactMarkdown>
                        </div>
                        <p className={`mt-2 text-xs ${tone.muted}`}>Target: {problem.target_module}</p>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            </section>

            <aside className="space-y-5">
              <section className={`rounded-lg border p-4 ${tone.panel}`}>
                <h2 className={`text-sm font-semibold ${tone.title}`}>Feedback</h2>
                <textarea
                  rows={6}
                  value={feedbackText}
                  onChange={(e) => setFeedbackText(e.target.value)}
                  placeholder="Remove problem 2 and rename problem 1..."
                  className={`mt-3 w-full resize-none rounded-lg border px-3 py-2 text-sm outline-none focus:border-emerald-500 ${tone.input}`}
                />
                <button
                  type="button"
                  onClick={handleFeedback}
                  disabled={loading === "feedback" || !feedbackText.trim()}
                  className="mt-3 w-full rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {loading === "feedback" ? "Updating..." : "Update proposed list"}
                </button>
              </section>

              <section className={`rounded-lg border p-4 ${tone.panel}`}>
                <h2 className={`text-sm font-semibold ${tone.title}`}>Save roadmap</h2>
                <input
                  value={roadmapTitle}
                  onChange={(e) => setRoadmapTitle(e.target.value)}
                  placeholder="Roadmap title"
                  className={`mt-3 w-full rounded-lg border px-3 py-2 text-sm outline-none focus:border-emerald-500 ${tone.input}`}
                />
                <button
                  type="button"
                  onClick={handleSave}
                  disabled={loading === "save" || !roadmapTitle.trim()}
                  className="mt-3 w-full rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {loading === "save" ? "Saving..." : "Save roadmap"}
                </button>
              </section>
            </aside>
          </div>
        ) : (
          <section className={`mt-5 rounded-lg border p-5 ${tone.panel}`}>
            <div className="flex items-center justify-between">
              <h2 className={`text-base font-semibold ${tone.title}`}>Roadmap timeline</h2>
              <span className={`text-xs ${tone.muted}`}>{roadmap?.problems?.length || 0} problems</span>
            </div>
            <div className="mt-4 space-y-3">
              {(roadmap?.problems || []).map((problem) => (
                <div key={problem.problem_id} className={`rounded-lg border p-4 ${tone.card}`}>
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className={`text-xs font-semibold ${tone.muted}`}>Step {problem.order_index}</p>
                      <h3 className={`font-semibold ${tone.title}`}>{problem.name}</h3>
                      <p className={`mt-1 text-xs ${tone.muted}`}>Problem #{problem.problem_id}</p>
                    </div>
                    <div className="flex gap-2">
                      {problem.has_materials ? (
                        <button
                          type="button"
                          onClick={() => handleGoToCoding(problem.problem_id)}
                          className="shrink-0 rounded-lg bg-blue-600 px-4 py-2 text-xs font-semibold text-white hover:bg-blue-500 transition-colors"
                        >
                          Go
                        </button>
                      ) : (
                        <button
                          type="button"
                          onClick={() => handleCreateDetailedly(problem.problem_id)}
                          disabled={creatingProblemId === problem.problem_id}
                          className={`shrink-0 rounded-lg px-3 py-2 text-xs font-semibold text-white transition-colors ${
                            creatingProblemId === problem.problem_id
                              ? "bg-emerald-400 cursor-wait"
                              : "bg-emerald-600 hover:bg-emerald-500"
                          }`}
                        >
                          {creatingProblemId === problem.problem_id ? "Creating..." : "Create Detailedly"}
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
