import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  createProblemsFromRepo,
  fetchDraftSessions,
  fetchRoadmaps,
} from "../../api";

const DEFAULT_USER_ID = "1";

export default function ResearchTab({ isLight = false }) {
  const navigate = useNavigate();
  const [draftSessions, setDraftSessions] = useState([]);
  const [roadmaps, setRoadmaps] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState("");
  const [formData, setFormData] = useState({
    roadmap_name: "",
    repository_url: "",
    level: "Intermediate",
    user_note: "",
    framework: "PyTorch",
    user_id: localStorage.getItem("user_id") || DEFAULT_USER_ID,
  });

  const tone = useMemo(() => ({
    title: isLight ? "text-slate-900" : "text-white",
    body: isLight ? "text-slate-700" : "text-slate-300",
    muted: isLight ? "text-slate-600" : "text-slate-400",
    panel: isLight ? "border-slate-300 bg-white/90" : "border-white/10 bg-slate-900/40",
    input: isLight
      ? "border-slate-300 bg-white text-slate-900 placeholder-slate-400"
      : "border-slate-800 bg-slate-950/70 text-slate-100 placeholder-slate-600",
    card: isLight ? "border-slate-300 bg-white" : "border-white/10 bg-slate-950/45",
  }), [isLight]);

  const userId = Number(formData.user_id || DEFAULT_USER_ID);

  const refreshRoadmaps = async () => {
    try {
      const [draftResult, roadmapResult] = await Promise.all([
        fetchDraftSessions(userId),
        fetchRoadmaps(userId),
      ]);
      setDraftSessions(draftResult?.data || []);
      setRoadmaps(roadmapResult?.data || []);
    } catch (err) {
      setError(err.message || "Failed to load roadmaps");
    }
  };

  useEffect(() => {
    refreshRoadmaps();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId]);

  const handleChange = (e) => {
    setFormData((prev) => ({ ...prev, [e.target.name]: e.target.value }));
  };

  const handleCreateRoadmap = async (e) => {
    e.preventDefault();
    setError("");
    setLoading("create");
    try {
      const result = await createProblemsFromRepo({
        ...formData,
        user_id: Number(formData.user_id),
      });
      const sessionId = result?.data?.session_id;
      await refreshRoadmaps();
      if (sessionId) {
        navigate(`/research/draft/${sessionId}`);
      }
    } catch (err) {
      setError(err.message || "Create roadmap failed");
    } finally {
      setLoading("");
    }
  };

  return (
    <section className="space-y-5">
      <div>
        <h2 className={`text-2xl font-bold tracking-wide ${tone.title}`}>RESEARCH</h2>
        <p className={`mt-2 text-sm ${tone.muted}`}>
          Create a research roadmap from a repository, then open any roadmap to refine or generate detailed problems.
        </p>
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-300">
          {error}
        </div>
      )}

      <div className="grid gap-5 xl:grid-cols-[420px_1fr]">
        <form onSubmit={handleCreateRoadmap} className={`rounded-lg border p-5 ${tone.panel}`}>
          <h3 className={`text-base font-semibold ${tone.title}`}>Create research roadmap</h3>
          <div className="mt-4 space-y-3">
            <label className={`block text-xs font-semibold uppercase ${tone.muted}`}>
              Roadmap name
              <input
                name="roadmap_name"
                value={formData.roadmap_name}
                onChange={handleChange}
                required
                placeholder="ResNet roadmap"
                className={`mt-1.5 w-full rounded-lg border px-3 py-2 text-sm outline-none focus:border-emerald-500 ${tone.input}`}
              />
            </label>

            <label className={`block text-xs font-semibold uppercase ${tone.muted}`}>
              GitHub repository
              <input
                name="repository_url"
                value={formData.repository_url}
                onChange={handleChange}
                required
                placeholder="https://github.com/KaimingHe/resnet"
                className={`mt-1.5 w-full rounded-lg border px-3 py-2 text-sm outline-none focus:border-emerald-500 ${tone.input}`}
              />
            </label>

            <div className="grid gap-3 sm:grid-cols-2">
              <label className={`block text-xs font-semibold uppercase ${tone.muted}`}>
                Level
                <select
                  name="level"
                  value={formData.level}
                  onChange={handleChange}
                  className={`mt-1.5 w-full rounded-lg border px-3 py-2 text-sm outline-none focus:border-emerald-500 ${tone.input}`}
                >
                  <option>Beginner</option>
                  <option>Intermediate</option>
                  <option>Advanced</option>
                </select>
              </label>
              <label className={`block text-xs font-semibold uppercase ${tone.muted}`}>
                Framework
                <input
                  name="framework"
                  value={formData.framework}
                  onChange={handleChange}
                  className={`mt-1.5 w-full rounded-lg border px-3 py-2 text-sm outline-none focus:border-emerald-500 ${tone.input}`}
                />
              </label>
            </div>

            <label className={`block text-xs font-semibold uppercase ${tone.muted}`}>
              User ID
              <input
                name="user_id"
                type="number"
                min="1"
                value={formData.user_id}
                onChange={handleChange}
                className={`mt-1.5 w-full rounded-lg border px-3 py-2 text-sm outline-none focus:border-emerald-500 ${tone.input}`}
              />
            </label>

            <label className={`block text-xs font-semibold uppercase ${tone.muted}`}>
              Additional note
              <textarea
                name="user_note"
                rows={5}
                value={formData.user_note}
                onChange={handleChange}
                placeholder="Focus on Residual Block; no need to rewrite train function"
                className={`mt-1.5 w-full resize-none rounded-lg border px-3 py-2 text-sm outline-none focus:border-emerald-500 ${tone.input}`}
              />
            </label>
          </div>

          <button
            type="submit"
            disabled={loading === "create"}
            className="mt-4 w-full rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading === "create" ? "Creating..." : "Create proposed list"}
          </button>
        </form>

        <div className={`rounded-lg border p-5 ${tone.panel}`}>
          <div className="flex items-center justify-between gap-3">
            <div>
              <h3 className={`text-base font-semibold ${tone.title}`}>Existing research roadmaps</h3>
              <p className={`mt-1 text-xs ${tone.muted}`}>Drafts and saved roadmaps for user #{userId}</p>
            </div>
            <span className={`text-xs ${tone.muted}`}>{draftSessions.length + roadmaps.length} items</span>
          </div>

          <div className="mt-4 grid gap-3 lg:grid-cols-2">
            {draftSessions.map((session) => (
              <button
                key={`draft-${session.id}`}
                type="button"
                onClick={() => navigate(`/research/draft/${session.id}`)}
                className={`rounded-lg border p-4 text-left transition ${
                  isLight
                    ? "border-amber-300 bg-amber-50 hover:border-emerald-500"
                    : "border-amber-500/25 bg-amber-500/10 hover:border-emerald-500/60"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <h4 className={`font-semibold ${tone.title}`}>{session.roadmap_name || "Untitled roadmap"}</h4>
                  <span className="rounded-full bg-amber-500/20 px-2 py-0.5 text-[11px] font-semibold text-amber-500">Draft</span>
                </div>
                <p className={`mt-2 truncate text-xs ${tone.muted}`}>{session.repository_url}</p>
                <p className={`mt-3 text-xs ${tone.muted}`}>Continue feedback or save</p>
              </button>
            ))}

            {roadmaps.map((roadmap) => (
              <button
                key={`roadmap-${roadmap.id}`}
                type="button"
                onClick={() => navigate(`/research/roadmap/${roadmap.id}`, { state: { previousPath: "/research" } })}
                className={`rounded-lg border p-4 text-left transition ${
                  isLight
                    ? "border-slate-300 bg-white hover:border-emerald-500"
                    : "border-white/10 bg-slate-950/45 hover:border-emerald-500/60"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <h4 className={`font-semibold ${tone.title}`}>{roadmap.name}</h4>
                  <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-[11px] font-semibold text-emerald-500">Saved</span>
                </div>
                <p className={`mt-2 truncate text-xs ${tone.muted}`}>{roadmap.repository_url}</p>
                <div className={`mt-3 flex items-center justify-between text-xs ${tone.muted}`}>
                  <span>{roadmap.problem_count || 0} problems</span>
                  <span>Open</span>
                </div>
              </button>
            ))}

            {draftSessions.length + roadmaps.length === 0 && (
              <p className={`text-sm ${tone.muted}`}>No research roadmaps yet.</p>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
