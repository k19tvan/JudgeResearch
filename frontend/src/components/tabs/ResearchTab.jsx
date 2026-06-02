import React, { useEffect, useState } from "react";
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
    num_test_cases: 3,
    user_id: localStorage.getItem("user_id") || DEFAULT_USER_ID,
  });

  const userId = Number(formData.user_id || DEFAULT_USER_ID);

  const refreshRoadmaps = async () => {
    if (!userId || userId <= 0) return;
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
    if (userId > 0) {
      localStorage.setItem("user_id", String(userId));
      refreshRoadmaps();
    }
  }, [userId]);

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === "number" ? Number(value) : value
    }));
  };

  const handleCreateRoadmap = async (e) => {
    e.preventDefault();
    setError("");
    setLoading("create");
    try {
      const result = await createProblemsFromRepo({
        ...formData,
        user_id: Number(formData.user_id),
        num_test_cases: Number(formData.num_test_cases)
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
    textPrimary:  "#0f172a",
    textSecondary:"#475569",
    textMuted:    "#94a3b8",
    inputBg:      "#ffffff",
    inputBorder:  "#cbd5e1",
    shadow:       "0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04)",
  } : {
    pageBg:       "transparent",
    surface:      "#0f172a",
    surfaceRaised:"#111827",
    border:       "rgba(255,255,255,0.07)",
    borderStrong: "rgba(255,255,255,0.12)",
    accent:       "#06b6d4",
    accentDark:   "#0891b2",
    accentBg:     "rgba(6,182,212,0.08)",
    accentBorder: "rgba(6,182,212,0.3)",
    textPrimary:  "#f1f5f9",
    textSecondary:"#64748b",
    textMuted:    "#475569",
    inputBg:      "#0c1524",
    inputBorder:  "rgba(255,255,255,0.08)",
    shadow:       "none",
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

  return (
    <div style={{ fontFamily: "'Inter var', 'Inter', sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');
        .tk-input:focus, .tk-select:focus {
          border-color: ${t.accent} !important;
          box-shadow: 0 0 0 3px ${t.accentBg} !important;
        }
        .tk-primary:hover { background: ${t.accentDark} !important; }
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
                <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
              </svg>
            </div>
            <h2 style={{
              margin: 0, fontSize: 18, fontWeight: 700,
              color: t.textPrimary, letterSpacing: "-0.02em",
            }}>
              Research Roadmaps
            </h2>
          </div>
          <p style={{ margin: 0, fontSize: 12, color: t.textSecondary, lineHeight: 1.5, paddingLeft: 40 }}>
            Create a research roadmap from a repository, then open any roadmap to refine or generate detailed problems.
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

      <div style={{ display: "grid", gridTemplateColumns: "420px 1fr", gap: 18, alignItems: "start" }}>
        {/* ── Left Form ── */}
        <form onSubmit={handleCreateRoadmap} style={{
          background: t.surface,
          border: `1px solid ${isLight ? t.accentBorder : t.border}`,
          borderRadius: 12, padding: 24,
          boxShadow: isLight ? "0 4px 12px rgba(5,150,105,0.1)" : "none",
          position: "relative", overflow: "hidden",
        }}>
          {/* Top stripe */}
          <div style={{
            position: "absolute", top: 0, left: 0, right: 0, height: 3,
            background: "linear-gradient(90deg, #10b981, #06b6d4, #818cf8)",
          }} />

          <h3 style={{
            margin: "0 0 20px", fontSize: 12, fontWeight: 700,
            color: isLight ? "#059669" : "#22d3ee",
            textTransform: "uppercase", letterSpacing: "0.1em",
          }}>
            Create Research Roadmap
          </h3>

          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div>
              <label style={labelStyle}>Roadmap Name</label>
              <input
                name="roadmap_name"
                value={formData.roadmap_name}
                onChange={handleChange}
                required
                placeholder="e.g. ResNet Roadmap"
                className="tk-input"
                style={inputStyle}
              />
            </div>

            <div>
              <label style={labelStyle}>GitHub Repository</label>
              <input
                name="repository_url"
                value={formData.repository_url}
                onChange={handleChange}
                required
                placeholder="e.g. https://github.com/KaimingHe/resnet"
                className="tk-input"
                style={inputStyle}
              />
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div>
                <label style={labelStyle}>Level</label>
                <select
                  name="level"
                  value={formData.level}
                  onChange={handleChange}
                  className="tk-select"
                  style={{ ...inputStyle, cursor: "pointer" }}
                >
                  <option>Beginner</option>
                  <option>Intermediate</option>
                  <option>Advanced</option>
                </select>
              </div>
              <div>
                <label style={labelStyle}>Framework</label>
                <input
                  name="framework"
                  value={formData.framework}
                  onChange={handleChange}
                  className="tk-input"
                  style={inputStyle}
                />
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div>
                <label style={labelStyle}>Test Cases</label>
                <input
                  name="num_test_cases"
                  type="number"
                  min="1"
                  max="10"
                  value={formData.num_test_cases}
                  onChange={handleChange}
                  className="tk-input"
                  style={inputStyle}
                />
              </div>
              <div>
                <label style={labelStyle}>User ID</label>
                <input
                  name="user_id"
                  type="number"
                  min="1"
                  value={formData.user_id}
                  onChange={handleChange}
                  className="tk-input"
                  style={inputStyle}
                />
              </div>
            </div>

            <div>
              <label style={labelStyle}>Additional Note</label>
              <textarea
                name="user_note"
                rows={4}
                value={formData.user_note}
                onChange={handleChange}
                placeholder="Focus on Residual Block; no need to rewrite train function..."
                className="tk-input"
                style={{ ...inputStyle, resize: "none", lineHeight: 1.6 }}
              />
            </div>

            <button
              type="submit"
              disabled={loading === "create"}
              className="tk-primary"
              style={{
                width: "100%", background: t.accent, border: "none", color: "#fff",
                borderRadius: 7, padding: "10px 20px",
                fontSize: 12, fontWeight: 600, cursor: "pointer",
                transition: "background 0.15s",
                boxShadow: isLight ? "0 1px 3px rgba(5,150,105,0.35)" : "none",
                opacity: loading === "create" ? 0.6 : 1,
              }}
            >
              {loading === "create" ? "Creating..." : "Create Proposed List"}
            </button>
          </div>
        </form>

        {/* ── Right Panel ── */}
        <div style={{
          background: t.surface,
          border: `1px solid ${t.border}`,
          borderRadius: 12, padding: 24,
          boxShadow: t.shadow,
        }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
            <div>
              <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: t.textPrimary }}>
                Existing Research Roadmaps
              </h3>
              <p style={{ margin: "2px 0 0", fontSize: 11, color: t.textMuted }}>
                Drafts and saved roadmaps for user #{userId}
              </p>
            </div>
            <span style={{ fontSize: 11, fontWeight: 700, color: t.textSecondary, background: isLight ? "#e2e8f0" : "rgba(255,255,255,0.07)", padding: "2px 8px", borderRadius: 10 }}>
              {draftSessions.length + roadmaps.length} items
            </span>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            {draftSessions.map((session) => (
              <button
                key={`draft-${session.id}`}
                type="button"
                onClick={() => navigate(`/research/draft/${session.id}`)}
                style={{
                  background: isLight ? "#fffbeb" : "rgba(245,158,11,0.04)",
                  border: `1px solid ${isLight ? "#fde68a" : "rgba(245,158,11,0.2)"}`,
                  borderRadius: 10, padding: 16, textAlign: "left", cursor: "pointer",
                  transition: "all 0.15s",
                }}
                onMouseEnter={e => e.currentTarget.style.borderColor = t.accent}
                onMouseLeave={e => e.currentTarget.style.borderColor = isLight ? "#fde68a" : "rgba(245,158,11,0.2)"}
              >
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8, gap: 8 }}>
                  <h4 style={{ margin: 0, fontSize: 13, fontWeight: 700, color: t.textPrimary, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                    {session.roadmap_name || "Untitled roadmap"}
                  </h4>
                  <span style={{ fontSize: 9, fontWeight: 700, textTransform: "uppercase", background: "rgba(245,158,11,0.12)", color: "#d97706", padding: "2px 6px", borderRadius: 4 }}>
                    Draft
                  </span>
                </div>
                <p style={{ margin: "0 0 10px", fontSize: 11, color: t.textSecondary, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                  {session.repository_url}
                </p>
                <p style={{ margin: 0, fontSize: 10, fontWeight: 600, color: "#d97706" }}>
                  Continue feedback or save
                </p>
              </button>
            ))}

            {roadmaps.map((roadmap) => (
              <button
                key={`roadmap-${roadmap.id}`}
                type="button"
                onClick={() => navigate(`/research/roadmap/${roadmap.id}`, { state: { previousPath: "/research" } })}
                style={{
                  background: isLight ? "#f8fafc" : "rgba(255,255,255,0.01)",
                  border: `1px solid ${t.border}`,
                  borderRadius: 10, padding: 16, textAlign: "left", cursor: "pointer",
                  transition: "all 0.15s",
                }}
                onMouseEnter={e => e.currentTarget.style.borderColor = t.accent}
                onMouseLeave={e => e.currentTarget.style.borderColor = t.border}
              >
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8, gap: 8 }}>
                  <h4 style={{ margin: 0, fontSize: 13, fontWeight: 700, color: t.textPrimary, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                    {roadmap.name}
                  </h4>
                  <span style={{ fontSize: 9, fontWeight: 700, textTransform: "uppercase", background: isLight ? "#ecfdf5" : "rgba(16,185,129,0.12)", color: isLight ? "#059669" : "#34d399", padding: "2px 6px", borderRadius: 4 }}>
                    Saved
                  </span>
                </div>
                <p style={{ margin: "0 0 10px", fontSize: 11, color: t.textSecondary, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                  {roadmap.repository_url}
                </p>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", fontSize: 10, color: t.textMuted, fontWeight: 600 }}>
                  <span>{roadmap.problem_count || 0} problems</span>
                  <span style={{ color: t.accent }}>Open ›</span>
                </div>
              </button>
            ))}

            {draftSessions.length + roadmaps.length === 0 && (
              <p style={{ margin: 0, fontSize: 12, color: t.textMuted, italic: true }}>
                No research roadmaps yet.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}