import React, { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams, useLocation } from "react-router-dom";
import {
  fetchDraftSessionDetail,
  fetchRoadmap,
  finalizeDraftSession,
  updateDraftSessionFeedback,
  createProblemDetailedly,
  saveStepToProblem,
  deleteRoadmap,
  requestRoadmapApproval,
  approveRoadmap,
  rejectRoadmap,
  unpublishRoadmap,
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
  const [creatingStepId, setCreatingStepId] = useState(null);
  const [savingStepId, setSavingStepId] = useState(null);

  const theme = localStorage.getItem("home_theme") || "dark";
  const isLight = theme === "light";

  const userRole = localStorage.getItem("user_role") || "user";
  const [currentUserId, setCurrentUserId] = useState(Number(localStorage.getItem("user_id") || "0"));

  // =============== CƠ CHẾ TỰ CHỮA LÀNH LOCAL STORAGE ===============
  useEffect(() => {
    try {
      const token = localStorage.getItem("access_token");
      if (token) {
        const base64Url = token.split('.')[1];
        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        const jsonPayload = decodeURIComponent(window.atob(base64).split('').map(c => 
          '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2)
        ).join(''));
        const payload = JSON.parse(jsonPayload);
        
        if (payload.user_id && String(localStorage.getItem("user_id")) !== String(payload.user_id)) {
          console.log("[Auto-Heal] Khôi phục user_id chính xác:", payload.user_id);
          localStorage.setItem("user_id", String(payload.user_id));
          setCurrentUserId(Number(payload.user_id));
        }
      }
    } catch (e) {
      console.error("[Auto-Heal Error] Không thể giải mã token để phục hồi ID:", e);
    }
  }, []);

  const isOwnerOrAdmin = useMemo(() => {
    if (userRole === "admin") return true;
    if (roadmap && Number(roadmap.user_id) === currentUserId) return true;
    return false;
  }, [roadmap, userRole, currentUserId]);

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

  const loadRoadmapData = async () => {
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

  useEffect(() => {
    loadRoadmapData();
  }, [mode, roadmapId, sessionId]);

  // Polling tự động khi Draft Session đang sinh dữ liệu dưới nền
  useEffect(() => {
    let intervalId;
    if (mode === "draft" && draft?.status === "processing") {
      intervalId = setInterval(async () => {
        try {
          const result = await fetchDraftSessionDetail(sessionId);
          const data = result?.data;
          setDraft(data);
          setProposedProblems(data?.proposed_problems || []);
          setRoadmapTitle(data?.roadmap_name || "");
        } catch (err) {
          console.error("Lỗi polling nháp:", err);
        }
      }, 3000);
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [mode, sessionId, draft?.status]);

  // Polling tự động khi Roadmap Timeline có bước đang được xử lý ("generating")
  useEffect(() => {
    let intervalId;
    const hasGenerating = roadmap?.problems?.some(p => p.step_status === "generating");
    if (mode === "roadmap" && hasGenerating) {
      intervalId = setInterval(async () => {
        try {
          const result = await fetchRoadmap(roadmapId);
          if (result?.data) {
            setRoadmap(result.data);
          }
        } catch (err) {
          console.error("Lỗi polling timeline:", err);
        }
      }, 3000);
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [mode, roadmapId, roadmap?.problems]);

  // Chỉnh sửa trực tiếp danh sách Proposed Problems
  const handleEditProposedField = (index, field, value) => {
    const updated = [...proposedProblems];
    updated[index] = { ...updated[index], [field]: value };
    setProposedProblems(updated);
  };

  const handleDeleteProposedItem = (index) => {
    const updated = proposedProblems.filter((_, i) => i !== index);
    setProposedProblems(updated);
  };

  // Lưu các chỉnh sửa thủ công trực tiếp lên Backend
  const handleSaveEdits = async () => {
    setLoading("save_edits");
    setError("");
    try {
      const token = localStorage.getItem("access_token");
      const response = await fetch(`http://localhost:21081/api/problems/draft_sessions/${sessionId}`, {        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify({ problems: proposedProblems })
      });
      const resData = await response.json();
      if (!response.ok) throw new Error(resData.detail || "Failed to save edits");
      alert("Proposed problems edited successfully!");
    } catch (err) {
      setError(err.message || "Failed to save edits");
    } finally {
      setLoading("");
    }
  };

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

  const handleCreateDetailedly = async (stepId) => {
    setError("");
    setCreatingStepId(stepId);
    try {
      const result = await createProblemDetailedly(stepId);
      if (result?.status === "success") {
        await loadRoadmapData();
      }
    } catch (err) {
      setError(err.message || "Create problem materials failed");
    } finally {
      setCreatingStepId(null);
    }
  };

  const handleGenerateAll = async () => {
    if (!roadmap?.id) return;
    setLoading("generate_all");
    setError("");
    try {
      const token = localStorage.getItem("access_token");
      const response = await fetch(`http://localhost:21081/api/roadmaps/${roadmap.id}/generate_all`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`
        }
      });
      const resData = await response.json();
      if (!response.ok) throw new Error(resData.detail || "Batch generation failed");
      await loadRoadmapData();
      alert("All pending steps are queued for AI generation.");
    } catch (err) {
      setError(err.message || "Failed to schedule batch generation");
    } finally {
      setLoading("");
    }
  };

  const handleSaveToProblem = async (stepId) => {
    setError("");
    setSavingStepId(stepId);
    try {
      const result = await saveStepToProblem(stepId);
      if (result?.status === "success") {
        await loadRoadmapData();
      }
    } catch (err) {
      setError(err.message || "Failed to save problem to system");
    } finally {
      setSavingStepId(null);
    }
  };

  // CHUYỂN HƯỚNG TRỰC TIẾP LÊN LIVE CODING SPACE Ở CHẾ ĐỘ PREVIEW DRAFT
  const handleOpenPreview = (stepId) => {
    navigate(`/livecoding/draft/${stepId}`);
  };

  const handlePublishOrApproval = async () => {
    if (!roadmap?.id) return;
    setLoading("publish");
    setError("");
    try {
      if (userRole === "admin") {
        await publishRoadmapDirectly(roadmap.id);
      } else {
        await requestRoadmapApproval(roadmap.id);
      }
      await loadRoadmapData();
    } catch (err) {
      setError(err.message || "Failed to update roadmap visibility");
    } finally {
      setLoading("");
    }
  };

  const handleAdminApprove = async () => {
    if (!roadmap?.id) return;
    setLoading("publish");
    setError("");
    try {
      await approveRoadmap(roadmap.id);
      await loadRoadmapData();
    } catch (err) {
      setError(err.message || "Failed to approve roadmap");
    } finally {
      setLoading("");
    }
  };

  const handleAdminReject = async () => {
    if (!roadmap?.id) return;
    setLoading("publish");
    setError("");
    try {
      await rejectRoadmap(roadmap.id);
      await loadRoadmapData();
    } catch (err) {
      setError(err.message || "Failed to reject roadmap");
    } finally {
      setLoading("");
    }
  };

  const handleUnpublish = async () => {
    if (!roadmap?.id) return;
    if (!window.confirm("Are you sure you want to unpublish this roadmap? Normal students will not be able to view it anymore.")) return;
    setLoading("publish");
    setError("");
    try {
      await unpublishRoadmap(roadmap.id);
      await loadRoadmapData();
    } catch (err) {
      setError(err.message || "Failed to unpublish roadmap");
    } finally {
      setLoading("");
    }
  };

  const handleDeleteRoadmap = async () => {
    if (!roadmap?.id) return;
    const confirmMessage = "Hành động này sẽ xóa bỏ hoàn toàn cấu trúc lộ trình cùng các tư liệu nháp chưa lưu. Các bài tập đã chuyển đổi thành công (Saved) sang kho chung sẽ được giữ lại độc lập.\n\nBạn có chắc chắn muốn xóa vĩnh viễn lộ trình này?";
    if (!window.confirm(confirmMessage)) return;
    setLoading("delete");
    setError("");
    try {
      await deleteRoadmap(roadmap.id);
      navigate(previousPath);
    } catch (err) {
      setError(err.message || "Failed to delete roadmap");
      setLoading("");
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
              <div className="flex items-center gap-3">
                <h1 className={`text-2xl font-bold ${tone.title}`}>{currentTitle}</h1>
                {mode === "roadmap" && roadmap && (
                  <span className={`rounded px-2.5 py-0.5 text-xs font-semibold uppercase ${
                    roadmap.status === "public"
                      ? "bg-emerald-500/15 text-emerald-500"
                      : roadmap.status === "pending"
                      ? "bg-amber-500/15 text-amber-500"
                      : "bg-slate-500/15 text-slate-400"
                  }`}>
                    {roadmap.status === "public" ? "Public" : roadmap.status === "pending" ? "Pending Approval" : "Private (Draft)"}
                  </span>
                )}
              </div>
              <p className={`mt-1 text-sm ${tone.muted}`}>
                {mode === "draft" ? draft?.repository_url : roadmap?.repository_url}
              </p>
            </div>
            <div className="flex items-center gap-2 flex-wrap">
              {mode === "roadmap" && roadmap && isOwnerOrAdmin && (
                <>
                  {roadmap.problems?.some(p => p.step_status === "pending" || p.step_status === "failed") && (
                    <button
                      type="button"
                      onClick={handleGenerateAll}
                      disabled={loading === "generate_all"}
                      className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-500 transition-colors"
                    >
                      {loading === "generate_all" ? "Queuing Steps..." : "Generate All Pending Steps"}
                    </button>
                  )}
                  {roadmap.status === "draft" && (
                    <button
                      type="button"
                      onClick={handlePublishOrApproval}
                      disabled={loading === "publish"}
                      className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-500 transition-colors"
                    >
                      {userRole === "admin" ? "Publish Directly" : "Publish to Public"}
                    </button>
                  )}
                  {roadmap.status === "pending" && userRole === "admin" && (
                    <>
                      <button
                        type="button"
                        onClick={handleAdminApprove}
                        disabled={loading === "publish"}
                        className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-500 transition-colors"
                      >
                        Approve
                      </button>
                      <button
                        type="button"
                        onClick={handleAdminReject}
                        disabled={loading === "publish"}
                        className="rounded-lg bg-rose-600 px-4 py-2 text-sm font-semibold text-white hover:bg-rose-500 transition-colors"
                      >
                        Reject
                      </button>
                    </>
                  )}
                  {roadmap.status === "public" && (
                    <button
                      type="button"
                      onClick={handleUnpublish}
                      disabled={loading === "publish"}
                      className="rounded-lg bg-slate-700 px-4 py-2 text-sm font-semibold text-slate-300 hover:bg-slate-600 transition-colors"
                    >
                      Unpublish
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={handleDeleteRoadmap}
                    disabled={loading === "delete"}
                    className="rounded-lg bg-rose-600/10 border border-rose-500/20 px-3 py-2 text-sm font-semibold text-rose-400 hover:bg-rose-600/20 transition-colors"
                  >
                    Delete Roadmap
                  </button>
                </>
              )}
              <button
                type="button"
                onClick={() => navigate("/", { state: { activeTab: "RESEARCH" } })}
                className="rounded-lg border border-emerald-500/40 px-4 py-2 text-sm font-semibold text-emerald-500 hover:bg-emerald-500/10 transition-colors"
              >
                Back
              </button>
            </div>
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
          draft?.status === "processing" ? (
            <div className="mt-8 flex flex-col items-center justify-center p-12 rounded-lg border border-dashed border-emerald-500/30 bg-emerald-500/5 text-center">
              <div className="h-10 w-10 animate-spin rounded-full border-4 border-emerald-500 border-t-transparent mb-4 animate-spin"></div>
              <h3 className={`text-lg font-semibold ${tone.title}`}>AI is analyzing the repository...</h3>
              <p className={`mt-2 text-sm max-w-md ${tone.muted}`}>
                DeepWiki is investigating the repository structure and drafting the optimal steps. 
                This process typically takes 30 to 60 seconds. Please hold on!
              </p>
            </div>
          ) : draft?.status === "failed" ? (
            <div className="mt-6 rounded-lg border border-red-500/30 bg-red-500/10 p-5 text-red-300">
              <h3 className="font-bold text-lg mb-2">Generation Failed</h3>
              <p className="text-sm font-mono whitespace-pre-wrap">{draft?.error_message || "Unknown AI error"}</p>
              <button 
                onClick={() => navigate("/")} 
                className="mt-4 rounded bg-red-600 px-4 py-2 text-xs font-semibold text-white hover:bg-red-500"
              >
                Back to Dashboard
              </button>
            </div>
          ) : (
            <div className="mt-5 grid gap-5 xl:grid-cols-[1fr_380px]">
              <section className={`rounded-lg border p-5 ${tone.panel}`}>
                <div className="flex items-center justify-between">
                  <h2 className={`text-base font-semibold ${tone.title}`}>Proposed problems</h2>
                  <div className="flex items-center gap-3">
                    <button
                      type="button"
                      onClick={handleSaveEdits}
                      disabled={loading === "save_edits"}
                      className="rounded bg-emerald-600 hover:bg-emerald-500 text-xs font-semibold text-white px-3 py-1"
                    >
                      {loading === "save_edits" ? "Saving..." : "Save Edits Locally"}
                    </button>
                    <span className={`text-xs ${tone.muted}`}>{proposedProblems.length} items</span>
                  </div>
                </div>
                <div className="mt-4 space-y-3">
                  {proposedProblems.map((problem, index) => (
                    <article key={`${problem.title}-${index}`} className={`rounded-lg border p-4 ${tone.card}`}>
                      <div className="flex gap-3">
                        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-emerald-600 text-sm font-bold text-white">
                          {index + 1}
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center justify-between gap-2">
                            <input
                              type="text"
                              value={problem.title}
                              onChange={(e) => handleEditProposedField(index, "title", e.target.value)}
                              className={`w-full font-semibold bg-transparent border-b border-transparent hover:border-slate-500 focus:border-emerald-500 outline-none ${tone.title}`}
                            />
                            <button
                              type="button"
                              onClick={() => handleDeleteProposedItem(index)}
                              className="text-rose-500 hover:text-rose-400 text-xs"
                            >
                              Delete
                            </button>
                          </div>
                          <textarea
                            rows={3}
                            value={problem.description}
                            onChange={(e) => handleEditProposedField(index, "description", e.target.value)}
                            className={`mt-2 w-full text-sm leading-6 bg-transparent border border-transparent hover:border-slate-500 focus:border-emerald-500 outline-none rounded p-1 resize-none ${tone.body}`}
                          />
                          <div className="mt-2 flex items-center gap-1.5">
                            <span className={`text-xs font-semibold ${tone.muted}`}>Target: </span>
                            <input
                              type="text"
                              value={problem.target_module}
                              onChange={(e) => handleEditProposedField(index, "target_module", e.target.value)}
                              className={`text-xs bg-transparent border-b border-transparent hover:border-slate-500 focus:border-emerald-500 outline-none ${tone.body}`}
                            />
                          </div>
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
          )
        ) : (
          <section className={`mt-5 rounded-lg border p-5 ${tone.panel}`}>
            <div className="flex items-center justify-between">
              <h2 className={`text-base font-semibold ${tone.title}`}>Roadmap timeline</h2>
              <span className={`text-xs ${tone.muted}`}>{roadmap?.problems?.length || 0} steps</span>
            </div>
            <div className="mt-4 space-y-3">
              {(roadmap?.problems || []).map((problem) => {
                const stepId = problem.step_id;
                const stepStatus = problem.step_status;
                const errMsg = problem.error_message;
                
                return (
                  <div key={stepId} className={`rounded-lg border p-4 ${tone.card}`}>
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className={`text-xs font-semibold ${tone.muted}`}>Step {problem.order_index}</p>
                        <h3 className={`font-semibold ${tone.title}`}>{problem.name}</h3>
                        
                        {stepStatus === 'saved' && (
                          <p className={`mt-1 text-xs ${tone.muted}`}>Problem ID: #{problem.problem_id}</p>
                        )}
                        {stepStatus === 'generating' && (
                          <div className="mt-1.5 flex items-center gap-2 text-amber-500">
                            <span className="h-3 w-3 animate-spin rounded-full border-2 border-amber-500 border-t-transparent animate-spin"></span>
                            <span className="text-[10px] font-semibold uppercase">Generating Detailed Materials...</span>
                          </div>
                        )}
                        {stepStatus === 'generated' && (
                          <span className="inline-block mt-1.5 rounded bg-amber-500/15 px-2 py-0.5 text-[10px] font-semibold text-amber-500">
                            Draft Materials Generated
                          </span>
                        )}
                        {stepStatus === 'pending' && (
                          <span className="inline-block mt-1.5 rounded bg-slate-500/15 px-2 py-0.5 text-[10px] font-semibold text-slate-400">
                            Pending AI Generation
                          </span>
                        )}
                        {stepStatus === 'failed' && (
                          <div className="mt-1.5 rounded bg-rose-500/10 border border-rose-500/20 p-2 text-xs text-rose-400 max-w-xl">
                            <p className="font-bold">Generation Failed:</p>
                            <p className="font-mono mt-1 text-[11px] whitespace-pre-wrap">{errMsg || "Unknown compilation error"}</p>
                          </div>
                        )}
                      </div>
                      
                      <div className="flex gap-2 shrink-0">
                        {stepStatus === "saved" && (
                          <button
                            type="button"
                            onClick={() => handleGoToCoding(problem.problem_id)}
                            className="shrink-0 rounded-lg bg-blue-600 px-4 py-2 text-xs font-semibold text-white hover:bg-blue-500 transition-colors"
                          >
                            Go
                          </button>
                        )}
                        
                        {isOwnerOrAdmin && (stepStatus === "pending" || stepStatus === "failed") && (
                          <button
                            type="button"
                            onClick={() => handleCreateDetailedly(stepId)}
                            disabled={creatingStepId === stepId}
                            className={`shrink-0 rounded-lg px-3 py-2 text-xs font-semibold text-white transition-colors ${
                              creatingStepId === stepId
                                ? "bg-emerald-400 cursor-wait"
                                : "bg-emerald-600 hover:bg-emerald-500"
                            }`}
                          >
                            {creatingStepId === stepId ? "Generating..." : "Create Detailedly"}
                          </button>
                        )}
                        
                        {isOwnerOrAdmin && stepStatus === "generated" && (
                          <div className="flex gap-2">
                            <button
                              type="button"
                              onClick={() => handleCreateDetailedly(stepId)}
                              disabled={creatingStepId === stepId}
                              className="shrink-0 rounded-lg bg-slate-700 hover:bg-slate-600 px-3 py-2 text-xs font-semibold text-slate-300 transition-colors"
                            >
                              {creatingStepId === stepId ? "Regenerating..." : "Regenerate"}
                            </button>
                            <button
                              type="button"
                              onClick={() => handleOpenPreview(stepId)}
                              className="shrink-0 rounded-lg bg-blue-600 hover:bg-blue-500 px-3 py-2 text-xs font-semibold text-white transition-colors"
                            >
                              Preview Draft
                            </button>
                            <button
                              type="button"
                              onClick={() => handleSaveToProblem(stepId)}
                              disabled={savingStepId === stepId}
                              className="shrink-0 rounded-lg bg-emerald-600 hover:bg-emerald-500 px-3 py-2 text-xs font-semibold text-white transition-colors"
                            >
                              {savingStepId === stepId ? "Saving..." : "Save to Problem"}
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}