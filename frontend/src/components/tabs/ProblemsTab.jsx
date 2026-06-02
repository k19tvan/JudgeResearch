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
    resetForm();
    setError("");
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
        setIsModalOpen(true);
      } else {
        alert(result.detail || "Failed to load problem content");
      }
    } catch (err) {
      alert("Error loading problem content: " + err.message);
    }
  };

  const handleDeleteProblemDirect = async (e, problemId) => {
    e.stopPropagation();
    const confirmDel = window.confirm("Are you sure you want to delete this problem permanently?");
    if (confirmDel) {
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
        const response = await fetch(`http://localhost:21081/api/problems/${editingProblemId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: currentUserId,
            name: formData.name,
            source: formData.source,
            statement_markdown: formData.statement_markdown,
            theory_markdown: formData.theory_markdown,
            tutorial_markdown: formData.tutorial_markdown,
            solution_markdown: formData.solution_markdown,
            coding_markdown: formData.coding_markdown,
            checker_markdown: formData.checker_markdown
          })
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

  const renderStatusIndicator = (problem) => {
    const score = problem.best_score;
    const status = problem.best_status;

    if (score === null || score === undefined) {
      return (
        <div 
          className={`h-2 w-2 rounded-full border ${
            isLight ? "border-slate-300 bg-slate-100" : "border-slate-700 bg-slate-900"
          }`} 
          title="Not attempted"
        />
      );
    }
    
    if (score === 100 || status === "accepted") {
      return (
        <div 
          className="h-2 w-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" 
          title="Solved (100 pts)"
        />
      );
    }

    if (score > 0 && score < 100) {
      return (
        <div 
          className="h-2 w-2 rounded-full bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.5)]" 
          title={`Partial Solved (${score} pts)`}
        />
      );
    }

    return (
      <div 
        className="h-2 w-2 rounded-full bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.5)]" 
        title="Attempted / Failed"
      />
    );
  };

  return (
    <section className="space-y-6">
      {/* HEADER BAR */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-4">
          <h2 className={`text-2xl font-bold tracking-wide ${isLight ? "text-slate-800" : "text-white"}`}>PROBLEMS</h2>
          <select
            value={filterMode}
            onChange={(e) => setFilterMode(e.target.value)}
            className={`rounded-lg border px-3 py-1.5 text-xs outline-none transition font-semibold shadow-sm ${
              isLight 
                ? "border-slate-200 bg-white text-slate-700 hover:border-slate-300" 
                : "border-white/5 bg-slate-900 text-slate-300 focus:border-cyan-500 hover:border-white/10"
            }`}
          >
            <option value="public">Public</option>
            <option value="private">My Problems</option> 
            <option value="all">All</option>
          </select>
        </div>

        {(role === "admin" || role === "contributor") && (
          <button
            type="button"
            onClick={() => { resetForm(); setIsModalOpen(true); }}
            className={`rounded-lg px-4 py-2.5 text-xs font-bold tracking-wider text-white transition-all duration-200 active:scale-[0.98] ${
              isLight
                ? "bg-emerald-600 hover:bg-emerald-500 shadow-md shadow-emerald-600/10 hover:shadow-emerald-600/20"
                : "bg-cyan-600 hover:bg-cyan-500 shadow-md shadow-cyan-600/10 hover:shadow-cyan-600/20"
            }`}
          >
            CREATE PROBLEM
          </button>
        )}
      </div>

      {error && !isModalOpen && (
        <div className="rounded-lg border border-red-500/20 bg-red-500/5 p-3.5 text-xs text-red-400">
          {error}
        </div>
      )}

      {/* PROBLEMS TABLE */}
      {isLoading ? (
        <p className={`text-xs ${isLight ? "text-slate-500" : "text-slate-400"} animate-pulse`}>Loading problems workspace...</p>
      ) : problems.length === 0 ? (
        <p className={`text-xs ${isLight ? "text-slate-500" : "text-slate-400"}`}>No created problems yet.</p>
      ) : (
        <div className={`overflow-x-auto rounded-xl border ${
          isLight ? "border-slate-200/80 bg-white shadow-sm" : "border-white/5 bg-slate-950/40"
        }`}>
          <table className="w-full text-left border-collapse min-w-[950px]">
            <thead>
              <tr className={`text-xs font-bold uppercase tracking-wider border-b ${
                isLight 
                  ? "bg-emerald-600 text-white border-emerald-700/50" 
                  : "bg-emerald-950/60 text-emerald-200 border-emerald-500/20"
              }`}>
                <th className="w-12 py-4 text-center">✔</th>
                <th className="px-4 py-4 w-44">ID</th>
                <th className="px-4 py-4">Problem</th>
                <th className="px-4 py-4 w-40">Category</th>
                <th className="px-4 py-4 w-24 text-center">Points</th>
                <th className="px-4 py-4 w-28 text-center">Visibility</th>
                <th className="px-4 py-4 w-36 text-center">Request</th>
                <th className="px-4 py-4 w-36 text-center">Actions</th>
                <th className="px-4 py-4 w-32 text-center">Status</th>
              </tr>
            </thead>

            <tbody className={`divide-y ${isLight ? "divide-slate-100" : "divide-white/5"}`}>
              {problems.map((problem, index) => {
                const isPrivate = problem.is_public === 0;
                const reqStatus = problem.request_status ? problem.request_status.toUpperCase() : "NONE";
                const bestStatus = problem.best_status ? problem.best_status.toLowerCase() : null;

                const rowBg = index % 2 === 0 
                  ? "bg-transparent" 
                  : (isLight ? "bg-slate-50/30" : "bg-slate-900/10");

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
                    className={`group cursor-pointer text-sm transition-colors duration-150 ${rowBg} ${
                      isLight ? "hover:bg-slate-50" : "hover:bg-slate-900/30"
                    }`}
                  >
                    {/* Status Indicator */}
                    <td className="py-4 text-center">
                      <div className="flex items-center justify-center">
                        {renderStatusIndicator(problem)}
                      </div>
                    </td>

                    {/* ID */}
                    <td className={`px-4 py-4 text-xs font-mono font-medium truncate max-w-[170px] ${
                      isLight ? "text-slate-500 group-hover:text-blue-600" : "text-slate-400 group-hover:text-cyan-400"
                    }`}>
                      {displayId}
                    </td>

                    {/* Name */}
                    <td className="px-4 py-4 font-semibold">
                      <span className={`tracking-wide transition-colors group-hover:underline ${isLight ? "text-slate-700" : "text-slate-100"}`}>
                        {problem.name}
                      </span>
                    </td>

                    {/* Category */}
                    <td className={`px-4 py-4 text-xs font-medium ${isLight ? "text-slate-500" : "text-slate-400"}`}>
                      Machine Learning
                    </td>

                    {/* Points */}
                    <td className={`px-4 py-4 text-xs font-mono font-semibold text-center ${isLight ? "text-slate-600" : "text-slate-300"}`}>
                      {problem.best_score !== null ? Number(problem.best_score).toFixed(2) : "0.00"}
                    </td>

                    {/* Visibility */}
                    <td className="px-4 py-4 text-center">
                      <span className={`text-[10px] font-semibold px-2 py-1 rounded-md ${
                        isPrivate 
                          ? (isLight ? "bg-amber-50 text-amber-600 border border-amber-200/60" : "bg-amber-500/10 text-amber-500 border border-amber-500/10") 
                          : (isLight ? "bg-emerald-50 text-emerald-600 border border-emerald-200/60" : "bg-emerald-500/10 text-emerald-500 border border-emerald-500/10")
                      }`}>
                        {isPrivate ? "Private" : "Public"}
                      </span>
                    </td>

                    {/* Request status / Request Actions */}
                    <td className="px-4 py-4 text-center" onClick={(e) => e.stopPropagation()}>
                      {isPrivate ? (
                        role === "admin" ? (
                          <div className="flex justify-center gap-1.5">
                            <button
                              type="button"
                              onClick={(e) => handleAdminApproveDirectly(e, problem.id, "approve")}
                              className="rounded-md bg-emerald-600 hover:bg-emerald-500 px-2 py-1 text-[10px] font-bold text-white transition shadow-sm"
                            >
                              Approve
                            </button>
                            <button
                              type="button"
                              onClick={(e) => handleAdminApproveDirectly(e, problem.id, "reject")}
                              className="rounded-md bg-rose-600 hover:bg-rose-500 px-2 py-1 text-[10px] font-bold text-white transition shadow-sm"
                            >
                              Reject
                            </button>
                          </div>
                        ) : (
                          problem.author_id === currentUserId ? (
                            reqStatus === "PENDING" ? (
                              <span className="text-[10px] font-medium text-slate-400">Pending Review</span>
                            ) : (
                              <button
                                type="button"
                                onClick={(e) => handleRequestPublic(e, problem.id)}
                                className="rounded-md bg-amber-500 hover:bg-amber-400 text-slate-955 px-2.5 py-1 text-[10px] font-bold transition shadow-sm"
                              >
                                Request Public
                              </button>
                            )
                          ) : (
                            <span className="text-[10px] text-slate-400 font-medium">-</span>
                          )
                        )
                      ) : (
                        <span className={`text-[10px] font-bold ${isLight ? "text-emerald-600" : "text-emerald-400"}`}>Published</span>
                      )}
                    </td>

                    {/* Sửa và Xóa */}
                    <td className="px-4 py-4 text-center" onClick={(e) => e.stopPropagation()}>
                      {role === "admin" || (role === "contributor" && problem.author_id === currentUserId && reqStatus !== "APPROVED") ? (
                        <div className="flex justify-center gap-1.5">
                          <button
                            type="button"
                            onClick={(e) => handleEditProblemInit(e, problem)}
                            className="rounded-md bg-indigo-600 hover:bg-indigo-500 px-2.5 py-1 text-[10px] font-bold text-white transition shadow-sm"
                          >
                            Edit
                          </button>
                          <button
                            type="button"
                            onClick={(e) => handleDeleteProblemDirect(e, problem.id)}
                            className="rounded-md bg-rose-600 hover:bg-rose-500 px-2.5 py-1 text-[10px] font-bold text-white transition shadow-sm"
                          >
                            Delete
                          </button>
                        </div>
                      ) : (
                        <span className="text-[10px] text-slate-400 font-medium">-</span>
                      )}
                    </td>

                    {/* Submission Status badge */}
                    <td className="px-4 py-4 text-center">
                      {!bestStatus ? (
                        <span className={`text-[10px] font-semibold px-2 py-1 rounded-md ${
                          isLight 
                            ? "bg-slate-100 text-slate-500 border border-slate-200" 
                            : "bg-slate-500/10 text-slate-400 border border-slate-500/10"
                        }`}>
                          Unattempted
                        </span>
                      ) : bestStatus === "accepted" ? (
                        <span className={`text-[10px] font-semibold px-2 py-1 rounded-md ${
                          isLight 
                            ? "bg-emerald-50 text-emerald-600 border border-emerald-200" 
                            : "bg-emerald-500/10 text-emerald-400 border border-emerald-500/10"
                        }`}>
                          Accepted
                        </span>
                      ) : bestStatus === "wrong_answer" || bestStatus === "wrong answer" ? (
                        <span className={`text-[10px] font-semibold px-2 py-1 rounded-md ${
                          isLight 
                            ? "bg-rose-50 text-rose-600 border border-rose-200" 
                            : "bg-rose-500/10 text-rose-400 border border-rose-500/10"
                        }`}>
                          Wrong Answer
                        </span>
                      ) : bestStatus === "runtime_error" || bestStatus === "runtime error" ? (
                        <span className={`text-[10px] font-semibold px-2 py-1 rounded-md ${
                          isLight 
                            ? "bg-amber-50 text-amber-600 border border-amber-200" 
                            : "bg-amber-500/10 text-amber-400 border border-amber-500/10"
                        }`}>
                          Runtime Error
                        </span>
                      ) : (
                        <span className={`text-[10px] font-semibold px-2 py-1 rounded-md ${
                          isLight 
                            ? "bg-cyan-50 text-cyan-600 border border-cyan-200" 
                            : "bg-cyan-500/10 text-cyan-400 border border-cyan-500/10"
                        }`}>
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
        <div className="fixed inset-0 z-[1200] flex items-start justify-center overflow-y-auto bg-black/65 px-4 py-10">
          <div className="my-auto w-full max-w-3xl rounded-2xl border border-white/10 bg-slate-950/95 p-6 shadow-[0_0_50px_rgba(0,0,0,0.7)]">
            <div className="mb-4 flex items-center justify-between">
              <h3 className="text-xl font-bold tracking-wide text-white">
                {editingProblemId ? "Edit Problem" : "Create Problem"}
              </h3>
              <button
                type="button"
                onClick={closeModal}
                className="rounded-md border border-slate-700 px-3 py-1 text-xs text-slate-300 hover:border-slate-500"
              >
                CLOSE
              </button>
            </div>

            {error && (
              <div className="mb-4 rounded-lg border border-red-500/30 bg-red-950/40 p-3 text-sm text-red-300">
                {error}
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-5">
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                    Name *
                  </label>
                  <input
                    name="name"
                    required
                    value={formData.name}
                    onChange={handleChange}
                    className="mt-1.5 w-full rounded-lg border border-slate-800 bg-slate-900/70 px-3 py-2.5 text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                    Source
                  </label>
                  <input
                    name="source"
                    value={formData.source}
                    onChange={handleChange}
                    className="mt-1.5 w-full rounded-lg border border-slate-800 bg-slate-900/70 px-3 py-2.5 text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
                  />
                </div>
              </div>

              <div>
                <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Statement Markdown *
                </label>
                <textarea
                  name="statement_markdown"
                  required
                  rows={5}
                  value={formData.statement_markdown}
                  onChange={handleChange}
                  className="mt-1.5 w-full rounded-lg border border-slate-800 bg-slate-900/70 px-3 py-2.5 text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Theory Markdown
                </label>
                <textarea
                  name="theory_markdown"
                  rows={4}
                  value={formData.theory_markdown}
                  onChange={handleChange}
                  className="mt-1.5 w-full rounded-lg border border-slate-800 bg-slate-900/70 px-3 py-2.5 text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Tutorial Markdown
                </label>
                <textarea
                  name="tutorial_markdown"
                  rows={4}
                  value={formData.tutorial_markdown}
                  onChange={handleChange}
                  className="mt-1.5 w-full rounded-lg border border-slate-800 bg-slate-900/70 px-3 py-2.5 text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Solution Markdown
                </label>
                <textarea
                  name="solution_markdown"
                  rows={4}
                  value={formData.solution_markdown}
                  onChange={handleChange}
                  className="mt-1.5 w-full rounded-lg border border-slate-800 bg-slate-900/70 px-3 py-2.5 text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Coding Markdown
                </label>
                <textarea
                  name="coding_markdown"
                  rows={4}
                  value={formData.coding_markdown}
                  onChange={handleChange}
                  className="mt-1.5 w-full rounded-lg border border-slate-800 bg-slate-900/70 px-3 py-2.5 text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Checker Markdown
                </label>
                <textarea
                  name="checker_markdown"
                  rows={4}
                  value={formData.checker_markdown}
                  onChange={handleChange}
                  className="mt-1.5 w-full rounded-lg border border-slate-800 bg-slate-900/70 px-3 py-2.5 text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
                />
              </div>

              {!editingProblemId && (
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                      Input Zip File (.zip)
                    </label>
                    <input
                      type="file"
                      accept=".zip"
                      onChange={(e) => setInputZipFile(e.target.files[0])}
                      className="mt-1.5 w-full rounded-lg border border-slate-800 bg-slate-900/70 px-3 py-2.5 text-sm text-slate-100"
                    />
                  </div>
                  <div>
                    <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                      Output Zip File (.zip)
                    </label>
                    <input
                      type="file"
                      accept=".zip"
                      onChange={(e) => setOutputZipFile(e.target.files[0])}
                      className="mt-1.5 w-full rounded-lg border border-slate-800 bg-slate-900/70 px-3 py-2.5 text-sm text-slate-100"
                    />
                  </div>
                </div>
              )}

              <div className="flex justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={closeModal}
                  className="rounded-lg border border-slate-700 bg-slate-900/60 px-4 py-2 text-xs font-semibold text-slate-300"
                >
                  CANCEL
                </button>
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="rounded-lg bg-gradient-to-r from-cyan-500 via-indigo-500 to-emerald-600 px-4 py-2 text-xs font-semibold tracking-wide text-white disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isSubmitting ? "PROCESSING..." : editingProblemId ? "SAVE CHANGES" : "CREATE"}
                </button>
              </div>
            </form>
          </div>
        </div>
      , document.body)}
    </section>
  );
}