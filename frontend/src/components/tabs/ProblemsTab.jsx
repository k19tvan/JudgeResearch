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

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setIsSubmitting(true);

    try {
      const userId = localStorage.getItem("user_id");
      if (!userId) {
        throw new Error("Không tìm thấy thông tin đăng nhập. Vui lòng đăng nhập lại.");
      }

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
      const created = result?.data;

      if (created) {
        setProblems((prev) => [created, ...prev]);
      }

      resetForm();
      closeModal();
      await loadProblems();
    } catch (err) {
      setError(err.message || "Create problem failed");
    } finally {
      setIsSubmitting(false);
    }
  };

  // Hàm gọi yêu cầu duyệt công khai từ phía người dùng thường
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

  // Hàm xử lý duyệt bài trực tiếp dành cho Admin trên bảng bài tập
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

  // Hàm sinh chấm tròn biểu thị trạng thái động (Solved Status) ở cột đầu tiên
  const renderStatusIndicator = (problem) => {
    const score = problem.best_score;
    const status = problem.best_status;

    if (score === null || score === undefined) {
      return (
        <div 
          className={`h-2.5 w-2.5 rounded-full border ${
            isLight ? "border-slate-300 bg-transparent" : "border-slate-800 bg-transparent"
          }`} 
          title="Not attempted"
        />
      );
    }
    
    if (score === 100 || status === "accepted") {
      return (
        <div 
          className="h-2.5 w-2.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]" 
          title="Solved (100 pts)"
        />
      );
    }

    if (score > 0 && score < 100) {
      return (
        <div 
          className="h-2.5 w-2.5 rounded-full bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.6)]" 
          title={`Partial Solved (${score} pts)`}
        />
      );
    }

    return (
      <div 
        className="h-2.5 w-2.5 rounded-full bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.6)]" 
        title="Attempted / Failed"
      />
    );
  };

  return (
    <section className="space-y-6">
      {/* HEADER BAR */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-4">
          <h2 className={`text-2xl font-bold tracking-wide ${isLight ? "text-slate-900" : "text-white"}`}>PROBLEMS</h2>
          <select
            value={filterMode}
            onChange={(e) => setFilterMode(e.target.value)}
            className={`rounded-md border px-2.5 py-1 text-xs outline-none transition font-semibold ${
              isLight 
                ? "border-slate-300 bg-white text-slate-800" 
                : "border-white/5 bg-slate-950/80 text-slate-300 focus:border-cyan-500"
            }`}
          >
            <option value="public">Public</option>
            {/* Đổi nhãn hiển thị tại đây để đồng bộ logic mới */}
            <option value="private">My Problems</option> 
            <option value="all">All</option>
          </select>
        </div>
        <button
          type="button"
          onClick={() => setIsModalOpen(true)}
          className={`rounded-lg px-4 py-2 text-xs font-bold tracking-wider text-white transition ${
            isLight
              ? "bg-emerald-600 hover:bg-emerald-500 shadow-md shadow-emerald-600/10"
              : "bg-cyan-600 hover:bg-cyan-500 shadow-md shadow-cyan-600/10"
          }`}
        >
          CREATE PROBLEM
        </button>
      </div>

      {error && !isModalOpen && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-300">
          {error}
        </div>
      )}

      {/* OJ / VNOI STYLE PROBLEMS TABLE */}
      {isLoading ? (
        <p className={`text-xs ${isLight ? "text-slate-500" : "text-slate-400"} animate-pulse`}>Loading problems workspace...</p>
      ) : problems.length === 0 ? (
        <p className={`text-xs ${isLight ? "text-slate-500" : "text-slate-400"}`}>No created problems yet.</p>
      ) : (
        <div className={`overflow-x-auto rounded-xl border ${
          isLight ? "border-slate-200 bg-white shadow-sm" : "border-white/5 bg-slate-950/40"
        }`}>
          <table className="w-full text-left border-collapse min-w-[800px]">
            {/* Table Header */}
            <thead>
              <tr className={`border-b text-xs font-bold uppercase tracking-wider ${
                isLight 
                  ? "bg-emerald-600 text-white border-slate-700" 
                  : "bg-emerald-600 text-slate-200 border-white/5"
              }`}>
                <th className="w-12 py-3 text-center border-r border-slate-700/50">✔</th>
                <th className="px-4 py-3 border-r border-slate-700/50 w-48">ID</th>
                <th className="px-4 py-3 border-r border-slate-700/50">Problem</th>
                <th className="px-4 py-3 border-r border-slate-700/50 w-40">Category</th>
                <th className="px-4 py-3 border-r border-slate-700/50 w-24 text-center">Points</th>
                <th className="px-4 py-3 border-r border-slate-700/50 w-28 text-center">Visibility</th>
                <th className="px-4 py-3 border-r border-slate-700/50 w-32 text-center">Request</th>
                <th className="px-4 py-3 w-28 text-center">Status</th>
              </tr>
            </thead>

            {/* Table Body */}
            <tbody>
              {problems.map((problem, index) => {
                const isPrivate = problem.is_public === 0;
                const reqStatus = problem.request_status ? problem.request_status.toUpperCase() : "NONE";
                const bestStatus = problem.best_status ? problem.best_status.toLowerCase() : null;

                // Đổi xen kẽ background dòng (Zebra-striping)
                const rowBg = index % 2 === 0 
                  ? (isLight ? "bg-white" : "bg-slate-900/10") 
                  : (isLight ? "bg-slate-50/50" : "bg-slate-950/25");

                // Tạo chuỗi ID tinh gọn từ tên bài tập (ví dụ: giou_matrix_12)
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
                    className={`cursor-pointer border-b text-sm transition-colors duration-150 ${rowBg} ${
                      isLight ? "border-slate-100 hover:bg-slate-100/30" : "border-white/5 hover:bg-slate-900/40"
                    }`}
                  >
                    {/* 1. Cột đầu tiên: Chấm tròn biểu thị trạng thái động (Solved Status) */}
                    <td className={`py-3 text-center border-r ${isLight ? "border-slate-100" : "border-white/5"}`}>
                      <div className="flex items-center justify-center">
                        {renderStatusIndicator(problem)}
                      </div>
                    </td>

                    {/* 2. ID Column */}
                    <td className={`px-4 py-3 text-xs font-mono font-semibold border-r truncate max-w-[190px] ${
                      isLight ? "border-slate-100 text-blue-600" : "border-white/5 text-cyan-400"
                    }`}>
                      {displayId}
                    </td>

                    {/* 3. Problem Name Column */}
                    <td className={`px-4 py-3 font-semibold border-r ${isLight ? "border-slate-100" : "border-white/5"}`}>
                      <div className="flex flex-col">
                        <span
                          className={`text-left hover:underline tracking-wide ${
                            isLight ? "text-slate-800 hover:text-blue-600" : "text-slate-100 hover:text-cyan-300"
                          }`}
                        >
                          {problem.name}
                        </span>
                      </div>
                    </td>

                    {/* 4. Category Column */}
                    <td className={`px-4 py-3 text-xs font-medium border-r ${
                      isLight ? "border-slate-100 text-slate-600" : "border-white/5 text-slate-400"
                    }`}>
                      Machine Learning
                    </td>

                    {/* 5. Points Column (Hiển thị điểm số cao nhất đạt được) */}
                    <td className={`px-4 py-3 text-xs font-mono font-semibold text-center border-r ${
                      isLight ? "border-slate-100 text-slate-700" : "border-white/5 text-slate-300"
                    }`}>
                      {problem.best_score !== null ? Number(problem.best_score).toFixed(2) : "0.00"}
                    </td>

                    {/* 6. Visibility Column */}
                    <td className={`px-4 py-3 text-center border-r ${isLight ? "border-slate-100" : "border-white/5"}`}>
                      {isPrivate ? (
                        <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-amber-500/10 text-amber-500 border border-amber-500/20">
                          Private
                        </span>
                      ) : (
                        <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
                          Public
                        </span>
                      )}
                    </td>

                    {/* 7. Cột 6B: Request Action Column (Luồng duyệt phân quyền động) */}
                    <td className={`px-4 py-3 text-center border-r ${isLight ? "border-slate-100" : "border-white/5"}`}>
                      {isPrivate ? (
                        role === "admin" ? (
                          // GIAO DIỆN ADMIN: Các nút duyệt nhanh trực tiếp
                          <div className="flex justify-center gap-1">
                            <button
                              type="button"
                              onClick={(e) => handleAdminApproveDirectly(e, problem.id, "approve")}
                              className="rounded bg-emerald-600 hover:bg-emerald-500 px-2 py-0.5 text-[9px] font-bold text-white transition shadow-sm"
                            >
                              APPROVE
                            </button>
                            <button
                              type="button"
                              onClick={(e) => handleAdminApproveDirectly(e, problem.id, "reject")}
                              className="rounded bg-rose-600 hover:bg-rose-500 px-2 py-0.5 text-[9px] font-bold text-white transition shadow-sm"
                            >
                              REJECT
                            </button>
                          </div>
                        ) : (
                          // GIAO DIỆN USER THƯỜNG: Nút gửi yêu cầu (Chỉ hiển thị nếu mình là chủ sở hữu bài tập)
                          problem.author_id === currentUserId ? (
                            reqStatus === "PENDING" ? (
                              <span className="text-[9px] font-semibold text-slate-400">Waiting Admin</span>
                            ) : (
                              <button
                                type="button"
                                onClick={(e) => handleRequestPublic(e, problem.id)}
                                className="rounded bg-amber-500 hover:bg-amber-400 px-2 py-0.5 text-[9px] font-bold text-slate-900 transition shadow-sm"
                              >
                                REQUEST PUBLIC
                              </button>
                            )
                          ) : (
                            <span className="text-[9px] text-slate-500">-</span>
                          )
                        )
                      ) : (
                        <span className="text-[9px] text-emerald-400 font-semibold">Published</span>
                      )}
                    </td>

                    {/* 8. Cột 7: Hiển thị Trạng thái nộp bài (Submission Status) */}
                    <td className="px-4 py-3 text-center">
                      {!bestStatus ? (
                        <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-slate-500/10 text-slate-400 border border-slate-500/20">
                          Unattempted
                        </span>
                      ) : bestStatus === "accepted" ? (
                        <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
                          Accepted
                        </span>
                      ) : bestStatus === "wrong_answer" || bestStatus === "wrong answer" ? (
                        <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20">
                          Wrong Answer
                        </span>
                      ) : bestStatus === "runtime_error" || bestStatus === "runtime error" ? (
                        <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-amber-500/10 text-amber-500 border border-amber-500/20">
                          Runtime Error
                        </span>
                      ) : (
                        <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
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

      {/* PORTAL FOR MANUAL PROBLEM CREATE MODAL */}
      {isModalOpen && createPortal(
        <div className="fixed inset-0 z-[1200] flex items-start justify-center overflow-y-auto bg-black/65 px-4 py-10">
          <div className="my-auto w-full max-w-3xl rounded-2xl border border-white/10 bg-slate-950/95 p-6 shadow-[0_0_50px_rgba(0,0,0,0.7)]">
            <div className="mb-4 flex items-center justify-between">
              <h3 className="text-xl font-bold tracking-wide text-white">Create Problem</h3>
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
                  {isSubmitting ? "CREATING..." : "CREATE"}
                </button>
              </div>
            </form>
          </div>
        </div>
      , document.body)}
    </section>
  );
}