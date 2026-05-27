import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createPortal } from "react-dom";
import { createManualProblem, fetchProblems } from "../../api";

export default function ProblemsTab({ isLight = false }) {
  const navigate = useNavigate();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [problems, setProblems] = useState([]);
  const [formData, setFormData] = useState({
    name: "",
    source: "",
    statement_markdown: "",
    theory_markdown: "",
    tutorial_markdown: "",
    solution_markdown: "",
    coding_markdown: "",
    author_id: "",
  });

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
      author_id: "",
    });
  };

  useEffect(() => {
    const loadProblems = async () => {
      setIsLoading(true);
      try {
        // Get user_id from localStorage to fetch user's own problems too
        const userId = localStorage.getItem("user_id");
        const result = await fetchProblems(userId ? Number(userId) : null);
        setProblems(result?.data || []);
      } catch (err) {
        setError(err.message || "Fetch problems failed");
      } finally {
        setIsLoading(false);
      }
    };

    loadProblems();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setIsSubmitting(true);

    try {
      const payload = {
        ...formData,
        author_id: Number(formData.author_id),
      };

      const result = await createManualProblem(payload);
      const created = result?.data;

      if (created) {
        setProblems((prev) => [
          {
            ...created,
            statement_markdown: formData.statement_markdown,
            theory_markdown: formData.theory_markdown,
            tutorial_markdown: formData.tutorial_markdown,
            coding_markdown: formData.coding_markdown,
          },
          ...prev,
        ]);
      }

      resetForm();
      closeModal();
    } catch (err) {
      setError(err.message || "Create problem failed");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <section>
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <h2 className={`text-2xl font-bold tracking-wide ${isLight ? "text-slate-900" : "text-white"}`}>PROBLEMS</h2>
        <button
          type="button"
          onClick={() => setIsModalOpen(true)}
          className="rounded-lg bg-gradient-to-r from-emerald-500 via-emerald-600 to-teal-600 px-4 py-2 text-xs font-semibold tracking-wide text-white transition-all duration-300 hover:from-emerald-400 hover:via-emerald-500 hover:to-teal-500"
        >
          CREATE PROBLEM
        </button>
      </div>

      {error && !isModalOpen && (
        <div className="mt-4 rounded-lg border border-red-500/30 bg-red-950/40 p-3 text-sm text-red-300">
          {error}
        </div>
      )}

      <div className="mt-6 space-y-3">
        {isLoading ? (
          <p className={`text-sm ${isLight ? "text-slate-700" : "text-slate-400"}`}>Loading problems...</p>
        ) : problems.length === 0 ? (
          <p className={`text-sm ${isLight ? "text-slate-700" : "text-slate-400"}`}>No created problems yet.</p>
        ) : (
          problems.map((problem) => (
            <article
              key={problem.id}
              className={`group cursor-pointer rounded-xl border px-4 py-3 transition-all duration-300 ${
                isLight
                  ? "border-slate-300/90 bg-white/85 hover:border-emerald-500/50 hover:bg-white"
                  : "border-white/10 bg-slate-900/40 hover:border-cyan-400/50 hover:bg-slate-900/70"
              }`}
              onClick={() =>
                navigate(`/livecoding/${problem.id}`, {
                  state: { problem },
                })
              }
            >
              <h3 className={`text-sm font-semibold tracking-wide transition-colors duration-300 ${
                isLight ? "text-slate-900 group-hover:text-emerald-700" : "text-slate-100 group-hover:text-cyan-300"
              }`}>
                {problem.name}
              </h3>
            </article>
          ))
        )}
      </div>

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
                  Author ID *
                </label>
                <input
                  type="number"
                  name="author_id"
                  required
                  min="1"
                  value={formData.author_id}
                  onChange={handleChange}
                  className="mt-1.5 w-full rounded-lg border border-slate-800 bg-slate-900/70 px-3 py-2.5 text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
                />
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
