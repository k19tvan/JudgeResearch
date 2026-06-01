// src/components/tabs/AdminQueueTab.jsx
import React, { useEffect, useState } from "react";

export default function AdminQueueTab({ isLight = false }) {
  const [requests, setRequests] = useState([]);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(null);
  const adminId = localStorage.getItem("user_id");

  const loadPendingRequests = async () => {
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:21081/api/problems/pending-requests?admin_id=${adminId}`);
      const result = await response.json();
      setRequests(result?.data || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPendingRequests();
  }, []);

  const handleAction = async (problemId, action) => {
    setActionLoading(problemId);
    try {
      const response = await fetch(`http://localhost:21081/api/problems/${problemId}/${action}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ admin_id: Number(adminId) }),
      });
      if (response.ok) {
        await loadPendingRequests(); // Tải lại danh sách sau khi hành động thành công
      }
    } catch (err) {
      console.error(err);
    } finally {
      setActionLoading(null);
    }
  };

  return (
    <div className="space-y-4">
      <div>
        <h2 className={`text-xl font-bold ${isLight ? "text-slate-900" : "text-white"}`}>ADMIN APPROVAL QUEUE</h2>
        <p className={`text-xs ${isLight ? "text-slate-500" : "text-slate-400"}`}>Approve or reject community requests to make private exercises public.</p>
      </div>

      {loading ? (
        <p className="text-xs text-slate-400 animate-pulse">Loading approval queue...</p>
      ) : requests.length === 0 ? (
        <p className="text-xs text-slate-400">All caught up! No pending requests.</p>
      ) : (
        <div className={`overflow-x-auto rounded-xl border ${isLight ? "border-slate-200 bg-white" : "border-white/5 bg-slate-950/40"}`}>
          <table className="w-full text-left border-collapse min-w-[700px]">
            <thead>
              <tr className={`border-b text-xs font-bold uppercase tracking-wider ${isLight ? "bg-slate-800 text-white" : "bg-slate-950 text-slate-200"}`}>
                <th className="px-4 py-3">ID</th>
                <th className="px-4 py-3">Problem Name</th>
                <th className="px-4 py-3">Author</th>
                <th className="px-4 py-3">Requested Date</th>
                <th className="px-4 py-3 text-center">Actions</th>
              </tr>
            </thead>
            <tbody>
              {requests.map((req, idx) => (
                <tr key={req.id} className={`border-b text-sm ${idx % 2 === 0 ? (isLight ? "bg-white" : "bg-slate-900/10") : (isLight ? "bg-slate-50/50" : "bg-slate-950/25")}`}>
                  <td className="px-4 py-3 font-mono text-xs">ml_{req.id}</td>
                  <td className="px-4 py-3 font-semibold">{req.name}</td>
                  <td className="px-4 py-3 text-xs text-slate-300">{req.author_name}</td>
                  <td className="px-4 py-3 text-xs text-slate-400">{new Date(req.created_at).toLocaleDateString()}</td>
                  <td className="px-4 py-3 text-center">
                    <div className="flex justify-center gap-2">
                      <button
                        type="button"
                        disabled={actionLoading === req.id}
                        onClick={() => handleAction(req.id, "approve")}
                        className="rounded bg-emerald-600 hover:bg-emerald-500 px-3 py-1 text-xs font-semibold text-white transition disabled:opacity-50"
                      >
                        APPROVE
                      </button>
                      <button
                        type="button"
                        disabled={actionLoading === req.id}
                        onClick={() => handleAction(req.id, "reject")}
                        className="rounded bg-rose-600 hover:bg-rose-500 px-3 py-1 text-xs font-semibold text-white transition disabled:opacity-50"
                      >
                        REJECT
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}