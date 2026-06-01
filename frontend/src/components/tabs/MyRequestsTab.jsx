// src/components/tabs/MyRequestsTab.jsx
import React, { useEffect, useState } from "react";

export default function MyRequestsTab({ isLight = false }) {
  const [requests, setRequests] = useState([]);
  const [loading, setLoading] = useState(true);
  const userId = localStorage.getItem("user_id");

  const loadMyRequests = async () => {
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:21081/api/problems/my-requests?user_id=${userId}`);
      const result = await response.json();
      setRequests(result?.data || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadMyRequests();
  }, []);

  return (
    <div className="space-y-4">
      <div>
        <h2 className={`text-xl font-bold ${isLight ? "text-slate-900" : "text-white"}`}>MY PUBLIC REQUESTS</h2>
        <p className={`text-xs ${isLight ? "text-slate-500" : "text-slate-400"}`}>Track status of your private problems requested to be public.</p>
      </div>

      {loading ? (
        <p className="text-xs text-slate-400 animate-pulse">Loading requests...</p>
      ) : requests.length === 0 ? (
        <p className="text-xs text-slate-400">You haven't requested any problems to be public yet.</p>
      ) : (
        <div className={`overflow-x-auto rounded-xl border ${isLight ? "border-slate-200 bg-white" : "border-white/5 bg-slate-950/40"}`}>
          <table className="w-full text-left border-collapse min-w-[600px]">
            <thead>
              <tr className={`border-b text-xs font-bold uppercase tracking-wider ${isLight ? "bg-slate-800 text-white" : "bg-slate-950 text-slate-200"}`}>
                <th className="px-4 py-3">Problem ID</th>
                <th className="px-4 py-3">Problem Name</th>
                <th className="px-4 py-3">Date Requested</th>
                <th className="px-4 py-3 text-center">Status</th>
              </tr>
            </thead>
            <tbody>
              {requests.map((req, idx) => (
                <tr key={req.id} className={`border-b text-sm ${idx % 2 === 0 ? (isLight ? "bg-white" : "bg-slate-900/10") : (isLight ? "bg-slate-50/50" : "bg-slate-950/25")}`}>
                  <td className="px-4 py-3 font-mono text-xs">ml_{req.id}</td>
                  <td className="px-4 py-3 font-semibold">{req.name}</td>
                  <td className="px-4 py-3 text-xs text-slate-400">{new Date(req.created_at).toLocaleDateString()}</td>
                  <td className="px-4 py-3 text-center">
                    {req.request_status === "PENDING" && <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">Pending</span>}
                    {req.request_status === "APPROVED" && <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">Approved</span>}
                    {req.request_status === "REJECTED" && <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20">Rejected</span>}
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