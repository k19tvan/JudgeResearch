// src/components/tabs/UsersTab.jsx
import React, { useEffect, useState } from "react";

export default function UsersTab({ isLight = false }) {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [actionLoading, setActionLoading] = useState(null);
  
  const currentUserId = localStorage.getItem("user_id");

  const loadUsersList = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`http://localhost:21081/api/admin/users?admin_id=${currentUserId}`);
      const result = await response.json();
      if (response.ok) {
        setUsers(result.data || []);
      } else {
        setError(result.detail || "Failed to load users list");
      }
    } catch (err) {
      setError("Failed to fetch users: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadUsersList();
  }, []);

  const handleUpdateUserRole = async (targetUserId, newRole) => {
    setActionLoading(targetUserId);
    try {
      const response = await fetch(`http://localhost:21081/api/admin/users/${targetUserId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          admin_id: Number(currentUserId),
          role: newRole,
        }),
      });
      if (response.ok) {
        await loadUsersList();
      } else {
        const result = await response.json();
        alert(result.detail || "Failed to update role");
      }
    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      setActionLoading(null);
    }
  };

  const handleToggleUserStatus = async (targetUserId, currentStatus) => {
    const newStatus = currentStatus === "active" ? "disabled" : "active";
    setActionLoading(targetUserId);
    try {
      const response = await fetch(`http://localhost:21081/api/admin/users/${targetUserId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          admin_id: Number(currentUserId),
          status: newStatus,
        }),
      });
      if (response.ok) {
        await loadUsersList();
      } else {
        const result = await response.json();
        alert(result.detail || "Failed to update status");
      }
    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      setActionLoading(null);
    }
  };

  return (
    <div className="space-y-4">
      <div>
        <h2 className={`text-xl font-bold ${isLight ? "text-slate-900" : "text-white"}`}>MANAGE SYSTEM USERS</h2>
        <p className={`text-xs ${isLight ? "text-slate-500" : "text-slate-400"}`}>
          Modify member roles or temporarily disable / activate user accounts.
        </p>
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-300">
          {error}
        </div>
      )}

      {loading ? (
        <p className="text-xs text-slate-400 animate-pulse">Loading users workspace...</p>
      ) : (
        <div className={`overflow-x-auto rounded-xl border ${isLight ? "border-slate-200 bg-white" : "border-white/5 bg-slate-950/40"}`}>
          <table className="w-full text-left border-collapse min-w-[800px]">
            <thead>
              <tr className={`border-b text-xs font-bold uppercase tracking-wider ${isLight ? "bg-slate-800 text-white" : "bg-slate-950 text-slate-200"}`}>
                <th className="px-4 py-3 w-20">ID</th>
                <th className="px-4 py-3">Username</th>
                <th className="px-4 py-3">Display Name</th>
                <th className="px-4 py-3">Email Address</th>
                <th className="px-4 py-3 text-center w-40">Role</th>
                <th className="px-4 py-3 text-center w-32">Status</th>
                <th className="px-4 py-3 text-center w-36">Actions</th>
              </tr>
            </thead>
            <tbody>
              {users.map((u, idx) => {
                const isSelf = Number(u.id) === Number(currentUserId);
                return (
                  <tr key={u.id} className={`border-b text-sm ${idx % 2 === 0 ? (isLight ? "bg-white" : "bg-slate-900/10") : (isLight ? "bg-slate-50/50" : "bg-slate-950/25")}`}>
                    <td className="px-4 py-3 font-mono text-xs">#{u.id}</td>
                    <td className="px-4 py-3 font-semibold">@{u.username}</td>
                    <td className="px-4 py-3">{u.display_name}</td>
                    <td className="px-4 py-3 text-xs font-mono">{u.email}</td>
                    <td className="px-4 py-3 text-center">
                      <select
                        value={u.role}
                        disabled={actionLoading === u.id || isSelf}
                        onChange={(e) => handleUpdateUserRole(u.id, e.target.value)}
                        className={`rounded border px-2 py-1 text-xs outline-none ${
                          isLight 
                            ? "border-slate-300 bg-white text-slate-800" 
                            : "border-slate-800 bg-slate-950 text-slate-300"
                        }`}
                      >
                        <option value="user">User</option>
                        <option value="contributor">Contributor</option>
                        <option value="admin">Admin</option>
                      </select>
                    </td>
                    <td className="px-4 py-3 text-center">
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${
                        u.status === "active"
                          ? "bg-emerald-500/10 text-emerald-500 border-emerald-500/20"
                          : "bg-rose-500/10 text-rose-500 border-rose-500/20"
                      }`}>
                        {u.status.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-center">
                      <button
                        type="button"
                        disabled={actionLoading === u.id || isSelf}
                        onClick={() => handleToggleUserStatus(u.id, u.status)}
                        className={`rounded px-3 py-1 text-xs font-semibold text-white transition disabled:opacity-50 disabled:cursor-not-allowed ${
                          u.status === "active" ? "bg-rose-600 hover:bg-rose-500" : "bg-emerald-600 hover:bg-emerald-500"
                        }`}
                      >
                        {u.status === "active" ? "DISABLE" : "ACTIVE"}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}