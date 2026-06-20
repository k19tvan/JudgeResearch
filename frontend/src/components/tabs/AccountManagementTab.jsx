import React, { useEffect, useMemo, useState } from "react";
import { API_BASE_URL, fetchManagedUserDetails, fetchManagedUsers, updateManagedUser } from "../../api";

const emptyForm = {
  display_name: "",
  email: "",
  role: "user",
  status: "active",
};

const emailPattern = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;

export default function AccountManagementTab({ isLight = false }) {
  const [users, setUsers] = useState([]);
  const [selectedUser, setSelectedUser] = useState(null);
  const [formData, setFormData] = useState(emptyForm);
  const [search, setSearch] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isDetailsLoading, setIsDetailsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [fieldErrors, setFieldErrors] = useState({});

  const currentUserId = Number(localStorage.getItem("user_id"));

  const tone = useMemo(() => ({
    title: isLight ? "text-slate-900" : "text-white",
    body: isLight ? "text-slate-700" : "text-slate-300",
    muted: isLight ? "text-slate-500" : "text-slate-400",
    panel: isLight ? "border-slate-200 bg-white shadow-sm" : "border-white/10 bg-slate-900/40",
    input: isLight
      ? "border-slate-300 bg-white text-slate-900 placeholder-slate-400 focus:border-emerald-500"
      : "border-slate-800 bg-slate-950/70 text-slate-100 placeholder-slate-600 focus:border-cyan-500",
    value: isLight ? "text-slate-800" : "text-slate-200",
    row: isLight ? "hover:bg-slate-50" : "hover:bg-white/[0.03]",
  }), [isLight]);

  const getAvatarUrl = (url) => {
    if (!url) return null;
    return url.startsWith("/") ? `${API_BASE_URL}${url}` : url;
  };

  const syncFormFromUser = (user) => {
    setFormData({
      display_name: user?.display_name || "",
      email: user?.email || "",
      role: user?.role || "user",
      status: user?.status || "active",
    });
  };

  const loadUsers = async (term = search) => {
    setIsLoading(true);
    setError("");
    try {
      const result = await fetchManagedUsers(term);
      setUsers(result?.data || []);
    } catch (err) {
      setError(err.message || "Fetch managed users failed");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadUsers("");
  }, []);

  const handleSearchSubmit = (event) => {
    event.preventDefault();
    loadUsers(search);
  };

  const handleSelectUser = async (userId) => {
    setIsDetailsLoading(true);
    setError("");
    setSuccess("");
    setFieldErrors({});
    try {
      const result = await fetchManagedUserDetails(userId);
      setSelectedUser(result?.data);
      syncFormFromUser(result?.data);
    } catch (err) {
      setError(err.message || "Fetch managed user failed");
    } finally {
      setIsDetailsLoading(false);
    }
  };

  const handleChange = (event) => {
    const { name, value } = event.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    setFieldErrors((prev) => {
      if (!prev[name]) return prev;
      const next = { ...prev };
      delete next[name];
      return next;
    });
  };

  const validateForm = () => {
    const nextErrors = {};
    const messages = [];
    const displayName = formData.display_name.trim();
    const email = formData.email.trim();

    if (!displayName) {
      nextErrors.display_name = true;
      messages.push("Display name is required.");
    }
    if (!email) {
      nextErrors.email = true;
      messages.push("Email is required.");
    } else if (!emailPattern.test(email)) {
      nextErrors.email = true;
      messages.push("Email format is invalid.");
    }

    setFieldErrors(nextErrors);
    if (messages.length) {
      setError(messages.join(" "));
      return false;
    }
    return true;
  };

  const handleCancel = () => {
    if (selectedUser) syncFormFromUser(selectedUser);
    setFieldErrors({});
    setError("");
    setSuccess("");
  };

  const handleBack = () => {
    setSelectedUser(null);
    setFormData(emptyForm);
    setFieldErrors({});
    setError("");
    setSuccess("");
  };

  const handleSave = async (event) => {
    event.preventDefault();
    setError("");
    setSuccess("");

    if (!selectedUser || !validateForm()) return;

    const payload = {
      display_name: formData.display_name.trim(),
      email: formData.email.trim(),
      role: formData.role,
      status: formData.status,
    };

    setIsSaving(true);
    try {
      const result = await updateManagedUser(selectedUser.id, payload);
      const updated = result?.data;
      setSelectedUser(updated);
      syncFormFromUser(updated);
      setUsers((prev) => prev.map((user) => (user.id === updated.id ? updated : user)));
      if (updated.id === currentUserId) {
        localStorage.setItem("user_role", updated.role);
      }
      setSuccess(result.message || "Update successful");
    } catch (err) {
      setError(err.message || "Update managed user failed");
    } finally {
      setIsSaving(false);
    }
  };

  const getStatusBadge = (status) => {
    const active = status === "active";
    return active
      ? (isLight ? "border-emerald-200 bg-emerald-50 text-emerald-700" : "border-emerald-500/20 bg-emerald-500/10 text-emerald-300")
      : (isLight ? "border-red-200 bg-red-50 text-red-700" : "border-red-500/20 bg-red-500/10 text-red-300");
  };

  const getRoleBadge = (role) => {
    if (role === "admin") return isLight ? "border-rose-200 bg-rose-50 text-rose-700" : "border-rose-500/20 bg-rose-500/10 text-rose-300";
    if (role === "contributor") return isLight ? "border-amber-200 bg-amber-50 text-amber-700" : "border-amber-500/20 bg-amber-500/10 text-amber-300";
    return isLight ? "border-cyan-200 bg-cyan-50 text-cyan-700" : "border-cyan-500/20 bg-cyan-500/10 text-cyan-300";
  };

  const getInputClass = (field) => {
    const errorClass = isLight
      ? "border-red-400 focus:border-red-500"
      : "border-red-500/70 focus:border-red-400";
    return `mt-1.5 w-full rounded-lg border px-3 py-2.5 text-sm outline-none ${fieldErrors[field] ? errorClass : tone.input}`;
  };

  return (
    <section className="space-y-6">
      <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div>
          <h2 className={`text-2xl font-bold tracking-wide ${tone.title}`}>ACCOUNT MANAGEMENT</h2>
          <p className={`mt-1 text-sm ${tone.muted}`}>Review accounts, update roles, and control account access.</p>
        </div>
        <form onSubmit={handleSearchSubmit} className="flex w-full gap-2 md:w-[360px]">
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search users..."
            className={`w-full rounded-lg border px-3 py-2 text-sm outline-none ${tone.input}`}
          />
          <button
            type="submit"
            className="rounded-lg bg-emerald-600 px-4 py-2 text-xs font-semibold text-white transition hover:bg-emerald-500"
          >
            Search
          </button>
        </form>
      </div>

      {(error || success) && (
        <div className={`rounded-lg border p-3 text-xs ${
          error
            ? (isLight ? "border-red-200 bg-red-50 text-red-700" : "border-red-500/30 bg-red-500/10 text-red-300")
            : (isLight ? "border-emerald-300 bg-emerald-50 text-emerald-700" : "border-emerald-500/30 bg-emerald-500/10 text-emerald-300")
        }`}>
          {error || success}
        </div>
      )}

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.25fr)_minmax(360px,0.75fr)]">
        <div className={`overflow-hidden rounded-lg border ${tone.panel}`}>
          <div className={`flex items-center justify-between border-b px-5 py-3 ${isLight ? "border-slate-100" : "border-white/5"}`}>
            <span className={`text-xs font-semibold uppercase tracking-widest ${tone.muted}`}>User Accounts</span>
            <span className={`text-xs ${tone.muted}`}>{users.length} records</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full min-w-[760px] text-left text-sm">
              <thead className={isLight ? "bg-slate-50 text-slate-500" : "bg-white/[0.02] text-slate-400"}>
                <tr>
                  <th className="px-5 py-3 text-xs font-semibold uppercase">Account</th>
                  <th className="px-5 py-3 text-xs font-semibold uppercase">Email</th>
                  <th className="px-5 py-3 text-xs font-semibold uppercase">Role</th>
                  <th className="px-5 py-3 text-xs font-semibold uppercase">Status</th>
                  <th className="px-5 py-3 text-xs font-semibold uppercase">Updated</th>
                </tr>
              </thead>
              <tbody className={isLight ? "divide-y divide-slate-100" : "divide-y divide-white/5"}>
                {isLoading ? (
                  <tr>
                    <td colSpan="5" className={`px-5 py-8 text-center text-sm ${tone.muted}`}>Loading accounts...</td>
                  </tr>
                ) : users.length === 0 ? (
                  <tr>
                    <td colSpan="5" className={`px-5 py-8 text-center text-sm ${tone.muted}`}>No accounts found.</td>
                  </tr>
                ) : users.map((user) => {
                  const selected = selectedUser?.id === user.id;
                  return (
                    <tr
                      key={user.id}
                      onClick={() => handleSelectUser(user.id)}
                      className={`${tone.row} ${selected ? (isLight ? "bg-emerald-50/70" : "bg-emerald-500/10") : ""} cursor-pointer transition`}
                    >
                      <td className="px-5 py-4">
                        <div className="flex items-center gap-3">
                          <div className={`h-9 w-9 overflow-hidden rounded-full border ${isLight ? "border-slate-200 bg-slate-100" : "border-white/10 bg-slate-800"}`}>
                            {user.avatar_url ? (
                              <img src={getAvatarUrl(user.avatar_url)} alt="" className="h-full w-full object-cover" />
                            ) : (
                              <div className={`flex h-full w-full items-center justify-center text-xs font-bold ${tone.muted}`}>
                                {(user.display_name || user.username || "?").slice(0, 2).toUpperCase()}
                              </div>
                            )}
                          </div>
                          <div className="min-w-0">
                            <p className={`truncate font-semibold ${tone.value}`}>{user.display_name}</p>
                            <p className={`truncate font-mono text-xs ${tone.muted}`}>@{user.username}{user.id === currentUserId ? " · You" : ""}</p>
                          </div>
                        </div>
                      </td>
                      <td className={`px-5 py-4 font-mono text-xs ${tone.value}`}>{user.email}</td>
                      <td className="px-5 py-4">
                        <span className={`rounded-full border px-2.5 py-1 text-[11px] font-bold uppercase ${getRoleBadge(user.role)}`}>{user.role}</span>
                      </td>
                      <td className="px-5 py-4">
                        <span className={`rounded-full border px-2.5 py-1 text-[11px] font-bold uppercase ${getStatusBadge(user.status)}`}>{user.status}</span>
                      </td>
                      <td className={`px-5 py-4 text-xs ${tone.muted}`}>
                        {user.updated_at ? new Date(user.updated_at).toLocaleDateString() : "-"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        <div className={`rounded-lg border ${tone.panel}`}>
          <div className={`border-b px-5 py-3 ${isLight ? "border-slate-100" : "border-white/5"}`}>
            <span className={`text-xs font-semibold uppercase tracking-widest ${tone.muted}`}>Account Details</span>
          </div>

          {!selectedUser ? (
            <div className={`px-5 py-8 text-sm ${tone.muted}`}>Select an account to view and edit details.</div>
          ) : isDetailsLoading ? (
            <div className={`px-5 py-8 text-sm ${tone.muted}`}>Loading account details...</div>
          ) : (
            <form onSubmit={handleSave} className="space-y-4 px-5 py-5">
              <div className="flex items-center gap-3">
                <div className={`h-12 w-12 overflow-hidden rounded-full border ${isLight ? "border-slate-200 bg-slate-100" : "border-white/10 bg-slate-800"}`}>
                  {selectedUser.avatar_url ? (
                    <img src={getAvatarUrl(selectedUser.avatar_url)} alt="" className="h-full w-full object-cover" />
                  ) : (
                    <div className={`flex h-full w-full items-center justify-center text-sm font-bold ${tone.muted}`}>
                      {(selectedUser.display_name || selectedUser.username || "?").slice(0, 2).toUpperCase()}
                    </div>
                  )}
                </div>
                <div className="min-w-0">
                  <p className={`truncate font-semibold ${tone.value}`}>@{selectedUser.username}</p>
                  <p className={`text-xs ${tone.muted}`}>Created {new Date(selectedUser.created_at).toLocaleDateString()}</p>
                </div>
              </div>

              <label className={`block text-xs font-semibold uppercase ${tone.muted}`}>
                Display Name
                <input
                  name="display_name"
                  value={formData.display_name}
                  onChange={handleChange}
                  className={getInputClass("display_name")}
                  aria-invalid={fieldErrors.display_name ? "true" : "false"}
                />
              </label>

              <label className={`block text-xs font-semibold uppercase ${tone.muted}`}>
                Email
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  className={getInputClass("email")}
                  aria-invalid={fieldErrors.email ? "true" : "false"}
                />
              </label>

              <div className="grid gap-4 sm:grid-cols-2">
                <label className={`block text-xs font-semibold uppercase ${tone.muted}`}>
                  Role
                  <select name="role" value={formData.role} onChange={handleChange} className={`mt-1.5 w-full rounded-lg border px-3 py-2.5 text-sm outline-none ${tone.input}`}>
                    <option value="user">User</option>
                    <option value="contributor">Contributor</option>
                    <option value="admin">Admin</option>
                  </select>
                </label>

                <label className={`block text-xs font-semibold uppercase ${tone.muted}`}>
                  Status
                  <select name="status" value={formData.status} onChange={handleChange} className={`mt-1.5 w-full rounded-lg border px-3 py-2.5 text-sm outline-none ${tone.input}`}>
                    <option value="active">Active</option>
                    <option value="disabled">Disabled</option>
                  </select>
                </label>
              </div>

              <div className="flex flex-wrap gap-3 pt-2">
                <button
                  type="submit"
                  disabled={isSaving}
                  className="rounded-lg bg-emerald-600 px-4 py-2.5 text-xs font-semibold text-white transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isSaving ? "SAVING..." : "SAVE"}
                </button>
                <button
                  type="button"
                  onClick={handleCancel}
                  disabled={isSaving}
                  className={`rounded-lg border px-4 py-2.5 text-xs font-semibold transition disabled:cursor-not-allowed disabled:opacity-60 ${
                    isLight ? "border-slate-300 text-slate-700 hover:bg-slate-100" : "border-white/10 text-slate-300 hover:bg-white/5"
                  }`}
                >
                  CANCEL
                </button>
                <button
                  type="button"
                  onClick={handleBack}
                  disabled={isSaving}
                  className={`rounded-lg border px-4 py-2.5 text-xs font-semibold transition disabled:cursor-not-allowed disabled:opacity-60 ${
                    isLight ? "border-slate-300 text-slate-700 hover:bg-slate-100" : "border-white/10 text-slate-300 hover:bg-white/5"
                  }`}
                >
                  BACK
                </button>
              </div>
            </form>
          )}
        </div>
      </div>
    </section>
  );
}
