// src/components/tabs/ProfileTab.jsx
import React, { useEffect, useState, useMemo } from "react";

export default function ProfileTab({ isLight = false }) {
  const [profile, setProfile] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [secretKey, setSecretKey] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const userId = localStorage.getItem("user_id");

  const tone = useMemo(() => ({
    title: isLight ? "text-slate-900" : "text-white",
    body: isLight ? "text-slate-700" : "text-slate-300",
    muted: isLight ? "text-slate-500" : "text-slate-400",
    panel: isLight ? "border-slate-200 bg-white shadow-sm" : "border-white/10 bg-slate-900/40",
    input: isLight
      ? "border-slate-300 bg-white text-slate-900 placeholder-slate-400 focus:border-emerald-500"
      : "border-slate-800 bg-slate-950/70 text-slate-100 placeholder-slate-600 focus:border-cyan-500",
    divider: isLight ? "divide-slate-100" : "divide-white/5",
    value: isLight ? "text-slate-800" : "text-slate-200",
  }), [isLight]);

  const loadUserProfile = async () => {
    setIsLoading(true);
    setError("");
    try {
      const response = await fetch(`http://localhost:21081/api/users/profile/${userId}`);
      if (!response.ok) throw new Error("Failed to load user profile");
      const result = await response.json();
      setProfile(result?.data);
    } catch (err) {
      setError(err.message || "An error occurred");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (userId) loadUserProfile();
  }, [userId]);

    const handleBecomeAdmin = async (e) => {
        e.preventDefault();
        setError("");
        setSuccess("");
        setIsSubmitting(true);

        try {
        const response = await fetch("http://localhost:21081/api/users/make-admin", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
            user_id: Number(userId),
            secret_key: secretKey,
            }),
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || "Verification failed");
        }

        // CẬP NHẬT TỨC THỜI: Lưu trực tiếp quyền Admin mới vào localStorage của trình duyệt
        localStorage.setItem("user_role", "admin");

        setSuccess(result.message);
        setSecretKey("");
        
        // Tải lại trang sau 1.5 giây để cập nhật lại thanh Sidebar và các Tab mới
        setTimeout(() => {
            window.location.reload();
        }, 1500);

        } catch (err) {
        setError(err.message || "Something went wrong");
        } finally {
        setIsSubmitting(false);
        }
    };
    
  // Avatar initials từ display_name hoặc username
  const initials = profile
    ? (profile.display_name || profile.username || "?")
        .split(" ").map((w) => w[0]).join("").slice(0, 2).toUpperCase()
    : "?";

  if (isLoading) {
    return (
      <div className="flex items-center gap-2">
        <div className={`h-1.5 w-1.5 rounded-full animate-bounce ${isLight ? "bg-slate-400" : "bg-slate-500"}`} style={{ animationDelay: "0ms" }} />
        <div className={`h-1.5 w-1.5 rounded-full animate-bounce ${isLight ? "bg-slate-400" : "bg-slate-500"}`} style={{ animationDelay: "150ms" }} />
        <div className={`h-1.5 w-1.5 rounded-full animate-bounce ${isLight ? "bg-slate-400" : "bg-slate-500"}`} style={{ animationDelay: "300ms" }} />
        <span className={`text-sm ${tone.muted}`}>Loading profile...</span>
      </div>
    );
  }

  return (
    <section className="space-y-8 max-w-3xl">

      {/* ── Page Header ── */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className={`text-2xl font-bold tracking-wide ${tone.title}`}>USER PROFILE</h2>
          <p className={`mt-1 text-sm ${tone.muted}`}>Manage your personal details and account privileges.</p>
        </div>
        {/* Avatar */}
        {profile && (
          <div className={`flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center text-sm font-bold border ${
            isLight
              ? "bg-slate-100 border-slate-200 text-slate-600"
              : "bg-slate-800 border-white/10 text-slate-300"
          }`}>
            {initials}
          </div>
        )}
      </div>

      {/* ── Top: Info row (full width) ── */}
      {profile && (
        <div className={`rounded-xl border ${tone.panel}`}>
          {/* Card header */}
          <div className={`px-6 py-3 border-b text-xs font-semibold uppercase tracking-widest ${tone.muted} ${isLight ? "border-slate-100" : "border-white/5"}`}>
            Personal Information
          </div>

          {/* 4-column stat strip */}
          <div className={`grid grid-cols-2 md:grid-cols-4 divide-y md:divide-y-0 md:divide-x ${isLight ? "divide-slate-100" : "divide-white/5"}`}>
            {[
              { label: "Username",     value: `@${profile.username}`,   mono: true },
              { label: "Display Name", value: profile.display_name,      mono: false },
              { label: "Email",        value: profile.email,             mono: true, small: true },
              { label: "Member Since", value: new Date(profile.created_at).toLocaleDateString(), mono: false },
            ].map(({ label, value, mono, small }) => (
              <div key={label} className="px-6 py-4 space-y-1">
                <p className={`text-xs uppercase font-semibold ${tone.muted}`}>{label}</p>
                <p className={`font-semibold ${small ? "text-xs" : "text-sm"} ${mono ? "font-mono" : ""} ${tone.value} truncate`}>
                  {value}
                </p>
              </div>
            ))}
          </div>

          {/* Role row — full width, subtle */}
          <div className={`px-6 py-3 flex items-center justify-between border-t ${isLight ? "border-slate-100 bg-slate-50/50" : "border-white/5 bg-white/[0.02]"}`}>
            <span className={`text-xs uppercase font-semibold ${tone.muted}`}>Account Role</span>
            <span className={`text-xs font-bold uppercase px-3 py-1 rounded-full ${
              profile.role === "admin"
                ? (isLight ? "text-rose-700 bg-rose-50 border border-rose-200" : "text-rose-400 bg-rose-500/10 border border-rose-500/20")
                : (isLight ? "text-cyan-700 bg-cyan-50 border border-cyan-200" : "text-cyan-400 bg-cyan-500/10 border border-cyan-500/20")
            }`}>
              {profile.role}
            </span>
          </div>
        </div>
      )}

      {/* ── Bottom: Elevate Privileges (full width) ── */}
      <div className={`rounded-xl border ${tone.panel}`}>
        <div className={`px-6 py-3 border-b text-xs font-semibold uppercase tracking-widest ${tone.muted} ${isLight ? "border-slate-100" : "border-white/5"}`}>
          Elevate Privileges
        </div>

        <div className="px-6 py-5">
          {profile?.role !== "admin" ? (
            <div className="flex flex-col md:flex-row md:items-start gap-6">
              {/* Left: description */}
              <div className="md:w-1/2 space-y-1">
                <p className={`text-sm font-semibold ${tone.body}`}>Activate Admin Access</p>
                <p className={`text-xs leading-relaxed ${tone.muted}`}>
                  If you have an invitation key or system master token, enter it to gain rights to approve public requests and manage resources.
                </p>
                {success && (
                  <div className={`mt-3 rounded-lg border p-3 text-xs ${
                    isLight ? "border-emerald-300 bg-emerald-50 text-emerald-700" : "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                  }`}>
                    {success}
                  </div>
                )}
                {error && (
                  <div className={`mt-3 rounded-lg border p-3 text-xs ${
                    isLight ? "border-red-200 bg-red-50 text-red-700" : "border-red-500/30 bg-red-500/10 text-red-300"
                  }`}>
                    {error}
                  </div>
                )}
              </div>

              {/* Right: form */}
              <form onSubmit={handleBecomeAdmin} className="md:w-1/2 space-y-3">
                <label className={`block text-xs font-semibold uppercase ${tone.muted}`}>
                  Admin Secret Key
                  <input
                    type="password"
                    required
                    value={secretKey}
                    onChange={(e) => setSecretKey(e.target.value)}
                    placeholder="Enter key to verify..."
                    className={`mt-1.5 w-full rounded-lg border px-3 py-2.5 text-sm outline-none font-mono ${tone.input}`}
                  />
                </label>
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="w-full rounded-lg bg-emerald-600 px-4 py-2.5 text-xs font-semibold text-white transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isSubmitting ? "VERIFYING..." : "ACTIVATE ADMIN PRIVILEGES"}
                </button>
              </form>
            </div>
          ) : (
            <div className={`flex items-center gap-4 rounded-lg border p-4 ${
              isLight ? "border-emerald-200 bg-emerald-50/50" : "border-emerald-500/20 bg-emerald-500/5"
            }`}>
              <div className={`text-xl ${isLight ? "text-emerald-600" : "text-emerald-400"}`}>✓</div>
              <div>
                <p className={`text-xs font-bold uppercase tracking-wider ${isLight ? "text-emerald-700" : "text-emerald-400"}`}>
                  Admin Privileges Active
                </p>
                <p className={`text-[11px] mt-0.5 ${isLight ? "text-slate-600" : "text-slate-400"}`}>
                  You are already a system administrator.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

    </section>
  );
}