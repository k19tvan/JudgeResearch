// src/components/tabs/ProfileTab.jsx
import React, { useEffect, useMemo, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { deactivateUserAccount, fetchUserProfile, updateUserProfile } from "../../api";

const emptyForm = {
  username: "",
  display_name: "",
  email: "",
  password: "",
};

const usernamePattern = /^[A-Za-z0-9._-]+$/;
const emailPattern = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;

export default function ProfileTab({ isLight = false, onProfileUpdate }) {
  const navigate = useNavigate();
  const [profile, setProfile] = useState(null);
  const [formData, setFormData] = useState(emptyForm);
  const [avatarFile, setAvatarFile] = useState(null);
  const [avatarPreview, setAvatarPreview] = useState(null);
  const [fieldErrors, setFieldErrors] = useState({});
  const fileInputRef = useRef(null);
  const [isEditing, setIsEditing] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [secretKey, setSecretKey] = useState("");
  const [contributorSecretKey, setContributorSecretKey] = useState(""); // Lưu khóa cộng tác viên
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isDeactivating, setIsDeactivating] = useState(false);

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

  const getAvatarUrl = (url) => {
    if (!url) return null;
    return url.startsWith("/") ? `http://localhost:21081${url}` : url;
  };

  const syncFormFromProfile = (record) => {
    setFormData({
      username: record?.username || "",
      display_name: record?.display_name || "",
      email: record?.email || "",
      password: "",
    });
    setAvatarFile(null);
    setAvatarPreview(getAvatarUrl(record?.avatar_url));
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const loadUserProfile = async () => {
    setIsLoading(true);
    setError("");
    try {
      if (!userId) throw new Error("Authentication required");
      const result = await fetchUserProfile(userId);
      setProfile(result?.data);
      syncFormFromProfile(result?.data);
      if (onProfileUpdate) {
        onProfileUpdate(result?.data?.username, result?.data?.avatar_url);
      }
    } catch (err) {
      setError(err.message || "Failed to load user profile");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadUserProfile();
  }, [userId]);

  const getInputClass = (field) => {
    const errorClass = isLight
      ? "border-red-400 focus:border-red-500"
      : "border-red-500/70 focus:border-red-400";
    return `mt-1.5 w-full rounded-lg border px-3 py-2.5 text-sm outline-none ${fieldErrors[field] ? errorClass : tone.input}`;
  };

  const validateProfileForm = () => {
    const nextErrors = {};
    const messages = [];
    const trimmedUsername = formData.username.trim();
    const trimmedDisplayName = formData.display_name.trim();
    const trimmedEmail = formData.email.trim();

    if (!trimmedUsername) {
      nextErrors.username = true;
      messages.push("Username is required.");
    } else {
      if (trimmedUsername.length < 3 || trimmedUsername.length > 32) {
        nextErrors.username = true;
        messages.push("Username must be 3-32 characters long.");
      }
      if (!usernamePattern.test(trimmedUsername)) {
        nextErrors.username = true;
        messages.push("Username may include letters, numbers, dots, underscores, and dashes only.");
      }
      if (trimmedDisplayName && trimmedUsername.toLowerCase() === trimmedDisplayName.toLowerCase()) {
        nextErrors.username = true;
        messages.push("Username cannot duplicate a display name.");
      }
    }

    if (!trimmedDisplayName) {
      nextErrors.display_name = true;
      messages.push("Display name is required.");
    }

    if (!trimmedEmail) {
      nextErrors.email = true;
      messages.push("Email is required.");
    } else if (!emailPattern.test(trimmedEmail)) {
      nextErrors.email = true;
      messages.push("Email format is invalid.");
    }

    if (avatarFile) {
      const allowedTypes = ["image/jpeg", "image/png", "image/webp"];
      if (!allowedTypes.includes(avatarFile.type)) {
        nextErrors.avatar_file = true;
        messages.push("Avatar must be a valid image file (JPG, PNG, WEBP).");
      } else if (avatarFile.size > 5 * 1024 * 1024) {
        nextErrors.avatar_file = true;
        messages.push("Avatar file size must be less than 5MB.");
      }
    }

    if (formData.password) {
      if (formData.password.length < 8) {
        nextErrors.password = true;
        messages.push("Password must be at least 8 characters long.");
      }
      if (!/[A-Z]/.test(formData.password)) {
        nextErrors.password = true;
        messages.push("Password must contain at least one uppercase letter (A-Z).");
      }
      if (!/\d/.test(formData.password)) {
        nextErrors.password = true;
        messages.push("Password must contain at least one digit (0-9).");
      }
      if (!/[!@#$%^&*(),.?":{}|<>]/.test(formData.password)) {
        nextErrors.password = true;
        messages.push("Password must contain at least one special character.");
      }
      if (/\s/.test(formData.password)) {
        nextErrors.password = true;
        messages.push("Password must not contain whitespace characters.");
      }
    }

    setFieldErrors(nextErrors);
    if (messages.length) {
      setError(messages.join(" "));
      return false;
    }
    return true;
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

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setAvatarFile(file);
      setAvatarPreview(URL.createObjectURL(file));
      setFieldErrors((prev) => {
        const next = { ...prev };
        delete next.avatar_file;
        return next;
      });
    }
  };

  const handleEdit = () => {
    syncFormFromProfile(profile);
    setFieldErrors({});
    setError("");
    setSuccess("");
    setIsEditing(true);
  };

  const handleBack = () => {
    syncFormFromProfile(profile);
    setFieldErrors({});
    setError("");
    setIsEditing(false);
  };

  const handleUpdateProfile = async (event) => {
    event.preventDefault();
    setError("");
    setSuccess("");

    if (!validateProfileForm()) return;

    const payload = new FormData();
    let hasChanges = false;
    
    const nextUsername = formData.username.trim();
    const nextDisplayName = formData.display_name.trim();
    const nextEmail = formData.email.trim();

    if (nextUsername !== profile.username) { payload.append("username", nextUsername); hasChanges = true; }
    if (nextDisplayName !== profile.display_name) { payload.append("display_name", nextDisplayName); hasChanges = true; }
    if (nextEmail !== profile.email) { payload.append("email", nextEmail); hasChanges = true; }
    if (formData.password) { payload.append("password", formData.password); hasChanges = true; }
    if (avatarFile) { payload.append("avatar", avatarFile); hasChanges = true; }

    if (!hasChanges) {
      setError("No profile fields changed.");
      return;
    }

    setIsSubmitting(true);
    try {
      const result = await updateUserProfile(userId, payload);
      const updatedProfile = {
        ...profile,
        username: nextUsername,
        display_name: nextDisplayName,
        email: nextEmail,
        avatar_url: result.data?.avatar_url || profile.avatar_url,
      };
      setProfile(updatedProfile);
      syncFormFromProfile(updatedProfile);
      if (nextUsername !== profile.username) localStorage.setItem("username", nextUsername);
      setSuccess(result.message || "Update successful");
      setIsEditing(false);
      if (onProfileUpdate) {
        onProfileUpdate(nextUsername, updatedProfile.avatar_url);
      }
    } catch (err) {
      setError(err.message || "Update user profile failed");
    } finally {
      setIsSubmitting(false);
    }
  };

  // Logic nâng cấp tài khoản lên Contributor
  const handleBecomeContributor = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess("");
    setIsSubmitting(true);

    try {
      const response = await fetch("http://localhost:21081/api/users/make-contributor", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: Number(userId),
          secret_key: contributorSecretKey,
        }),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || "Verification failed");
      }

      localStorage.setItem("user_role", "contributor");
      setSuccess(result.message);
      setContributorSecretKey("");

      setTimeout(() => {
        window.location.reload();
      }, 1500);
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setIsSubmitting(false);
    }
  };

  // Logic nâng cấp tài khoản lên Admin
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

      localStorage.setItem("user_role", "admin");
      setSuccess(result.message);
      setSecretKey("");

      setTimeout(() => {
        window.location.reload();
      }, 1500);
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDeactivateAccount = async () => {
    const confirmed = window.confirm("Deactivate account permanently and delete all associated data?");
    if (!confirmed) return;

    setError("");
    setSuccess("");
    setIsDeactivating(true);
    try {
      await deactivateUserAccount(userId);
      localStorage.removeItem("access_token");
      localStorage.removeItem("refresh_token");
      localStorage.removeItem("username");
      localStorage.removeItem("user_role");
      localStorage.removeItem("user_id");
      localStorage.removeItem("avatar_url");
      navigate("/login", { replace: true });
    } catch (err) {
      setError(err.message || "Account deactivation failed");
    } finally {
      setIsDeactivating(false);
    }
  };

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
      <div className="flex items-center justify-between">
        <div>
          <h2 className={`text-2xl font-bold tracking-wide ${tone.title}`}>USER PROFILE</h2>
          <p className={`mt-1 text-sm ${tone.muted}`}>Manage your personal details and account privileges.</p>
        </div>
        {profile && (
          <div className={`flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center text-sm font-bold border ${
            isLight
              ? "bg-slate-100 border-slate-200 text-slate-600"
              : "bg-slate-800 border-white/10 text-slate-300"
          }`}>
            {profile.avatar_url ? (
              <img
                src={getAvatarUrl(profile.avatar_url)}
                alt=""
                className="h-full w-full rounded-full object-cover"
              />
            ) : initials}
          </div>
        )}
      </div>

      {(success || error) && (
        <div className={`rounded-lg border p-3 text-xs ${
          error
            ? (isLight ? "border-red-200 bg-red-50 text-red-700" : "border-red-500/30 bg-red-500/10 text-red-300")
            : (isLight ? "border-emerald-300 bg-emerald-50 text-emerald-700" : "border-emerald-500/30 bg-emerald-500/10 text-emerald-300")
        }`}>
          {error || success}
        </div>
      )}

      {profile && (
        <div className={`rounded-lg border ${tone.panel}`}>
          <div className={`px-6 py-3 border-b text-xs font-semibold uppercase tracking-widest ${tone.muted} ${isLight ? "border-slate-100" : "border-white/5"}`}>
            Personal Information
          </div>

          {!isEditing ? (
            <>
              <div className={`grid grid-cols-2 md:grid-cols-4 divide-y md:divide-y-0 md:divide-x ${isLight ? "divide-slate-100" : "divide-white/5"}`}>
                {[
                  { label: "Username", value: `@${profile.username}`, mono: true },
                  { label: "Display Name", value: profile.display_name, mono: false },
                  { label: "Email", value: profile.email, mono: true, small: true },
                  { label: "Avatar", value: profile.avatar_url ? "Custom" : "Initials", mono: false },
                ].map(({ label, value, mono, small }) => (
                  <div key={label} className="px-6 py-4 space-y-1">
                    <p className={`text-xs uppercase font-semibold ${tone.muted}`}>{label}</p>
                    <p className={`font-semibold ${small ? "text-xs" : "text-sm"} ${mono ? "font-mono" : ""} ${tone.value} truncate`}>
                      {value}
                    </p>
                  </div>
                ))}
              </div>

              <div className={`px-6 py-3 flex items-center justify-between border-t ${isLight ? "border-slate-100 bg-slate-50/50" : "border-white/5 bg-white/[0.02]"}`}>
                <span className={`text-xs uppercase font-semibold ${tone.muted}`}>Account Role</span>
                <span className={`text-xs font-bold uppercase px-3 py-1 rounded-full border ${
                  profile.role === "admin"
                    ? "text-rose-400 bg-rose-500/10 border-rose-500/20"
                    : profile.role === "contributor"
                    ? "text-amber-400 bg-amber-500/10 border-amber-500/20"
                    : "text-cyan-400 bg-cyan-500/10 border-cyan-500/20"
                }`}>
                  {profile.role}
                </span>
              </div>

              <div className={`px-6 py-4 border-t ${isLight ? "border-slate-100" : "border-white/5"}`}>
                <button
                  type="button"
                  onClick={handleEdit}
                  className="rounded-lg bg-emerald-600 px-4 py-2.5 text-xs font-semibold text-white transition hover:bg-emerald-500"
                >
                  Edit Profile
                </button>
              </div>
            </>
          ) : (
            <form onSubmit={handleUpdateProfile} className="px-6 py-5 space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <label className={`block text-xs font-semibold uppercase ${tone.muted}`}>
                  Username
                  <input
                    name="username"
                    value={formData.username}
                    onChange={handleChange}
                    className={getInputClass("username")}
                    aria-invalid={fieldErrors.username ? "true" : "false"}
                  />
                </label>
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
              </div>
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
              <label className={`block text-xs font-semibold uppercase ${tone.muted}`}>
                Avatar
                <div className="mt-1.5 flex items-center gap-4">
                  {avatarPreview ? (
                    <img src={avatarPreview} alt="Avatar preview" className="w-12 h-12 rounded-full object-cover border border-slate-200" />
                  ) : (
                    <div className="w-12 h-12 rounded-full bg-slate-200 flex items-center justify-center text-slate-400 text-xs font-bold border border-slate-300">
                      {initials}
                    </div>
                  )}
                  <input
                    type="file"
                    accept="image/jpeg, image/png, image/webp"
                    onChange={handleFileChange}
                    ref={fileInputRef}
                    className={`block w-full text-sm ${isLight ? "text-slate-500" : "text-slate-400"}
                      file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0
                      file:text-xs file:font-semibold
                      ${isLight ? "file:bg-emerald-50 file:text-emerald-700 hover:file:bg-emerald-100" : "file:bg-emerald-500/10 file:text-emerald-400 hover:file:bg-emerald-500/20"}
                    `}
                    aria-invalid={fieldErrors.avatar_file ? "true" : "false"}
                  />
                </div>
              </label>
              <label className={`block text-xs font-semibold uppercase ${tone.muted}`}>
                New Password
                <input
                  type="password"
                  name="password"
                  value={formData.password}
                  onChange={handleChange}
                  placeholder="Leave blank to keep current password"
                  className={getInputClass("password")}
                  aria-invalid={fieldErrors.password ? "true" : "false"}
                />
              </label>
              <div className="flex flex-wrap gap-3">
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="rounded-lg bg-emerald-600 px-4 py-2.5 text-xs font-semibold text-white transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isSubmitting ? "UPDATING..." : "UPDATE"}
                </button>
                <button
                  type="button"
                  onClick={handleBack}
                  disabled={isSubmitting}
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
      )}

      {/* ELEVATE PRIVILEGES PANEL */}
      <div className={`rounded-lg border ${tone.panel}`}>
        <div className={`px-6 py-3 border-b text-xs font-semibold uppercase tracking-widest ${tone.muted} ${isLight ? "border-slate-100" : "border-white/5"}`}>
          Elevate Privileges
        </div>

        <div className="px-6 py-5 space-y-6">
          {profile?.role === "user" && (
            <div className="flex flex-col md:flex-row md:items-start gap-6 border-b border-white/5 pb-6">
              <div className="md:w-1/2 space-y-1">
                <p className={`text-sm font-semibold ${tone.body}`}>Become a Contributor</p>
                <p className={`text-xs leading-relaxed ${tone.muted}`}>
                  Enter a contributor invitation code to gain problem creation capabilities and research roadmaps access.
                </p>
              </div>

              <form onSubmit={handleBecomeContributor} className="md:w-1/2 space-y-3">
                <label className={`block text-xs font-semibold uppercase ${tone.muted}`}>
                  Contributor Secret Key
                  <input
                    type="password"
                    required
                    value={contributorSecretKey}
                    onChange={(e) => setContributorSecretKey(e.target.value)}
                    placeholder="Enter contributor key..."
                    className={`mt-1.5 w-full rounded-lg border px-3 py-2.5 text-sm outline-none font-mono ${tone.input}`}
                  />
                </label>
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="w-full rounded-lg bg-amber-600 px-4 py-2.5 text-xs font-semibold text-white transition hover:bg-amber-500 disabled:opacity-60"
                >
                  {isSubmitting ? "VERIFYING..." : "ACTIVATE CONTRIBUTOR PRIVILEGES"}
                </button>
              </form>
            </div>
          )}

          {profile?.role !== "admin" ? (
            <div className="flex flex-col md:flex-row md:items-start gap-6">
              <div className="md:w-1/2 space-y-1">
                <p className={`text-sm font-semibold ${tone.body}`}>Activate Admin Access</p>
                <p className={`text-xs leading-relaxed ${tone.muted}`}>
                  If you have an invitation key or system master token, enter it to gain admin roles to approve requests.
                </p>
              </div>

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
                  className="w-full rounded-lg bg-emerald-600 px-4 py-2.5 text-xs font-semibold text-white transition hover:bg-emerald-500 disabled:opacity-60"
                >
                  {isSubmitting ? "VERIFYING..." : "ACTIVATE ADMIN PRIVILEGES"}
                </button>
              </form>
            </div>
          ) : (
            <div className={`flex items-center gap-4 rounded-lg border p-4 ${
              isLight ? "border-emerald-200 bg-emerald-50/50" : "border-emerald-500/20 bg-emerald-500/5"
            }`}>
              <div className={`text-xs font-bold ${isLight ? "text-emerald-600" : "text-emerald-400"}`}>OK</div>
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

      {/* ACCOUNT DEACTIVATION PANEL */}
      <div className={`rounded-lg border ${isLight ? "border-red-200 bg-red-50/70" : "border-red-500/20 bg-red-500/5"}`}>
        <div className={`px-6 py-3 border-b text-xs font-semibold uppercase tracking-widest ${isLight ? "border-red-100 text-red-700" : "border-red-500/20 text-red-300"}`}>
          Account Deactivation
        </div>
        <div className="px-6 py-5 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <p className={`text-sm ${isLight ? "text-red-700" : "text-red-200"}`}>
            Permanently delete this account and its associated data.
          </p>
          <button
            type="button"
            onClick={handleDeactivateAccount}
            disabled={isDeactivating}
            className="rounded-lg bg-red-600 px-4 py-2.5 text-xs font-semibold text-white transition hover:bg-red-500 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isDeactivating ? "DEACTIVATING..." : "DEACTIVATE ACCOUNT"}
          </button>
        </div>
      </div>
    </section>
  );
}