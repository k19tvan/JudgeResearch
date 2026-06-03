// src/components/tabs/ProfileTab.jsx
import React, { useEffect, useState, useRef } from "react";
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
  const [contributorSecretKey, setContributorSecretKey] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isDeactivating, setIsDeactivating] = useState(false);

  // States quản lý việc ẩn/hiện form nhập mã kích hoạt quyền hạn
  const [showContributorForm, setShowContributorForm] = useState(false);
  const [showAdminForm, setShowAdminForm] = useState(false);

  const userId = localStorage.getItem("user_id");

  const t = isLight ? {
    pageBg:       "#f1f5f9",
    surface:      "#ffffff",
    surfaceRaised:"#f8fafc",
    border:       "#e2e8f0",
    borderStrong: "#cbd5e1",
    accent:       "#059669",
    accentDark:   "#047857",
    accentBg:     "#ecfdf5",
    accentBorder: "#6ee7b7",
    textPrimary:  "#0f172a",
    textSecondary:"#475569",
    textMuted:    "#94a3b8",
    inputBg:      "#ffffff",
    inputBorder:  "#cbd5e1",
    shadow:       "0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04)",
  } : {
    pageBg:       "transparent",
    surface:      "#0f172a",
    surfaceRaised:"#111827",
    border:       "rgba(255,255,255,0.07)",
    borderStrong: "rgba(255,255,255,0.12)",
    accent:       "#06b6d4",
    accentDark:   "#0891b2",
    accentBg:     "rgba(6,182,212,0.08)",
    accentBorder: "rgba(6,182,212,0.3)",
    textPrimary:  "#f1f5f9",
    textSecondary:"#64748b",
    textMuted:    "#475569",
    inputBg:      "#0c1524",
    inputBorder:  "rgba(255,255,255,0.08)",
    shadow:       "none",
  };

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

  const inputStyle = {
    width: "100%", boxSizing: "border-box",
    background: t.inputBg,
    border: `1px solid ${t.inputBorder}`,
    borderRadius: 7, padding: "9px 12px",
    fontSize: 13, color: t.textPrimary,
    outline: "none", transition: "border-color 0.15s, box-shadow 0.15s",
    fontFamily: "inherit",
  };

  const labelStyle = {
    display: "block", fontSize: 11, fontWeight: 600,
    color: t.textSecondary, letterSpacing: "0.06em",
    textTransform: "uppercase", marginBottom: 6,
  };

  if (isLoading) {
    return (
      <div style={{ fontFamily: "'Inter var', 'Inter', sans-serif", color: t.textSecondary, fontSize: 13 }}>
        <p className="animate-pulse">Loading profile...</p>
      </div>
    );
  }

  return (
    <div style={{ fontFamily: "'Inter var', 'Inter', sans-serif", maxWidth: 768, margin: "0 auto" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');
        .tk-input:focus {
          border-color: ${t.accent} !important;
          box-shadow: 0 0 0 3px ${t.accentBg} !important;
        }
        .tk-primary:hover { background: ${t.accentDark} !important; }
        .tk-ghost:hover { background: ${isLight ? "#f1f5f9" : "rgba(255,255,255,0.05)"} !important; }
        
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(-4px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in {
          animation: fadeIn 0.18s ease-out forwards;
        }
      `}</style>

      {/* ── Page Header ── */}
      <div style={{
        display: "flex", alignItems: "flex-start",
        justifyContent: "space-between", marginBottom: 24, gap: 16, flexWrap: "wrap",
      }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 5 }}>
            <div style={{
              width: 30, height: 30, borderRadius: 8,
              background: isLight ? "#059669" : "rgba(6,182,212,0.12)",
              border: isLight ? "none" : "1px solid rgba(6,182,212,0.2)",
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={isLight ? "#fff" : "#22d3ee"} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                <circle cx="12" cy="7" r="4" />
              </svg>
            </div>
            <h2 style={{
              margin: 0, fontSize: 18, fontWeight: 700,
              color: t.textPrimary, letterSpacing: "-0.02em",
            }}>
              User Profile
            </h2>
          </div>
          <p style={{ margin: 0, fontSize: 12, color: t.textSecondary, lineHeight: 1.5, paddingLeft: 40 }}>
            Manage your personal details and account privileges.
          </p>
        </div>

        {profile && (
          <div style={{
            width: 44, height: 44, borderRadius: "50%",
            background: isLight ? "#e2e8f0" : "rgba(255,255,255,0.05)",
            border: `1.5px solid ${t.border}`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 14, fontWeight: 700, color: t.textPrimary, overflow: "hidden", flexShrink: 0
          }}>
            {profile.avatar_url ? (
              <img
                src={getAvatarUrl(profile.avatar_url)}
                alt="Avatar"
                style={{ width: "100%", height: "100%", objectFit: "cover" }}
              />
            ) : initials}
          </div>
        )}
      </div>

      {(success || error) && (
        <div style={{
          marginBottom: 20, padding: "10px 14px", borderRadius: 8,
          background: error ? (isLight ? "#fff1f2" : "rgba(239,68,68,0.08)") : (isLight ? "#ecfdf5" : "rgba(16,185,129,0.08)"),
          border: `1px solid ${error ? (isLight ? "#fecdd3" : "rgba(239,68,68,0.2)") : (isLight ? "#6ee7b7" : "rgba(16,185,129,0.2)")}`,
          color: error ? (isLight ? "#be123c" : "#f87171") : (isLight ? "#065f46" : "#34d399"), fontSize: 12,
        }}>
          {error || success}
        </div>
      )}

      {profile && (
        <div style={{
          background: t.surface,
          border: `1px solid ${t.border}`,
          borderRadius: 12, overflow: "hidden",
          boxShadow: t.shadow, marginBottom: 24,
        }}>
          <div style={{
            padding: "12px 20px", background: isLight ? "#f8fafc" : "rgba(255,255,255,0.02)",
            borderBottom: `1px solid ${t.border}`,
            fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.09em", color: t.textSecondary
          }}>
            Personal Information
          </div>

          {!isEditing ? (
            <div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", borderBottom: `1px solid ${t.border}` }}>
                <div style={{ padding: "16px 20px", borderRight: `1px solid ${t.border}` }}>
                  <span style={{ display: "block", fontSize: 10, fontWeight: 700, textTransform: "uppercase", color: t.textMuted, marginBottom: 4 }}>Username</span>
                  <span style={{ fontSize: 13, fontWeight: 600, color: t.textPrimary, fontFamily: "'DM Mono', monospace" }}>@{profile.username}</span>
                </div>
                <div style={{ padding: "16px 20px" }}>
                  <span style={{ display: "block", fontSize: 10, fontWeight: 700, textTransform: "uppercase", color: t.textMuted, marginBottom: 4 }}>Display Name</span>
                  <span style={{ fontSize: 13, fontWeight: 600, color: t.textPrimary }}>{profile.display_name}</span>
                </div>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", borderBottom: `1px solid ${t.border}` }}>
                <div style={{ padding: "16px 20px", borderRight: `1px solid ${t.border}` }}>
                  <span style={{ display: "block", fontSize: 10, fontWeight: 700, textTransform: "uppercase", color: t.textMuted, marginBottom: 4 }}>Email Address</span>
                  <span style={{ fontSize: 13, fontWeight: 600, color: t.textPrimary, fontFamily: "'DM Mono', monospace" }}>{profile.email}</span>
                </div>
                <div style={{ padding: "16px 20px" }}>
                  <span style={{ display: "block", fontSize: 10, fontWeight: 700, textTransform: "uppercase", color: t.textMuted, marginBottom: 4 }}>Account Role</span>
                  <span style={{
                    display: "inline-flex", padding: "2px 8px", borderRadius: 4, fontSize: 10, fontWeight: 700, textTransform: "uppercase",
                    background: profile.role === "admin" ? (isLight ? "#fff1f2" : "rgba(239,68,68,0.1)") : (profile.role === "contributor" ? (isLight ? "#fffbeb" : "rgba(245,158,11,0.1)") : (isLight ? "#f0f9ff" : "rgba(6,182,212,0.1)")),
                    color: profile.role === "admin" ? (isLight ? "#be123c" : "#f87171") : (profile.role === "contributor" ? (isLight ? "#d97706" : "#fbbf24") : (isLight ? "#0369a1" : "#22d3ee")),
                    border: `1px solid ${profile.role === "admin" ? (isLight ? "#fecdd3" : "rgba(239,68,68,0.2)") : (profile.role === "contributor" ? (isLight ? "#fde68a" : "rgba(245,158,11,0.2)") : (isLight ? "#bae6fd" : "rgba(6,182,212,0.2)"))}`,
                  }}>
                    {profile.role}
                  </span>
                </div>
              </div>
              <div style={{ padding: "16px 20px" }}>
                <button
                  type="button"
                  onClick={handleEdit}
                  className="tk-primary"
                  style={{
                    background: t.accent, border: "none", color: "#fff",
                    borderRadius: 7, padding: "8px 20px",
                    fontSize: 12, fontWeight: 600, cursor: "pointer",
                    transition: "background 0.15s",
                    boxShadow: isLight ? "0 1px 3px rgba(5,150,105,0.35)" : "none",
                  }}
                >
                  Edit Profile
                </button>
              </div>
            </div>
          ) : (
            <form onSubmit={handleUpdateProfile} style={{ padding: 20 }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
                <div>
                  <label style={labelStyle}>Username</label>
                  <input
                    name="username"
                    value={formData.username}
                    onChange={handleChange}
                    className="tk-input"
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={labelStyle}>Display Name</label>
                  <input
                    name="display_name"
                    value={formData.display_name}
                    onChange={handleChange}
                    className="tk-input"
                    style={inputStyle}
                  />
                </div>
              </div>
              <div style={{ marginBottom: 16 }}>
                <label style={labelStyle}>Email</label>
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  className="tk-input"
                  style={inputStyle}
                />
              </div>
              <div style={{ marginBottom: 16 }}>
                <label style={labelStyle}>Avatar</label>
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  {avatarPreview ? (
                    <img src={avatarPreview} alt="Preview" style={{ width: 44, height: 44, borderRadius: "50%", objectFit: "cover", border: `1px solid ${t.border}` }} />
                  ) : (
                    <div style={{ width: 44, height: 44, borderRadius: "50%", background: isLight ? "#f1f5f9" : "rgba(255,255,255,0.05)", border: `1px solid ${t.border}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, fontWeight: 700 }}>
                      {initials}
                    </div>
                  )}
                  <input
                    type="file"
                    accept="image/jpeg, image/png, image/webp"
                    onChange={handleFileChange}
                    ref={fileInputRef}
                    style={{ fontSize: 12, color: t.textSecondary }}
                  />
                </div>
              </div>
              <div style={{ marginBottom: 20 }}>
                <label style={labelStyle}>New Password</label>
                <input
                  type="password"
                  name="password"
                  value={formData.password}
                  onChange={handleChange}
                  placeholder="Leave blank to keep current password"
                  className="tk-input"
                  style={inputStyle}
                />
              </div>
              <div style={{ display: "flex", gap: 10 }}>
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="tk-primary"
                  style={{
                    background: t.accent, border: "none", color: "#fff",
                    borderRadius: 7, padding: "8px 20px",
                    fontSize: 12, fontWeight: 600, cursor: "pointer",
                    transition: "background 0.15s",
                    boxShadow: isLight ? "0 1px 3px rgba(5,150,105,0.35)" : "none",
                  }}
                >
                  {isSubmitting ? "Updating..." : "Update"}
                </button>
                <button
                  type="button"
                  onClick={handleBack}
                  className="tk-ghost"
                  style={{
                    background: "transparent", border: `1px solid ${t.border}`, color: t.textSecondary,
                    borderRadius: 7, padding: "8px 20px",
                    fontSize: 12, fontWeight: 500, cursor: "pointer",
                    transition: "background 0.15s",
                  }}
                >
                  Cancel
                </button>
              </div>
            </form>
          )}
        </div>
      )}

      {/* ── Elevate Privileges Panel (Cân đối dạng ngăn thả mượt mà) ── */}
      <div style={{
        background: t.surface,
        border: `1px solid ${t.border}`,
        borderRadius: 12, overflow: "hidden",
        boxShadow: t.shadow, marginBottom: 24,
      }}>
        <div style={{
          padding: "12px 20px", background: isLight ? "#f8fafc" : "rgba(255,255,255,0.02)",
          borderBottom: `1px solid ${t.border}`,
          fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.09em", color: t.textSecondary
        }}>
          Elevate Privileges
        </div>

        <div style={{ padding: "8px 20px 20px" }}>
          {profile?.role === "user" && (
            <div style={{ 
              borderBottom: `1px solid ${t.border}`, 
              padding: "14px 0",
              marginBottom: 4 
            }}>
              <div 
                onClick={() => setShowContributorForm(!showContributorForm)}
                style={{ 
                  display: "flex", 
                  alignItems: "center", 
                  justifyContent: "space-between", 
                  cursor: "pointer",
                }}
              >
                <div style={{ paddingRight: 16 }}>
                  <p style={{ margin: "0 0 4px", fontSize: 13, fontWeight: 700, color: t.textPrimary }}>
                    Become a Contributor
                  </p>
                  <p style={{ margin: 0, fontSize: 11, color: t.textSecondary, lineHeight: 1.5 }}>
                    Gain problem creation capabilities and research roadmaps access.
                  </p>
                </div>
                <span style={{ 
                  fontSize: 12, 
                  color: t.textSecondary,
                  transform: showContributorForm ? "rotate(180deg)" : "rotate(0deg)",
                  transition: "transform 0.2s ease-in-out",
                  userSelect: "none",
                  padding: "4px 8px"
                }}>
                  ▼
                </span>
              </div>

              {showContributorForm && (
                <form 
                  onSubmit={handleBecomeContributor} 
                  className="animate-fade-in"
                  style={{ 
                    marginTop: 16, 
                    display: "flex", 
                    gap: 16, 
                    alignItems: "flex-end", 
                    flexWrap: "wrap" 
                  }}
                >
                  <div style={{ flex: 1, minWidth: 240 }}>
                    <label style={{ ...labelStyle, marginBottom: 6 }}>Contributor Key</label>
                    <input
                      type="password"
                      required
                      value={contributorSecretKey}
                      onChange={(e) => setContributorSecretKey(e.target.value)}
                      className="tk-input"
                      placeholder="Enter contributor invitation key..."
                      style={inputStyle}
                    />
                  </div>
                  <button
                    type="submit"
                    disabled={isSubmitting}
                    className="tk-primary"
                    style={{
                      background: t.accent, border: "none", color: "#fff",
                      borderRadius: 7, padding: "9px 18px",
                      fontSize: 11, fontWeight: 700, cursor: "pointer",
                      textTransform: "uppercase", letterSpacing: "0.04em"
                    }}
                  >
                    Activate
                  </button>
                </form>
              )}
            </div>
          )}

          {profile?.role !== "admin" ? (
            <div style={{ padding: "14px 0 4px" }}>
              <div 
                onClick={() => setShowAdminForm(!showAdminForm)}
                style={{ 
                  display: "flex", 
                  alignItems: "center", 
                  justifyContent: "space-between", 
                  cursor: "pointer",
                }}
              >
                <div style={{ paddingRight: 16 }}>
                  <p style={{ margin: "0 0 4px", fontSize: 13, fontWeight: 700, color: t.textPrimary }}>
                    Activate Admin Access
                  </p>
                  <p style={{ margin: 0, fontSize: 11, color: t.textSecondary, lineHeight: 1.5 }}>
                    Gain system administrator privileges to review and approve community assets.
                  </p>
                </div>
                <span style={{ 
                  fontSize: 12, 
                  color: t.textSecondary,
                  transform: showAdminForm ? "rotate(180deg)" : "rotate(0deg)",
                  transition: "transform 0.2s ease-in-out",
                  userSelect: "none",
                  padding: "4px 8px"
                }}>
                  ▼
                </span>
              </div>

              {showAdminForm && (
                <form 
                  onSubmit={handleBecomeAdmin} 
                  className="animate-fade-in"
                  style={{ 
                    marginTop: 16, 
                    display: "flex", 
                    gap: 16, 
                    alignItems: "flex-end", 
                    flexWrap: "wrap" 
                  }}
                >
                  <div style={{ flex: 1, minWidth: 240 }}>
                    <label style={{ ...labelStyle, marginBottom: 6 }}>Admin Secret Key</label>
                    <input
                      type="password"
                      required
                      value={secretKey}
                      onChange={(e) => setSecretKey(e.target.value)}
                      className="tk-input"
                      placeholder="Enter admin system master token..."
                      style={inputStyle}
                    />
                  </div>
                  <button
                    type="submit"
                    disabled={isSubmitting}
                    className="tk-primary"
                    style={{
                      background: t.accent, border: "none", color: "#fff",
                      borderRadius: 7, padding: "9px 18px",
                      fontSize: 11, fontWeight: 700, cursor: "pointer",
                      textTransform: "uppercase", letterSpacing: "0.04em"
                    }}
                  >
                    Activate
                  </button>
                </form>
              )}
            </div>
          ) : (
            <div 
              className="animate-fade-in"
              style={{
                display: "flex", alignItems: "center", gap: 12, padding: "12px 16px", borderRadius: 8,
                background: isLight ? "#ecfdf5" : "rgba(16,185,129,0.08)",
                border: `1px solid ${isLight ? "#6ee7b7" : "rgba(16,185,129,0.2)"}`,
                color: isLight ? "#065f46" : "#34d399", marginTop: 8
              }}
            >
              <span style={{ fontSize: 14 }}>🛡️</span>
              <div>
                <p style={{ margin: 0, fontSize: 12, fontWeight: 700, textTransform: "uppercase" }}>Admin Privileges Active</p>
                <p style={{ margin: "2px 0 0", fontSize: 11, opacity: 0.8 }}>You are already a system administrator.</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Deactivation Panel ── */}
      <div style={{
        background: isLight ? "#fff1f2" : "rgba(244,63,94,0.03)",
        border: `1px solid ${isLight ? "#fecdd3" : "rgba(244,63,94,0.15)"}`,
        borderRadius: 12, overflow: "hidden", marginBottom: 24,
      }}>
        <div style={{
          padding: "12px 20px", background: isLight ? "#ffe4e6" : "rgba(244,63,94,0.06)",
          borderBottom: `1px solid ${isLight ? "#fecdd3" : "rgba(244,63,94,0.15)"}`,
          fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.09em",
          color: isLight ? "#be123c" : "#fb7185"
        }}>
          Account Deactivation
        </div>
        <div style={{ padding: 20, display: "flex", justifyContent: "space-between", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
          <p style={{ margin: 0, fontSize: 12, color: isLight ? "#9f1239" : "#fca5a5" }}>
            Permanently delete this account and its associated data.
          </p>
          <button
            type="button"
            onClick={handleDeactivateAccount}
            disabled={isDeactivating}
            style={{
              background: isLight ? "#e11d48" : "transparent",
              border: isLight ? "none" : "1px solid rgba(244,63,94,0.3)",
              color: isLight ? "#fff" : "#fb7185",
              padding: "8px 16px", borderRadius: 7,
              fontSize: 11, fontWeight: 700, letterSpacing: "0.06em",
              cursor: "pointer", transition: "all 0.15s",
              textTransform: "uppercase",
              opacity: isDeactivating ? 0.6 : 1
            }}
          >
            {isDeactivating ? "Deactivating..." : "Deactivate Account"}
          </button>
        </div>
      </div>
    </div>
  );
}