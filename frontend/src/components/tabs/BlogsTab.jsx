// src/components/tabs/BlogsTab.jsx
import React, { useState, useEffect } from "react";

function Avatar({ name = "?" }) {
  const initials = name.split(" ").map(w => w[0]).slice(0, 2).join("").toUpperCase();
  const palette = ["#0891b2", "#059669", "#7c3aed", "#db2777", "#d97706", "#2563eb"];
  const hue = palette[name.charCodeAt(0) % palette.length];
  return (
    <div style={{
      width: 28, height: 28, borderRadius: "50%",
      background: `${hue}22`, border: `1.5px solid ${hue}55`,
      display: "flex", alignItems: "center", justifyContent: "center",
      fontSize: 10, fontWeight: 700, color: hue, flexShrink: 0,
      fontFamily: "'DM Mono', monospace",
    }}>
      {initials}
    </div>
  );
}

export default function BlogsTab({ isLight = false }) {
  const [blogs, setBlogs] = useState([]);
  const [selectedBlog, setSelectedBlog] = useState(null);
  const [comments, setComments] = useState([]);
  const [newCommentContent, setNewCommentContent] = useState("");
  const [replyingTo, setReplyingTo] = useState(null);
  const [replyContent, setReplyContent] = useState("");
  const [editingCommentId, setEditingCommentId] = useState(null);
  const [editContent, setEditContent] = useState("");

  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newTitle, setNewTitle] = useState("");
  const [newContent, setNewContent] = useState("");

  const userId = localStorage.getItem("user_id");
  const userRole = localStorage.getItem("user_role") || "user";

  const loadBlogs = async () => {
    try {
      const response = await fetch(`http://localhost:21081/api/blogs?user_id=${userId}`);
      const result = await response.json();
      if (result.status === "success") {
        setBlogs(result.data);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const loadComments = async (blogId) => {
    try {
      const response = await fetch(`http://localhost:21081/api/comments?blog_id=${blogId}&user_id=${userId}`);
      const result = await response.json();
      if (result.status === "success") {
        setComments(result.data);
      }
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    loadBlogs();
  }, []);

  const handleSelectBlog = (blog) => {
    setSelectedBlog(blog);
    loadComments(blog.id);
  };

  const handleBack = () => {
    setSelectedBlog(null);
    setComments([]);
    setReplyingTo(null);
    setEditingCommentId(null);
  };

  const handleCreateBlog = async (e) => {
    e.preventDefault();
    if (!newTitle.trim() || !newContent.trim()) {
      alert("Tiêu đề và nội dung không được để trống!");
      return;
    }
    try {
      const response = await fetch("http://localhost:21081/api/blogs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: newTitle,
          content: newContent,
          author_id: Number(userId),
        }),
      });
      const result = await response.json();
      if (response.ok) {
        setNewTitle("");
        setNewContent("");
        setShowCreateForm(false);
        loadBlogs();
      } else {
        alert(result.detail || "Gặp lỗi khi tạo bài viết.");
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleVote = async (targetId, type, isComment = false) => {
    const payload = {
      user_id: Number(userId),
      vote_type: type,
    };
    if (isComment) {
      payload.comment_id = targetId;
    } else {
      payload.blog_id = targetId;
    }

    try {
      const response = await fetch("http://localhost:21081/api/votes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (response.ok) {
        if (isComment) {
          loadComments(selectedBlog.id);
        } else {
          const resp = await fetch(`http://localhost:21081/api/blogs/${selectedBlog ? selectedBlog.id : targetId}?user_id=${userId}`);
          const res = await resp.json();
          if (res.status === "success") {
            if (selectedBlog) {
              setSelectedBlog(res.data);
            }
            loadBlogs();
          }
        }
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleAddComment = async (e) => {
    e.preventDefault();
    if (!newCommentContent.trim()) {
      alert("Nội dung thảo luận không thể bỏ trống!");
      return;
    }
    try {
      const response = await fetch("http://localhost:21081/api/comments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: newCommentContent,
          user_id: Number(userId),
          blog_id: selectedBlog.id,
        }),
      });
      if (response.ok) {
        setNewCommentContent("");
        loadComments(selectedBlog.id);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleAddReply = async (comment) => {
    if (!replyContent.trim()) {
      alert("Nội dung phản hồi không thể bỏ trống!");
      return;
    }
    try {
      const response = await fetch("http://localhost:21081/api/comments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: replyContent,
          user_id: Number(userId),
          blog_id: selectedBlog.id,
          parent_id: comment.id,
        }),
      });
      if (response.ok) {
        setReplyContent("");
        setReplyingTo(null);
        loadComments(selectedBlog.id);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleEditComment = async (commentId) => {
    if (!editContent.trim()) {
      alert("Nội dung bình luận không thể bỏ trống!");
      return;
    }
    try {
      const response = await fetch(`http://localhost:21081/api/comments/${commentId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: editContent,
          user_id: Number(userId),
        }),
      });
      if (response.ok) {
        setEditingCommentId(null);
        setEditContent("");
        loadComments(selectedBlog.id);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleDeleteComment = async (commentId) => {
    if (!window.confirm("Bạn có chắc chắn muốn xóa bình luận này?")) return;
    try {
      const response = await fetch(`http://localhost:21081/api/comments/${commentId}/delete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: Number(userId),
        }),
      });
      if (response.ok) {
        loadComments(selectedBlog.id);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const t = isLight ? {
    pageBg: "#f1f5f9",
    surface: "#ffffff",
    surfaceRaised: "#f8fafc",
    border: "#e2e8f0",
    borderStrong: "#cbd5e1",
    accent: "#059669",
    accentDark: "#047857",
    accentBg: "#ecfdf5",
    accentBorder: "#6ee7b7",
    textPrimary: "#0f172a",
    textSecondary: "#475569",
    textMuted: "#94a3b8",
    inputBg: "#ffffff",
    inputBorder: "#cbd5e1",
    shadow: "0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04)",
  } : {
    pageBg: "transparent",
    surface: "#0f172a",
    surfaceRaised: "#111827",
    border: "rgba(255,255,255,0.07)",
    borderStrong: "rgba(255,255,255,0.12)",
    accent: "#06b6d4",
    accentDark: "#0891b2",
    accentBg: "rgba(6,182,212,0.08)",
    accentBorder: "rgba(6,182,212,0.3)",
    textPrimary: "#f1f5f9",
    textSecondary: "#64748b",
    textMuted: "#475569",
    inputBg: "#0c1524",
    inputBorder: "rgba(255,255,255,0.08)",
    shadow: "none",
  };

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

  // Build Comment Tree
  const commentMap = {};
  comments.forEach((c) => {
    commentMap[c.id] = { ...c, children: [] };
  });
  const rootComments = [];
  comments.forEach((c) => {
    if (c.parent_id && commentMap[c.parent_id]) {
      commentMap[c.parent_id].children.push(commentMap[c.id]);
    } else {
      rootComments.push(commentMap[c.id]);
    }
  });

  const renderCommentNode = (node, depth = 0, isLastChild = true) => {
    const isOwner = Number(userId) === node.user_id;
    const isAdmin = node.user_role === "admin";

    return (
      <div key={node.id} style={{ marginLeft: depth > 0 ? 32 : 0, marginTop: 16 }}>
        <div style={{ display: "flex", gap: 12 }}>
          {/* Avatar + Connector */}
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", flexShrink: 0 }}>
            <Avatar name={node.user_name || "?"} />
            {!isLastChild && (
              <div style={{
                width: 1, flex: 1, marginTop: 6,
                background: isLight ? "#e2e8f0" : "rgba(255,255,255,0.06)",
                minHeight: 16,
              }} />
            )}
          </div>

          <div style={{ flex: 1 }}>
            {/* Sender row */}
            <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 6 }}>
              <span style={{ fontSize: 13, fontWeight: 600, color: t.textPrimary }}>
                {node.user_name}
              </span>
              <span style={{
                fontSize: 9, fontWeight: 800, letterSpacing: "0.09em",
                textTransform: "uppercase", padding: "2px 6px", borderRadius: 3,
                background: isAdmin
                  ? (isLight ? "#fff1f2" : "rgba(239,68,68,0.1)")
                  : (isLight ? "#f0f9ff" : "rgba(6,182,212,0.1)"),
                color: isAdmin
                  ? (isLight ? "#be123c" : "#f87171")
                  : (isLight ? "#0369a1" : "#22d3ee"),
                border: `1px solid ${isAdmin
                  ? (isLight ? "#fecdd3" : "rgba(239,68,68,0.25)")
                  : (isLight ? "#bae6fd" : "rgba(6,182,212,0.25)")}`,
              }}>
                {node.user_role}
              </span>
              <span style={{ marginLeft: "auto", fontSize: 10, color: t.textMuted, fontFamily: "'DM Mono', monospace" }}>
                {new Date(node.created_at).toLocaleString("en-GB", {
                  day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit"
                })}
              </span>
            </div>

            {/* Bubble Body */}
            <div style={{
              padding: "11px 15px",
              borderRadius: "4px 12px 12px 12px",
              background: isAdmin
                ? (isLight ? "#f0fdf4" : "rgba(16,185,129,0.07)")
                : (isLight ? "#f8fafc" : "rgba(255,255,255,0.04)"),
              border: `1px solid ${isAdmin
                ? (isLight ? "#bbf7d0" : "rgba(16,185,129,0.18)")
                : (isLight ? "#e2e8f0" : "rgba(255,255,255,0.07)")}`,
              fontSize: 13, lineHeight: 1.7,
              color: isLight ? "#1e293b" : "#cbd5e1",
              position: "relative",
            }}>
              {isAdmin && (
                <div style={{
                  position: "absolute", top: 0, left: 0, bottom: 0,
                  width: 3, borderRadius: "4px 0 0 12px",
                  background: isLight ? "#10b981" : "#059669",
                }} />
              )}

              {editingCommentId === node.id ? (
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  <textarea
                    value={editContent}
                    onChange={(e) => setEditContent(e.target.value)}
                    className="tk-input"
                    rows={2}
                    style={{ ...inputStyle, background: t.surface }}
                  />
                  <div style={{ display: "flex", gap: 8 }}>
                    <button
                      type="button"
                      onClick={() => handleEditComment(node.id)}
                      className="tk-primary"
                      style={{
                        background: t.accent, border: "none", color: "#fff",
                        borderRadius: 5, padding: "5px 12px", fontSize: 11, fontWeight: 700, cursor: "pointer"
                      }}
                    >
                      Lưu
                    </button>
                    <button
                      type="button"
                      onClick={() => { setEditingCommentId(null); setEditContent(""); }}
                      className="tk-ghost"
                      style={{
                        background: "transparent", border: `1px solid ${t.border}`, color: t.textSecondary,
                        borderRadius: 5, padding: "5px 12px", fontSize: 11, fontWeight: 700, cursor: "pointer"
                      }}
                    >
                      Hủy
                    </button>
                  </div>
                </div>
              ) : (
                <div style={{ marginLeft: isAdmin ? 8 : 0, whiteSpace: "pre-wrap" }}>
                  {node.content}
                </div>
              )}
            </div>

            {/* Bubble Actions */}
            <div style={{ mt: 8, display: "flex", alignItems: "center", gap: 14, fontSize: 11, color: t.textMuted, marginTop: 6, paddingLeft: 4 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <button
                  onClick={() => handleVote(node.id, 1, true)}
                  style={{ background: "none", border: "none", color: node.user_vote === 1 ? t.accent : t.textMuted, cursor: "pointer", fontSize: 12 }}
                >
                  ▲
                </button>
                <span style={{ fontFamily: "'DM Mono', monospace", fontWeight: 600 }}>{node.score}</span>
                <button
                  onClick={() => handleVote(node.id, -1, true)}
                  style={{ background: "none", border: "none", color: node.user_vote === -1 ? "#ef4444" : t.textMuted, cursor: "pointer", fontSize: 12 }}
                >
                  ▼
                </button>
              </div>

              <button
                onClick={() => {
                  setReplyingTo(node.id);
                  setReplyContent(depth > 0 ? `@${node.user_name} ` : "");
                }}
                style={{ background: "none", border: "none", color: t.accent, cursor: "pointer", fontWeight: 600 }}
              >
                Reply
              </button>

              {isOwner && (
                <button
                  onClick={() => { setEditingCommentId(node.id); setEditContent(node.content); }}
                  style={{ background: "none", border: "none", color: t.textSecondary, cursor: "pointer" }}
                >
                  Edit
                </button>
              )}

              {(isOwner || isAdmin) && (
                <button
                  onClick={() => handleDeleteComment(node.id)}
                  style={{ background: "none", border: "none", color: "#f43f5e", cursor: "pointer" }}
                >
                  Delete
                </button>
              )}
            </div>

            {/* Reply Input Form */}
            {replyingTo === node.id && (
              <div style={{ marginTop: 12, paddingLeft: 12, borderLeft: `2px solid ${t.accent}33` }}>
                <textarea
                  rows={2}
                  value={replyContent}
                  onChange={(e) => setReplyContent(e.target.value)}
                  placeholder="Nhập nội dung phản hồi..."
                  className="tk-input"
                  style={inputStyle}
                />
                <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
                  <button
                    type="button"
                    onClick={() => handleAddReply(node)}
                    className="tk-primary"
                    style={{
                      background: t.accent, border: "none", color: "#fff",
                      borderRadius: 5, padding: "5px 12px", fontSize: 11, fontWeight: 700, cursor: "pointer"
                    }}
                  >
                    Gửi phản hồi
                  </button>
                  <button
                    type="button"
                    onClick={() => { setReplyingTo(null); setReplyContent(""); }}
                    className="tk-ghost"
                    style={{
                      background: "transparent", border: `1px solid ${t.border}`, color: t.textSecondary,
                      borderRadius: 5, padding: "5px 12px", fontSize: 11, fontWeight: 700, cursor: "pointer"
                    }}
                  >
                    Hủy
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {node.children.map((child, idx) => renderCommentNode(child, depth + 1, idx === node.children.length - 1))}
      </div>
    );
  };

  return (
    <div style={{ fontFamily: "'Inter var', 'Inter', sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');
        .tk-input:focus {
          border-color: ${t.accent} !important;
          box-shadow: 0 0 0 3px ${t.accentBg} !important;
        }
        .tk-primary:hover { background: ${t.accentDark} !important; }
        .tk-ghost:hover { background: ${isLight ? "#f1f5f9" : "rgba(255,255,255,0.05)"} !important; }
      `}</style>

      {/* ── Page Header ── */}
      {!selectedBlog ? (
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
                  <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                  <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
                </svg>
              </div>
              <h2 style={{
                margin: 0, fontSize: 18, fontWeight: 700,
                color: t.textPrimary, letterSpacing: "-0.02em",
              }}>
                Blogs & Articles
              </h2>
            </div>
            <p style={{ margin: 0, fontSize: 12, color: t.textSecondary, lineHeight: 1.5, paddingLeft: 40 }}>
              Discover research roadmaps and learning discussions from the platform community.
            </p>
          </div>

          <button
            onClick={() => setShowCreateForm(!showCreateForm)}
            className="tk-primary"
            style={{
              display: "flex", alignItems: "center", gap: 7,
              background: showCreateForm ? (isLight ? "#f1f5f9" : "rgba(255,255,255,0.06)") : t.accent,
              color: showCreateForm ? t.textSecondary : "#fff",
              border: showCreateForm ? `1px solid ${t.border}` : "none",
              borderRadius: 7, padding: "8px 16px",
              fontSize: 12, fontWeight: 600, letterSpacing: "0.02em",
              cursor: "pointer", transition: "all 0.15s",
              boxShadow: (!showCreateForm && isLight) ? "0 1px 3px rgba(5,150,105,0.3)" : "none",
            }}
          >
            {showCreateForm ? "Cancel" : "New Article"}
          </button>
        </div>
      ) : (
        <div style={{ marginBottom: 20 }}>
          <button
            onClick={handleBack}
            style={{
              background: "transparent", border: "none",
              color: t.accent, fontSize: 12, fontWeight: 600,
              cursor: "pointer", display: "flex", alignItems: "center", gap: 5,
              padding: 0,
            }}
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M9 11L5 7L9 3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            Back to Blogs
          </button>
        </div>
      )}

      {/* ── Create Blog Form ── */}
      {!selectedBlog && showCreateForm && (
        <div style={{
          background: t.surface,
          border: `1px solid ${isLight ? t.accentBorder : t.border}`,
          borderRadius: 12, padding: 24, marginBottom: 24,
          position: "relative", overflow: "hidden",
          boxShadow: isLight ? "0 4px 12px rgba(5,150,105,0.1)" : "none",
        }}>
          <div style={{
            position: "absolute", top: 0, left: 0, right: 0, height: 3,
            background: "linear-gradient(90deg, #10b981, #06b6d4, #818cf8)",
          }} />

          <h3 style={{
            margin: "0 0 20px", fontSize: 12, fontWeight: 700,
            color: isLight ? "#059669" : "#22d3ee",
            textTransform: "uppercase", letterSpacing: "0.1em",
          }}>
            New Shared Article
          </h3>

          <form onSubmit={handleCreateBlog} style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div>
              <label style={labelStyle}>Article Title</label>
              <input
                type="text" required placeholder="Nhập tiêu đề..."
                value={newTitle}
                onChange={(e) => setNewTitle(e.target.value)}
                className="tk-input"
                style={inputStyle}
              />
            </div>
            <div>
              <label style={labelStyle}>Content</label>
              <textarea
                required rows={6} placeholder="Nhập nội dung chia sẻ kỹ thuật chi tiết..."
                value={newContent}
                onChange={(e) => setNewContent(e.target.value)}
                className="tk-input"
                style={{ ...inputStyle, resize: "vertical", lineHeight: 1.6 }}
              />
            </div>
            <div style={{ display: "flex", justifyContent: "flex-end", gap: 10 }}>
              <button
                type="button" className="tk-ghost"
                onClick={() => { setShowCreateForm(false); setNewTitle(""); setNewContent(""); }}
                style={{
                  background: "transparent", border: `1px solid ${t.border}`,
                  color: t.textSecondary, borderRadius: 7, padding: "8px 16px",
                  fontSize: 12, fontWeight: 500, cursor: "pointer",
                }}
              >
                Cancel
              </button>
              <button
                type="submit" className="tk-primary"
                style={{
                  background: t.accent, border: "none", color: "#fff",
                  borderRadius: 7, padding: "8px 20px",
                  fontSize: 12, fontWeight: 600, cursor: "pointer",
                  boxShadow: isLight ? "0 1px 3px rgba(5,150,105,0.35)" : "none",
                }}
              >
                Save Article
              </button>
            </div>
          </form>
        </div>
      )}

      {/* ── Blogs List ── */}
      {!selectedBlog ? (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18 }}>
          {blogs.map((b) => (
            <div
              key={b.id}
              style={{
                background: t.surface,
                border: `1px solid ${t.border}`,
                borderRadius: 12, padding: 20, display: "flex", flexDirection: "column",
                boxShadow: t.shadow,
              }}
            >
              <h3 style={{ margin: "0 0 8px", fontSize: 14, fontWeight: 700, color: t.textPrimary }}>
                {b.title}
              </h3>
              <p style={{ margin: "0 0 12px", fontSize: 11, color: t.textMuted }}>
                Tác giả: {b.author_name} • {new Date(b.created_at).toLocaleDateString()}
              </p>
              <p style={{
                margin: "0 0 16px", fontSize: 12, lineHeight: 1.6, color: t.textSecondary,
                display: "-webkit-box", WebkitLineClamp: 3, WebkitBoxOrient: "vertical", overflow: "hidden"
              }}>
                {b.content}
              </p>

              <div style={{
                display: "flex", alignItems: "center", justifyContent: "space-between",
                borderTop: `1px dashed ${t.border}`, paddingTop: 14, marginTop: "auto"
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <button
                    onClick={() => handleVote(b.id, 1)}
                    style={{ background: "none", border: "none", color: b.user_vote === 1 ? t.accent : t.textMuted, cursor: "pointer", fontSize: 12 }}
                  >
                    ▲
                  </button>
                  <span style={{ fontSize: 11, fontFamily: "'DM Mono', monospace", fontWeight: 600, color: t.textSecondary }}>{b.score}</span>
                  <button
                    onClick={() => handleVote(b.id, -1)}
                    style={{ background: "none", border: "none", color: b.user_vote === -1 ? "#ef4444" : t.textMuted, cursor: "pointer", fontSize: 12 }}
                  >
                    ▼
                  </button>
                </div>

                <button
                  onClick={() => handleSelectBlog(b)}
                  style={{
                    background: isLight ? "#ecfdf5" : "rgba(16,185,129,0.08)",
                    border: `1px solid ${isLight ? "#6ee7b7" : "rgba(16,185,129,0.2)"}`,
                    color: isLight ? "#065f46" : "#34d399",
                    padding: "5px 12px", borderRadius: 6, fontSize: 11, fontWeight: 700,
                    cursor: "pointer", textTransform: "uppercase", letterSpacing: "0.04em"
                  }}
                >
                  Comments
                </button>
              </div>
            </div>
          ))}
        </div>
      ) : (
        /* ── Blog Detail View ── */
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <div style={{
            background: t.surface, border: `1px solid ${t.border}`, borderRadius: 12, padding: 24,
            boxShadow: t.shadow,
          }}>
            <h1 style={{ margin: "0 0 10px", fontSize: 18, fontWeight: 700, color: t.textPrimary }}>
              {selectedBlog.title}
            </h1>
            <p style={{ margin: "0 0 16px", fontSize: 11, color: t.textMuted }}>
              Tác giả: {selectedBlog.author_name} • {new Date(selectedBlog.created_at).toLocaleString()}
            </p>
            <p style={{
              margin: "0 0 20px", fontSize: 13, lineHeight: 1.7, color: t.textSecondary,
              whiteSpace: "pre-wrap", borderLeft: `3px solid ${t.accent}`, paddingLeft: 14
            }}>
              {selectedBlog.content}
            </p>

            <div style={{ display: "flex", alignItems: "center", gap: 10, borderTop: `1px solid ${t.border}`, paddingTop: 14 }}>
              <span style={{ fontSize: 11, color: t.textMuted, fontWeight: 600 }}>Helpful Article?</span>
              <button
                onClick={() => handleVote(selectedBlog.id, 1)}
                style={{ background: "none", border: "none", color: selectedBlog.user_vote === 1 ? t.accent : t.textMuted, cursor: "pointer", fontSize: 12 }}
              >
                ▲ Upvote
              </button>
              <span style={{ fontSize: 12, fontFamily: "'DM Mono', monospace", fontWeight: 700, color: t.textPrimary }}>{selectedBlog.score}</span>
              <button
                onClick={() => handleVote(selectedBlog.id, -1)}
                style={{ background: "none", border: "none", color: selectedBlog.user_vote === -1 ? "#ef4444" : t.textMuted, cursor: "pointer", fontSize: 12 }}
              >
                ▼ Downvote
              </button>
            </div>
          </div>

          {/* Discussion threads */}
          <div style={{
            background: t.surface, border: `1px solid ${t.border}`, borderRadius: 12, padding: 24,
            boxShadow: t.shadow,
          }}>
            <h3 style={{ margin: "0 0 16px", fontSize: 14, fontWeight: 700, color: t.textPrimary, textTransform: "uppercase", letterSpacing: "0.08em" }}>
              Community Discussion
            </h3>

            <form onSubmit={handleAddComment} style={{ marginBottom: 20 }}>
              <textarea
                rows={3} value={newCommentContent}
                onChange={(e) => setNewCommentContent(e.target.value)}
                placeholder="Ý kiến đóng góp, nhận xét cá nhân của bạn..."
                className="tk-input"
                style={{ ...inputStyle, marginBottom: 10 }}
              />
              <button
                type="submit" className="tk-primary"
                style={{
                  background: t.accent, border: "none", color: "#fff",
                  borderRadius: 7, padding: "8px 20px", fontSize: 12, fontWeight: 600,
                  cursor: "pointer", transition: "background 0.15s",
                }}
              >
                Post Comment
              </button>
            </form>

            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {rootComments.length === 0 ? (
                <p style={{ margin: 0, fontSize: 12, color: t.textMuted, italic: true }}>
                  Chưa có bình luận nào cho bài viết này. Hãy là người đầu tiên chia sẻ ý kiến!
                </p>
              ) : (
                rootComments.map((cmt) => renderCommentNode(cmt))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}