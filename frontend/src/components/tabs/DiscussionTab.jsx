// src/components/tabs/DiscussionTab.jsx
import React, { useState, useEffect } from "react";

function Avatar({ name = "?" }) {
  const initials = name.split(" ").map(w => w[0]).slice(0, 2).join("").toUpperCase();
  const palette = ["#0891b2","#059669","#7c3aed","#db2777","#d97706","#2563eb"];
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

export default function DiscussionTab({ problemId, isLight = false }) {
  const [comments, setComments] = useState([]);
  const [newComment, setNewComment] = useState("");
  const [replyingTo, setReplyingTo] = useState(null);
  const [replyContent, setReplyContent] = useState("");
  const [editingCommentId, setEditingCommentId] = useState(null);
  const [editContent, setEditContent] = useState("");

  const userId = localStorage.getItem("user_id");
  const userRole = localStorage.getItem("user_role") || "user";

  const loadComments = async () => {
    try {
      const response = await fetch(`http://localhost:21081/api/comments?problem_id=${problemId}&user_id=${userId}`);
      const result = await response.json();
      if (result.status === "success") {
        setComments(result.data);
      }
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    if (problemId) {
      loadComments();
    }
  }, [problemId]);

  const handleAddComment = async (e) => {
    e.preventDefault();
    if (!newComment.trim()) {
      alert("Discussion content cannot be empty!");
      return;
    }
    try {
      const response = await fetch("http://localhost:21081/api/comments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: newComment,
          user_id: Number(userId),
          problem_id: Number(problemId),
        }),
      });
      if (response.ok) {
        setNewComment("");
        loadComments();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleAddReply = async (comment) => {
    if (!replyContent.trim()) {
      alert("Reply content cannot be empty!");
      return;
    }
    try {
      const response = await fetch("http://localhost:21081/api/comments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: replyContent,
          user_id: Number(userId),
          problem_id: Number(problemId),
          parent_id: comment.id,
        }),
      });
      if (response.ok) {
        setReplyContent("");
        setReplyingTo(null);
        loadComments();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleEditComment = async (commentId) => {
    if (!editContent.trim()) {
      alert("Comment content cannot be empty!");
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
        loadComments();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleDeleteComment = async (commentId) => {
    if (!window.confirm("Are you sure you want to delete this comment?")) return;
    try {
      const response = await fetch(`http://localhost:21081/api/comments/${commentId}/delete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: Number(userId),
        }),
      });
      if (response.ok) {
        loadComments();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleVote = async (commentId, type) => {
    try {
      const response = await fetch("http://localhost:21081/api/votes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: Number(userId),
          comment_id: commentId,
          vote_type: type,
        }),
      });
      if (response.ok) {
        loadComments();
      }
    } catch (err) {
      console.error(err);
    }
  };

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
    surface:      "#131b2e",
    surfaceRaised:"#182239",
    border:       "rgba(255,255,255,0.12)",
    borderStrong: "rgba(255,255,255,0.2)",
    accent:       "#06b6d4",
    accentDark:   "#0891b2",
    accentBg:     "rgba(6,182,212,0.08)",
    accentBorder: "rgba(6,182,212,0.3)",
    textPrimary:  "#f1f5f9",
    textSecondary:"#94a3b8",
    textMuted:    "#64748b",
    inputBg:      "#0d1322",
    inputBorder:  "rgba(255,255,255,0.15)",
    shadow:       "0 4px 20px rgba(0,0,0,0.4)",
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

  // Build Tree
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
                      Save
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
                      Cancel
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
                  onClick={() => handleVote(node.id, 1)}
                  style={{ background: "none", border: "none", color: node.user_vote === 1 ? t.accent : t.textMuted, cursor: "pointer", fontSize: 12 }}
                >
                  ▲
                </button>
                <span style={{ fontFamily: "'DM Mono', monospace", fontWeight: 600 }}>{node.score}</span>
                <button
                  onClick={() => handleVote(node.id, -1)}
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
                  placeholder="Enter your reply..."
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
                    Reply
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
                    Cancel
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

      {/* Header bar */}
      <div style={{ borderBottom: `1px solid ${t.border}`, paddingBottom: 10, marginBottom: 20 }}>
        <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: t.textPrimary, textTransform: "uppercase", letterSpacing: "0.08em" }}>
          Discussion & Solution Space
        </h3>
        <p style={{ margin: "2px 0 0", fontSize: 11, color: t.textMuted }}>
          Discuss with the community and resolve algorithm issues.
        </p>
      </div>

      <form onSubmit={handleAddComment} style={{ marginBottom: 20 }}>
        <textarea
          rows={2}
          value={newComment}
          onChange={(e) => setNewComment(e.target.value)}
          placeholder="Enter your question or share your thoughts on this lesson..."
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
          Post Discussion Comment
        </button>
      </form>

      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {rootComments.length === 0 ? (
          <p style={{ margin: 0, fontSize: 12, color: t.textMuted, italic: true }}>
            No discussions yet. Start the first conversation!
          </p>
        ) : (
          rootComments.map((node) => renderCommentNode(node))
        )}
      </div>
    </div>
  );
}