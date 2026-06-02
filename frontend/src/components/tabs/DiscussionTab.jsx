// src/components/tabs/DiscussionTab.jsx
import React, { useState, useEffect } from "react";

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
      alert("Nội dung thảo luận không thể bỏ trống!");
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
        loadComments();
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

  const renderCommentNode = (node, depth = 0) => {
    const isOwner = Number(userId) === node.user_id;
    const isAdmin = userRole === "admin";

    return (
      <div key={node.id} className="mt-4" style={{ marginLeft: `${depth * 20}px` }}>
        <div className={`rounded-lg border p-3 ${isLight ? "border-slate-200 bg-white" : "border-white/5 bg-slate-900/30"}`}>
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-1.5">
              <span className={`font-semibold ${isLight ? "text-slate-800" : "text-emerald-400"}`}>{node.user_name}</span>
              <span className="text-[10px] text-slate-500 capitalize">({node.user_role})</span>
            </div>
            <span className="text-[10px] text-slate-500">{new Date(node.created_at).toLocaleString()}</span>
          </div>

          {editingCommentId === node.id ? (
            <div className="mt-2 space-y-1">
              <textarea
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
                className={`w-full text-xs rounded p-2 focus:ring-1 focus:ring-emerald-500 border outline-none ${isLight ? "bg-slate-50 border-slate-300 text-slate-900" : "bg-slate-950 border-white/10 text-white"}`}
              />
              <div className="flex gap-1.5">
                <button
                  onClick={() => handleEditComment(node.id)}
                  className="px-2 py-1 bg-emerald-600 hover:bg-emerald-500 text-white rounded text-[10px]"
                >
                  Lưu
                </button>
                <button
                  onClick={() => { setEditingCommentId(null); setEditContent(""); }}
                  className="px-2 py-1 bg-slate-600 hover:bg-slate-500 text-white rounded text-[10px]"
                >
                  Hủy
                </button>
              </div>
            </div>
          ) : (
            <p className={`mt-1.5 text-xs whitespace-pre-wrap ${isLight ? "text-slate-700" : "text-slate-200"}`}>{node.content}</p>
          )}

          <div className="mt-2 flex items-center gap-3 text-[10px] text-slate-400">
            <div className="flex items-center gap-1">
              <button
                onClick={() => handleVote(node.id, 1)}
                className={`hover:text-emerald-400 ${node.user_vote === 1 ? "text-emerald-500" : ""}`}
              >
                ▲
              </button>
              <span className="font-mono">{node.score}</span>
              <button
                onClick={() => handleVote(node.id, -1)}
                className={`hover:text-rose-400 ${node.user_vote === -1 ? "text-rose-500" : ""}`}
              >
                ▼
              </button>
            </div>

            <button
              onClick={() => {
                setReplyingTo(node.id);
                setReplyContent(depth > 0 ? `@${node.user_name} ` : "");
              }}
              className="hover:text-emerald-400"
            >
              Phản hồi
            </button>

            {isOwner && (
              <button onClick={() => { setEditingCommentId(node.id); setEditContent(node.content); }} className="hover:text-emerald-400">
                Sửa
              </button>
            )}

            {(isOwner || isAdmin) && (
              <button onClick={() => handleDeleteComment(node.id)} className="hover:text-rose-400">
                Xóa
              </button>
            )}
          </div>

          {replyingTo === node.id && (
            <div className="mt-2 space-y-1.5 pl-3 border-l-2 border-emerald-500/20">
              <textarea
                rows={2}
                value={replyContent}
                onChange={(e) => setReplyContent(e.target.value)}
                placeholder="Nhập nội dung phản hồi..."
                className={`w-full text-xs rounded p-1.5 focus:ring-1 focus:ring-emerald-500 border outline-none ${isLight ? "bg-slate-50 border-slate-300 text-slate-900" : "bg-slate-950 border-white/10 text-white"}`}
              />
              <div className="flex gap-1.5">
                <button
                  onClick={() => handleAddReply(node)}
                  className="px-2.5 py-1 bg-emerald-600 text-white rounded text-[10px] font-semibold"
                >
                  Gửi
                </button>
                <button
                  onClick={() => { setReplyingTo(null); setReplyContent(""); }}
                  className="px-2.5 py-1 bg-slate-600 text-white rounded text-[10px] font-semibold"
                >
                  Hủy
                </button>
              </div>
            </div>
          )}
        </div>
        {node.children.map((child) => renderCommentNode(child, depth + 1))}
      </div>
    );
  };

  return (
    <div className="space-y-4">
      <div className="border-b border-slate-800 pb-2">
        <h3 className={`text-sm font-bold ${isLight ? "text-slate-900" : "text-white"}`}>TRANG THẢO LUẬN & ĐÁP ÁN BÀI TẬP</h3>
        <p className="text-[10px] text-slate-400">Thảo luận cùng cộng đồng, đặt câu hỏi giải quyết lỗi thuật toán.</p>
      </div>

      <form onSubmit={handleAddComment} className="space-y-2">
        <textarea
          rows={2}
          value={newComment}
          onChange={(e) => setNewComment(e.target.value)}
          placeholder="Nhập câu hỏi hoặc chia sẻ ý kiến của bạn về bài học này..."
          className={`w-full text-xs rounded-lg p-2.5 focus:ring-1 focus:ring-emerald-500 border outline-none ${isLight ? "bg-slate-50 border-slate-300 text-slate-900" : "bg-slate-950 border-white/10 text-white"}`}
        />
        <button
          type="submit"
          className="px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded text-xs font-semibold"
        >
          Tạo bình luận mới (Add Comment)
        </button>
      </form>

      <div className="space-y-3 mt-4">
        {rootComments.length === 0 ? (
          <p className="text-xs text-slate-400 italic">Chưa có chủ đề thảo luận nào. Hãy bắt đầu câu hỏi đầu tiên!</p>
        ) : (
          rootComments.map((node) => renderCommentNode(node))
        )}
      </div>
    </div>
  );
}