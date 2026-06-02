// src/components/tabs/BlogsTab.jsx
import React, { useState, useEffect } from "react";

export default function BlogsTab({ isLight = false }) {
  const [blogs, setBlogs] = useState([]);
  const [selectedBlog, setSelectedBlog] = useState(null);
  const [comments, setComments] = useState([]);
  const [newCommentContent, setNewCommentContent] = useState("");
  const [replyingTo, setReplyingTo] = useState(null); // id của bình luận đang được phản hồi
  const [replyContent, setReplyContent] = useState("");
  const [editingCommentId, setEditingCommentId] = useState(null);
  const [editContent, setEditContent] = useState("");

  // Bài viết mới
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
          // Update selected blog score or refresh list
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

  // Tạo cây bình luận phân cấp
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
      <div key={node.id} className="mt-4" style={{ marginLeft: `${depth * 24}px` }}>
        <div className={`rounded-xl border p-4 ${isLight ? "border-slate-200 bg-white" : "border-white/5 bg-slate-900/40"}`}>
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-2">
              <div className="h-7 w-7 rounded-full bg-emerald-600 flex items-center justify-center font-bold text-white text-xs">
                {node.user_name.slice(0, 2).toUpperCase()}
              </div>
              <div>
                <span className={`text-xs font-semibold ${isLight ? "text-slate-800" : "text-white"}`}>{node.user_name}</span>
                <span className="text-[10px] ml-2 text-slate-400 capitalize">{node.user_role}</span>
              </div>
            </div>
            <span className="text-[10px] text-slate-400">{new Date(node.created_at).toLocaleString()}</span>
          </div>

          {editingCommentId === node.id ? (
            <div className="mt-2 space-y-2">
              <textarea
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
                className={`w-full text-sm rounded-lg p-2 focus:ring-1 focus:ring-emerald-500 border outline-none ${isLight ? "bg-slate-50 border-slate-300 text-slate-900" : "bg-slate-950 border-white/10 text-white"}`}
              />
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => handleEditComment(node.id)}
                  className="px-3 py-1 bg-emerald-600 hover:bg-emerald-500 text-white rounded text-xs font-semibold"
                >
                  Lưu
                </button>
                <button
                  type="button"
                  onClick={() => { setEditingCommentId(null); setEditContent(""); }}
                  className="px-3 py-1 bg-slate-600 hover:bg-slate-500 text-white rounded text-xs font-semibold"
                >
                  Hủy
                </button>
              </div>
            </div>
          ) : (
            <p className={`mt-2 text-sm whitespace-pre-wrap ${isLight ? "text-slate-700" : "text-slate-200"}`}>
              {node.content}
            </p>
          )}

          {/* Comment Footer (Upvote, Reply, Edit, Delete) */}
          <div className="mt-3 flex items-center gap-4 text-xs">
            <div className="flex items-center gap-1">
              <button
                onClick={() => handleVote(node.id, 1, true)}
                className={`hover:text-emerald-400 transition ${node.user_vote === 1 ? "text-emerald-500" : "text-slate-400"}`}
                title="Upvote"
              >
                ▲
              </button>
              <span className="font-mono text-xs">{node.score}</span>
              <button
                onClick={() => handleVote(node.id, -1, true)}
                className={`hover:text-rose-400 transition ${node.user_vote === -1 ? "text-rose-500" : "text-slate-400"}`}
                title="Downvote"
              >
                ▼
              </button>
            </div>

            <button
              onClick={() => {
                setReplyingTo(node.id);
                // Nếu phản hồi comment con (nested reply), tự động gắn thẻ @Username của tác giả ở đầu
                setReplyContent(depth > 0 ? `@${node.user_name} ` : "");
              }}
              className="text-slate-400 hover:text-emerald-400 transition"
            >
              Reply
            </button>

            {isOwner && (
              <button
                onClick={() => { setEditingCommentId(node.id); setEditContent(node.content); }}
                className="text-slate-400 hover:text-emerald-400 transition"
              >
                Edit
              </button>
            )}

            {(isOwner || isAdmin) && (
              <button
                onClick={() => handleDeleteComment(node.id)}
                className="text-slate-400 hover:text-rose-400 transition"
              >
                Delete
              </button>
            )}
          </div>

          {/* Reply Form */}
          {replyingTo === node.id && (
            <div className="mt-3 space-y-2 pl-4 border-l-2 border-emerald-500/30">
              <textarea
                rows={2}
                value={replyContent}
                onChange={(e) => setReplyContent(e.target.value)}
                placeholder="Nhập nội dung phản hồi..."
                className={`w-full text-xs rounded-lg p-2 focus:ring-1 focus:ring-emerald-500 border outline-none ${isLight ? "bg-slate-50 border-slate-300 text-slate-900" : "bg-slate-950 border-white/10 text-white"}`}
              />
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => handleAddReply(node)}
                  className="px-3 py-1 bg-emerald-600 hover:bg-emerald-500 text-white rounded text-xs font-semibold"
                >
                  Gửi phản hồi
                </button>
                <button
                  type="button"
                  onClick={() => { setReplyingTo(null); setReplyContent(""); }}
                  className="px-3 py-1 bg-slate-600 hover:bg-slate-500 text-white rounded text-xs font-semibold"
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
    <div className="space-y-6">
      {/* Nút quay lại hoặc thêm bài mới */}
      {!selectedBlog ? (
        <div className="flex justify-between items-center">
          <div>
            <h2 className={`text-2xl font-bold tracking-wide ${isLight ? "text-slate-950" : "text-white"}`}>BLOGS & ARTICLES</h2>
            <p className="text-xs text-slate-400">Khám phá các bài viết nghiên cứu và thảo luận học tập hàng đầu từ cộng đồng.</p>
          </div>
          <button
            onClick={() => setShowCreateForm(!showCreateForm)}
            className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-sm font-semibold transition"
          >
            {showCreateForm ? "Đóng form" : "Tạo bài viết mới"}
          </button>
        </div>
      ) : (
        <button
          onClick={handleBack}
          className="flex items-center gap-2 text-sm text-emerald-500 hover:text-emerald-400 transition font-semibold"
        >
          ← Quay lại danh sách Blogs
        </button>
      )}

      {/* Tạo bài viết mới form */}
      {!selectedBlog && showCreateForm && (
        <form onSubmit={handleCreateBlog} className={`p-6 rounded-xl border space-y-4 ${isLight ? "border-slate-200 bg-white" : "border-white/5 bg-slate-950/40"}`}>
          <h3 className={`text-md font-bold ${isLight ? "text-slate-900" : "text-white"}`}>TẠO BÀI VIẾT CHIA SẺ MỚI</h3>
          <div className="space-y-1">
            <label className="text-xs text-slate-400 font-semibold block">TIÊU ĐỀ BÀI VIẾT</label>
            <input
              type="text"
              required
              placeholder="Nhập tiêu đề..."
              value={newTitle}
              onChange={(e) => setNewTitle(e.target.value)}
              className={`w-full text-sm rounded-lg p-2 focus:ring-1 focus:ring-emerald-500 border outline-none ${isLight ? "bg-slate-50 border-slate-300 text-slate-900" : "bg-slate-950 border-white/10 text-white"}`}
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs text-slate-400 font-semibold block">NỘI DUNG</label>
            <textarea
              required
              rows={6}
              placeholder="Nhập nội dung chia sẻ kỹ thuật chi tiết..."
              value={newContent}
              onChange={(e) => setNewContent(e.target.value)}
              className={`w-full text-sm rounded-lg p-2 focus:ring-1 focus:ring-emerald-500 border outline-none ${isLight ? "bg-slate-50 border-slate-300 text-slate-900" : "bg-slate-950 border-white/10 text-white"}`}
            />
          </div>
          <div className="flex gap-2">
            <button
              type="submit"
              className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-bold transition"
            >
              Lưu bài viết
            </button>
            <button
              type="button"
              onClick={() => { setShowCreateForm(false); setNewTitle(""); setNewContent(""); }}
              className="px-4 py-2 bg-slate-600 hover:bg-slate-500 text-white rounded-lg text-xs font-bold transition"
            >
              Hủy
            </button>
          </div>
        </form>
      )}

      {/* Trang danh sách các Blog */}
      {!selectedBlog ? (
        <div className="grid gap-4 md:grid-cols-2">
          {blogs.map((b) => (
            <div
              key={b.id}
              className={`p-5 rounded-xl border flex flex-col justify-between transition hover:-translate-y-0.5 duration-200 ${isLight ? "border-slate-200 bg-white hover:border-slate-300" : "border-white/5 bg-slate-900/40 hover:border-white/10"}`}
            >
              <div>
                <h3 className={`text-md font-bold mb-2 ${isLight ? "text-slate-900" : "text-white"}`}>{b.title}</h3>
                <p className="text-xs text-slate-400 mb-4 flex items-center gap-2">
                  <span>Tác giả: {b.author_name}</span>
                  <span>•</span>
                  <span>{new Date(b.created_at).toLocaleDateString()}</span>
                </p>
                <p className={`text-sm line-clamp-3 mb-4 ${isLight ? "text-slate-600" : "text-slate-300"}`}>
                  {b.content}
                </p>
              </div>

              <div className="flex items-center justify-between border-t border-dashed border-slate-700/20 pt-4 mt-auto">
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleVote(b.id, 1)}
                    className={`hover:text-emerald-400 text-sm transition ${b.user_vote === 1 ? "text-emerald-500" : "text-slate-400"}`}
                  >
                    ▲
                  </button>
                  <span className="font-mono text-xs">{b.score}</span>
                  <button
                    onClick={() => handleVote(b.id, -1)}
                    className={`hover:text-rose-400 text-sm transition ${b.user_vote === -1 ? "text-rose-500" : "text-slate-400"}`}
                  >
                    ▼
                  </button>
                </div>

                <button
                  onClick={() => handleSelectBlog(b)}
                  className="px-3 py-1.5 bg-emerald-600/10 hover:bg-emerald-600 text-emerald-500 hover:text-white rounded-lg text-xs font-semibold transition"
                >
                  Bình luận (Comments)
                </button>
              </div>
            </div>
          ))}
        </div>
      ) : (
        /* Trang chi tiết bài viết kèm luồng thảo luận */
        <div className="space-y-6">
          <div className={`p-6 rounded-xl border space-y-4 ${isLight ? "border-slate-200 bg-white" : "border-white/5 bg-slate-900/40"}`}>
            <h1 className={`text-2xl font-bold ${isLight ? "text-slate-950" : "text-white"}`}>{selectedBlog.title}</h1>
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <span className="font-semibold text-slate-300">{selectedBlog.author_name}</span>
              <span>•</span>
              <span>{new Date(selectedBlog.created_at).toLocaleString()}</span>
            </div>
            <p className={`text-base leading-relaxed whitespace-pre-wrap ${isLight ? "text-slate-800" : "text-slate-200"}`}>
              {selectedBlog.content}
            </p>

            <div className="flex items-center gap-2 pt-4 border-t border-slate-700/10">
              <span className="text-xs text-slate-400 font-semibold mr-2">Mức độ hữu ích:</span>
              <button
                onClick={() => handleVote(selectedBlog.id, 1)}
                className={`hover:text-emerald-400 text-sm transition ${selectedBlog.user_vote === 1 ? "text-emerald-500" : "text-slate-400"}`}
              >
                ▲ Upvote
              </button>
              <span className="font-mono text-sm px-2 font-bold">{selectedBlog.score}</span>
              <button
                onClick={() => handleVote(selectedBlog.id, -1)}
                className={`hover:text-rose-400 text-sm transition ${selectedBlog.user_vote === -1 ? "text-rose-500" : "text-slate-400"}`}
              >
                ▼ Downvote
              </button>
            </div>
          </div>

          {/* Phần Thảo luận bài viết */}
          <div className="space-y-4">
            <h2 className={`text-lg font-bold tracking-wide ${isLight ? "text-slate-900" : "text-white"}`}>THẢO LUẬN CỘNG ĐỒNG</h2>

            <form onSubmit={handleAddComment} className="space-y-2">
              <textarea
                rows={3}
                value={newCommentContent}
                onChange={(e) => setNewCommentContent(e.target.value)}
                placeholder="Ý kiến đóng góp, nhận xét cá nhân của bạn..."
                className={`w-full text-sm rounded-lg p-3 focus:ring-1 focus:ring-emerald-500 border outline-none ${isLight ? "bg-slate-50 border-slate-300 text-slate-900" : "bg-slate-950 border-white/10 text-white"}`}
              />
              <button
                type="submit"
                className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-bold transition"
              >
                Lưu (Save)
              </button>
            </form>

            {/* Render cây ý kiến đóng góp */}
            <div className="space-y-4">
              {rootComments.length === 0 ? (
                <p className="text-xs text-slate-400 italic">Chưa có bình luận nào cho bài viết này. Hãy là người đầu tiên chia sẻ ý kiến!</p>
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