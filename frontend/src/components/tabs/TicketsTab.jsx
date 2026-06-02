// src/components/tabs/TicketsTab.jsx
import React, { useState, useEffect } from "react";

export default function TicketsTab({ isLight = false }) {
  const [tickets, setTickets] = useState([]);
  const [selectedTicket, setSelectedTicket] = useState(null);
  const [replies, setReplies] = useState([]);
  const [replyText, setReplyText] = useState("");
  
  // Tạo Ticket Mới Form
  const [showForm, setShowForm] = useState(false);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");

  const userId = localStorage.getItem("user_id");
  const userRole = localStorage.getItem("user_role") || "user";

  const loadTickets = async () => {
    try {
      const response = await fetch(`http://localhost:21081/api/tickets?user_id=${userId}`);
      const result = await response.json();
      if (result.status === "success") {
        setTickets(result.data);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const loadTicketDetail = async (ticketId) => {
    try {
      const response = await fetch(`http://localhost:21081/api/tickets/${ticketId}?user_id=${userId}`);
      const result = await response.json();
      if (result.status === "success") {
        setSelectedTicket(result.data);
        setReplies(result.data.replies || []);
      }
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    loadTickets();
  }, []);

  const handleCreateTicket = async (e) => {
    e.preventDefault();
    if (!title.trim() || !description.trim()) {
      alert("Vui lòng điền đầy đủ tiêu đề và nội dung sự cố.");
      return;
    }
    try {
      const response = await fetch("http://localhost:21081/api/tickets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: Number(userId),
          title,
          description,
        }),
      });
      if (response.ok) {
        setTitle("");
        setDescription("");
        setShowForm(false);
        loadTickets();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handlePostReply = async (e) => {
    e.preventDefault();
    if (!replyText.trim()) {
      alert("Nội dung không thể để trống!");
      return;
    }
    try {
      const response = await fetch(`http://localhost:21081/api/tickets/${selectedTicket.id}/replies`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: Number(userId),
          message: replyText,
        }),
      });
      if (response.ok) {
        setReplyText("");
        loadTicketDetail(selectedTicket.id);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleToggleStatus = async (statusValue) => {
    try {
      const response = await fetch(`http://localhost:21081/api/tickets/${selectedTicket.id}/status`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: Number(userId),
          status: statusValue,
        }),
      });
      if (response.ok) {
        loadTicketDetail(selectedTicket.id);
        loadTickets();
      }
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="space-y-6">
      {!selectedTicket ? (
        <div className="flex justify-between items-center">
          <div>
            <h2 className={`text-2xl font-bold tracking-wide ${isLight ? "text-slate-950" : "text-white"}`}>SUPPORT TICKETS</h2>
            <p className="text-xs text-slate-400">Gửi các yêu cầu xử lý lỗi kỹ thuật, dịch vụ máy chủ, hoặc phản hồi đào tạo.</p>
          </div>
          <button
            onClick={() => setShowForm(!showForm)}
            className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-sm font-semibold transition animate-pulse"
          >
            {showForm ? "Đóng form" : "Tạo ticket mới (Add Ticket)"}
          </button>
        </div>
      ) : (
        <button
          onClick={() => setSelectedTicket(null)}
          className="text-emerald-500 hover:text-emerald-400 transition font-semibold text-sm flex items-center gap-2"
        >
          ← Quay lại danh sách hỗ trợ
        </button>
      )}

      {/* Form Tạo Ticket Mới */}
      {!selectedTicket && showForm && (
        <form onSubmit={handleCreateTicket} className={`p-6 rounded-xl border space-y-4 ${isLight ? "border-slate-200 bg-white" : "border-white/5 bg-slate-950/40"}`}>
          <h3 className={`text-md font-bold ${isLight ? "text-slate-900" : "text-white"}`}>TẠO YÊU CẦU HỖ TRỢ KỸ THUẬT</h3>
          <div className="space-y-1">
            <label className="text-xs text-slate-400 font-semibold block">TIÊU ĐỀ SỰ CỐ</label>
            <input
              type="text"
              required
              placeholder="VD: Không thể kết nối tới máy chủ DeepWiki / Lỗi biên dịch TLE..."
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className={`w-full text-sm rounded-lg p-2.5 focus:ring-1 focus:ring-emerald-500 border outline-none ${isLight ? "bg-slate-50 border-slate-300 text-slate-900" : "bg-slate-950 border-white/10 text-white"}`}
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs text-slate-400 font-semibold block">MÔ TẢ CHI TIẾT LỖI GẶP PHẢI</label>
            <textarea
              required
              rows={4}
              placeholder="Vui lòng cung cấp mã lỗi, ảnh chụp màn hình hoặc hướng dẫn tái tạo sự cố..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className={`w-full text-sm rounded-lg p-2.5 focus:ring-1 focus:ring-emerald-500 border outline-none ${isLight ? "bg-slate-50 border-slate-300 text-slate-900" : "bg-slate-950 border-white/10 text-white"}`}
            />
          </div>
          <div className="flex gap-2">
            <button
              type="submit"
              className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-bold transition"
            >
              Lưu (Save)
            </button>
            <button
              type="button"
              onClick={() => { setShowForm(false); setTitle(""); setDescription(""); }}
              className="px-4 py-2 bg-slate-600 hover:bg-slate-500 text-white rounded-lg text-xs font-bold transition"
            >
              Hủy
            </button>
          </div>
        </form>
      )}

      {/* Hiển thị danh sách Ticket */}
      {!selectedTicket ? (
        <div className={`overflow-x-auto rounded-xl border ${isLight ? "border-slate-200 bg-white" : "border-white/5 bg-slate-950/40"}`}>
          <table className="w-full text-left border-collapse min-w-[600px]">
            <thead>
              <tr className={`border-b text-xs font-bold uppercase tracking-wider ${isLight ? "bg-slate-800 text-white" : "bg-slate-950 text-slate-200"}`}>
                <th className="px-4 py-3">Mã Ticket</th>
                <th className="px-4 py-3">Tiêu đề sự cố</th>
                <th className="px-4 py-3">Người yêu cầu</th>
                <th className="px-4 py-3">Trạng thái</th>
                <th className="px-4 py-3">Cập nhật lúc</th>
                <th className="px-4 py-3 text-center">Hành động</th>
              </tr>
            </thead>
            <tbody>
              {tickets.length === 0 ? (
                <tr>
                  <td colSpan={6} className="text-center py-6 text-slate-400 text-xs italic">
                    Chưa ghi nhận yêu cầu hỗ trợ kỹ thuật nào.
                  </td>
                </tr>
              ) : (
                tickets.map((t, idx) => (
                  <tr key={t.id} className={`border-b text-sm ${idx % 2 === 0 ? (isLight ? "bg-white" : "bg-slate-900/10") : (isLight ? "bg-slate-50/50" : "bg-slate-950/25")}`}>
                    <td className="px-4 py-3 font-mono text-xs">#TCK_{t.id}</td>
                    <td className="px-4 py-3 font-semibold">{t.title}</td>
                    <td className="px-4 py-3 text-xs text-slate-300">{t.creator_name}</td>
                    <td className="px-4 py-3 text-xs">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${t.status === "open" ? "bg-emerald-600/20 text-emerald-400" : "bg-slate-700/30 text-slate-400"}`}>
                        {t.status.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-xs text-slate-400">{new Date(t.updated_at || t.created_at).toLocaleString()}</td>
                    <td className="px-4 py-3 text-center">
                      <button
                        onClick={() => loadTicketDetail(t.id)}
                        className="rounded bg-emerald-600 hover:bg-emerald-500 px-3 py-1 text-xs font-semibold text-white transition"
                      >
                        CHI TIẾT
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      ) : (
        /* Giao diện xem chi tiết Ticket & luồng Phản hồi tương tác */
        <div className="grid md:grid-cols-3 gap-6">
          <div className="md:col-span-2 space-y-4">
            <div className={`p-6 rounded-xl border space-y-4 ${isLight ? "border-slate-200 bg-white" : "border-white/5 bg-slate-900/40"}`}>
              <div className="flex justify-between items-start">
                <h1 className={`text-xl font-bold ${isLight ? "text-slate-950" : "text-white"}`}>{selectedTicket.title}</h1>
                <span className={`px-2.5 py-0.5 rounded text-xs font-bold ${selectedTicket.status === "open" ? "bg-emerald-600/20 text-emerald-400" : "bg-slate-700/30 text-slate-400"}`}>
                  {selectedTicket.status.toUpperCase()}
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Gửi bởi: <span className="font-semibold text-slate-300">{selectedTicket.creator_name}</span> • {new Date(selectedTicket.created_at).toLocaleString()}
              </p>
              <div className={`p-4 rounded-lg text-sm border whitespace-pre-wrap ${isLight ? "bg-slate-50 border-slate-200 text-slate-800" : "bg-slate-950 border-white/5 text-slate-200"}`}>
                {selectedTicket.description}
              </div>
            </div>

            {/* Chuỗi tin nhắn trao đổi trong Ticket */}
            <div className="space-y-4">
              <h3 className={`text-md font-bold tracking-wide ${isLight ? "text-slate-900" : "text-white"}`}>LỊCH SỬ TRAO ĐỔI</h3>
              <div className="space-y-3">
                {replies.length === 0 ? (
                  <p className="text-xs text-slate-400 italic">Chưa có phản hồi nào từ kỹ thuật viên.</p>
                ) : (
                  replies.map((reply) => (
                    <div
                      key={reply.id}
                      className={`p-4 rounded-xl border space-y-2 ${reply.replier_role === "admin" ? (isLight ? "bg-emerald-50/50 border-emerald-200" : "bg-emerald-950/10 border-emerald-500/20") : (isLight ? "bg-slate-50 border-slate-200" : "bg-slate-900/25 border-white/5")}`}
                    >
                      <div className="flex justify-between items-center text-xs">
                        <div className="flex items-center gap-1">
                          <span className="font-bold text-slate-300">{reply.replier_name}</span>
                          <span className={`text-[9px] px-1 py-0.2 rounded font-mono uppercase ${reply.replier_role === "admin" ? "bg-rose-500/10 text-rose-400" : "bg-slate-700/20 text-slate-400"}`}>
                            {reply.replier_role}
                          </span>
                        </div>
                        <span className="text-[10px] text-slate-400">{new Date(reply.created_at).toLocaleString()}</span>
                      </div>
                      <p className={`text-sm whitespace-pre-wrap ${isLight ? "text-slate-700" : "text-slate-200"}`}>{reply.message}</p>
                    </div>
                  ))
                )}
              </div>

              {/* Soạn thảo phản hồi */}
              <form onSubmit={handlePostReply} className="space-y-2 pt-2">
                <textarea
                  rows={3}
                  value={replyText}
                  onChange={(e) => setReplyText(e.target.value)}
                  placeholder="Nhập nội dung phản hồi, trao đổi làm rõ thêm vấn đề..."
                  className={`w-full text-sm rounded-lg p-3 focus:ring-1 focus:ring-emerald-500 border outline-none ${isLight ? "bg-slate-50 border-slate-300 text-slate-900" : "bg-slate-950 border-white/10 text-white"}`}
                />
                <button
                  type="submit"
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-bold transition"
                >
                  Phản hồi (Reply)
                </button>
              </form>
            </div>
          </div>

          {/* Sidebar cập nhật trạng thái */}
          <div className="space-y-4">
            <div className={`p-4 rounded-xl border space-y-4 ${isLight ? "border-slate-200 bg-white" : "border-white/5 bg-slate-900/40"}`}>
              <h3 className={`text-xs font-bold uppercase tracking-wider ${isLight ? "text-slate-900" : "text-white"}`}>Quản lý Ticket</h3>
              <p className="text-xs text-slate-400 leading-relaxed">Cập nhật trạng thái xử lý sau khi sự cố kỹ thuật hoặc thắc mắc đã được hỗ trợ thành công.</p>
              
              <div className="flex flex-col gap-2 pt-2">
                <button
                  onClick={() => handleToggleStatus("resolved")}
                  disabled={selectedTicket.status === "resolved"}
                  className="w-full py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 disabled:text-slate-500 text-white rounded text-xs font-bold transition"
                >
                  Đánh dấu Đã xử lý (Resolved)
                </button>
                <button
                  onClick={() => handleToggleStatus("open")}
                  disabled={selectedTicket.status === "open"}
                  className="w-full py-2 bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:text-slate-500 text-white rounded text-xs font-bold transition"
                >
                  Mở lại Ticket (Re-open)
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}