// src/components/tabs/TicketsTab.jsx
import React, { useState, useEffect } from "react";

export default function TicketsTab({ isLight = false }) {
  const [tickets, setTickets] = useState([]);
  const [selectedTicket, setSelectedTicket] = useState(null);
  const [replies, setReplies] = useState([]);
  const [replyText, setReplyText] = useState("");
  
  // Create New Ticket Form
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
      alert("Please fill in both the title and issue description.");
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
      alert("Reply message cannot be empty!");
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
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className={`text-2xl font-bold tracking-wide ${isLight ? "text-slate-800" : "text-white"}`}>SUPPORT TICKETS</h2>
            <p className={`text-xs ${isLight ? "text-slate-500" : "text-slate-400"}`}>Submit technical issue requests, server error reports, or platform curriculum feedback.</p>
          </div>
          <button
            onClick={() => setShowForm(!showForm)}
            className={`rounded-lg px-4 py-2.5 text-xs font-bold tracking-wider text-white transition-all duration-200 active:scale-[0.98] ${
              isLight
                ? "bg-emerald-600 hover:bg-emerald-500 shadow-md shadow-emerald-600/10 hover:shadow-emerald-600/20"
                : "bg-cyan-600 hover:bg-cyan-500 shadow-md shadow-cyan-600/10 hover:shadow-cyan-600/20"
            }`}
          >
            {showForm ? "CANCEL" : "CREATE TICKET"}
          </button>
        </div>
      ) : (
        <button
          onClick={() => setSelectedTicket(null)}
          className={`flex items-center gap-2 text-xs font-bold transition-colors ${
            isLight ? "text-emerald-600 hover:text-emerald-500" : "text-cyan-400 hover:text-cyan-300"
          }`}
        >
          ← Back to Support List
        </button>
      )}

      {/* Form: Create Support Ticket */}
      {!selectedTicket && showForm && (
        <form onSubmit={handleCreateTicket} className={`p-6 rounded-xl border space-y-5 ${isLight ? "border-slate-200 bg-white shadow-sm" : "border-white/5 bg-slate-950/40"}`}>
          <h3 className="text-md font-bold text-white uppercase tracking-wider">Create Technical Support Request</h3>
          
          <div>
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">Issue Title *</label>
            <input
              type="text"
              required
              placeholder="e.g. Cannot connect to DeepWiki server / TLE compilation error..."
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className={`mt-1.5 w-full rounded-lg border px-3 py-2.5 text-sm focus:outline-none transition-all ${
                isLight 
                  ? "border-slate-200 bg-white text-slate-700 focus:border-emerald-500" 
                  : "border-slate-800 bg-slate-900/70 text-slate-100 focus:border-cyan-500"
              }`}
            />
          </div>

          <div>
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">Detailed Description *</label>
            <textarea
              required
              rows={4}
              placeholder="Please provide error codes, screenshots, traceback outputs, or instructions to reproduce the issue..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className={`mt-1.5 w-full rounded-lg border px-3 py-2.5 text-sm focus:outline-none transition-all ${
                isLight 
                  ? "border-slate-200 bg-white text-slate-700 focus:border-emerald-500" 
                  : "border-slate-800 bg-slate-900/70 text-slate-100 focus:border-cyan-500"
              }`}
            />
          </div>

          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={() => { setShowForm(false); setTitle(""); setDescription(""); }}
              className="rounded-lg border border-slate-700 bg-slate-900/60 px-4 py-2 text-xs font-semibold text-slate-300"
            >
              CANCEL
            </button>
            <button
              type="submit"
              className="rounded-lg bg-gradient-to-r from-cyan-500 via-indigo-500 to-emerald-600 px-4 py-2 text-xs font-semibold tracking-wide text-white"
            >
              CREATE
            </button>
          </div>
        </form>
      )}

      {/* Support Tickets Data Table */}
      {!selectedTicket ? (
        <div className={`overflow-x-auto rounded-xl border ${
          isLight ? "border-slate-200/80 bg-white shadow-sm" : "border-white/5 bg-slate-950/40"
        }`}>
          <table className="w-full text-left border-collapse min-w-[700px]">
            <thead>
              <tr className={`text-xs font-bold uppercase tracking-wider border-b ${
                isLight 
                  ? "bg-emerald-600 text-white border-emerald-700/50" 
                  : "bg-emerald-950/60 text-emerald-200 border-emerald-500/20"
              }`}>
                <th className="px-4 py-4 w-28">Ticket ID</th>
                <th className="px-4 py-4">Issue Title</th>
                <th className="px-4 py-4 w-40">Requester</th>
                <th className="px-4 py-4 w-28 text-center">Status</th>
                <th className="px-4 py-4 w-44">Updated At</th>
                <th className="px-4 py-4 w-32 text-center">Action</th>
              </tr>
            </thead>
            <tbody className={`divide-y ${isLight ? "divide-slate-100" : "divide-white/5"}`}>
              {tickets.length === 0 ? (
                <tr>
                  <td colSpan={6} className="text-center py-6 text-slate-400 text-xs italic">
                    No support tickets have been recorded.
                  </td>
                </tr>
              ) : (
                tickets.map((t, idx) => {
                  const rowBg = idx % 2 === 0 
                    ? "bg-transparent" 
                    : (isLight ? "bg-slate-50/30" : "bg-slate-900/10");
                  const status = t.status ? t.status.toLowerCase() : "open";

                  return (
                    <tr 
                      key={t.id} 
                      className={`group text-sm transition-colors duration-150 ${rowBg} ${
                        isLight ? "hover:bg-slate-50" : "hover:bg-slate-900/30"
                      }`}
                    >
                      <td className={`px-4 py-4 text-xs font-mono font-medium ${
                        isLight ? "text-slate-500" : "text-slate-400"
                      }`}>
                        #TCK_{t.id}
                      </td>
                      <td className={`px-4 py-4 font-semibold ${isLight ? "text-slate-700" : "text-slate-100"}`}>
                        {t.title}
                      </td>
                      <td className={`px-4 py-4 text-xs font-medium ${isLight ? "text-slate-500" : "text-slate-400"}`}>
                        {t.creator_name}
                      </td>
                      <td className="px-4 py-4 text-center">
                        <span className={`text-[10px] font-semibold px-2 py-1 rounded-md ${
                          status === "open"
                            ? (isLight ? "bg-emerald-50 text-emerald-600 border border-emerald-200" : "bg-emerald-500/10 text-emerald-400 border border-emerald-500/10")
                            : (isLight ? "bg-slate-100 text-slate-500 border border-slate-200" : "bg-slate-500/10 text-slate-400 border border-slate-500/10")
                        }`}>
                          {status.toUpperCase()}
                        </span>
                      </td>
                      <td className={`px-4 py-4 text-xs ${isLight ? "text-slate-500" : "text-slate-400"}`}>
                        {new Date(t.updated_at || t.created_at).toLocaleString()}
                      </td>
                      <td className="px-4 py-4 text-center">
                        <button
                          onClick={() => loadTicketDetail(t.id)}
                          className={`rounded-md px-3 py-1 text-xs font-bold text-white transition shadow-sm ${
                            isLight ? "bg-emerald-600 hover:bg-emerald-500" : "bg-cyan-600 hover:bg-cyan-500"
                          }`}
                        >
                          DETAILS
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      ) : (
        /* Detailed Ticket View & Technical Correspondence History */
        <div className="grid md:grid-cols-3 gap-6">
          <div className="md:col-span-2 space-y-4">
            <div className={`p-6 rounded-xl border space-y-4 ${isLight ? "border-slate-200 bg-white shadow-sm" : "border-white/5 bg-slate-900/40"}`}>
              <div className="flex justify-between items-start">
                <h1 className={`text-xl font-bold ${isLight ? "text-slate-950" : "text-white"}`}>{selectedTicket.title}</h1>
                <span className={`text-[10px] font-semibold px-2 py-1 rounded-md ${
                  selectedTicket.status === "open"
                    ? (isLight ? "bg-emerald-50 text-emerald-600 border border-emerald-200" : "bg-emerald-500/10 text-emerald-400 border border-emerald-500/10")
                    : (isLight ? "bg-slate-100 text-slate-500 border border-slate-200" : "bg-slate-500/10 text-slate-400 border border-slate-500/10")
                }`}>
                  {selectedTicket.status.toUpperCase()}
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Submitted by: <span className="font-semibold text-slate-300">{selectedTicket.creator_name}</span> • {new Date(selectedTicket.created_at).toLocaleString()}
              </p>
              <div className={`p-4 rounded-lg text-sm border whitespace-pre-wrap ${isLight ? "bg-slate-50 border-slate-200 text-slate-800" : "bg-slate-950 border-white/5 text-slate-200"}`}>
                {selectedTicket.description}
              </div>
            </div>

            {/* Conversation Replies Loop */}
            <div className="space-y-4">
              <h3 className={`text-sm font-bold tracking-wide uppercase ${isLight ? "text-slate-900" : "text-white"}`}>Conversation History</h3>
              <div className="space-y-3">
                {replies.length === 0 ? (
                  <p className="text-xs text-slate-400 italic">No responses from technical administrators yet.</p>
                ) : (
                  replies.map((reply) => {
                    const isAdminReply = reply.replier_role === "admin";
                    const replyBg = isAdminReply
                      ? (isLight ? "bg-emerald-50/50 border-emerald-200" : "bg-emerald-950/10 border-emerald-500/20")
                      : (isLight ? "bg-slate-50 border-slate-200" : "bg-slate-900/25 border-white/5");

                    return (
                      <div key={reply.id} className={`p-4 rounded-xl border space-y-2 ${replyBg}`}>
                        <div className="flex justify-between items-center text-xs">
                          <div className="flex items-center gap-1.5">
                            <span className="font-bold text-slate-300">{reply.replier_name}</span>
                            <span className={`text-[9px] px-1.5 py-0.5 rounded font-mono uppercase ${
                              reply.replier_role === "admin" 
                                ? "bg-rose-500/10 text-rose-400" 
                                : "bg-slate-700/20 text-slate-400"
                            }`}>
                              {reply.replier_role}
                            </span>
                          </div>
                          <span className="text-[10px] text-slate-400">{new Date(reply.created_at).toLocaleString()}</span>
                        </div>
                        <p className={`text-sm whitespace-pre-wrap ${isLight ? "text-slate-700" : "text-slate-200"}`}>{reply.message}</p>
                      </div>
                    );
                  })
                )}
              </div>

              {/* Compose Correspondence Reply */}
              <form onSubmit={handlePostReply} className="space-y-2 pt-2">
                <textarea
                  rows={3}
                  value={replyText}
                  onChange={(e) => setReplyText(e.target.value)}
                  placeholder="Type your response here to clarify the issue or coordinate a resolution..."
                  className={`mt-1.5 w-full rounded-lg border px-3 py-2.5 text-sm focus:outline-none transition-all ${
                    isLight 
                      ? "border-slate-200 bg-white text-slate-700 focus:border-emerald-500" 
                      : "border-slate-800 bg-slate-900/70 text-slate-100 focus:border-cyan-500"
                  }`}
                />
                <button
                  type="submit"
                  className={`rounded-lg px-4 py-2 text-xs font-bold text-white transition ${
                    isLight ? "bg-emerald-600 hover:bg-emerald-500" : "bg-cyan-600 hover:bg-cyan-500"
                  }`}
                >
                  Reply
                </button>
              </form>
            </div>
          </div>

          {/* Sidebar: Status Control Pane */}
          <div className="space-y-4">
            <div className={`p-4 rounded-xl border space-y-4 ${isLight ? "border-slate-200 bg-white shadow-sm" : "border-white/5 bg-slate-900/40"}`}>
              <h3 className={`text-xs font-bold uppercase tracking-wider ${isLight ? "text-slate-900" : "text-white"}`}>Ticket Management</h3>
              <p className="text-xs text-slate-400 leading-relaxed">Update the resolution status once the technical issue, bug, or inquiry has been successfully resolved.</p>
              
              <div className="flex flex-col gap-2 pt-2">
                <button
                  onClick={() => handleToggleStatus("resolved")}
                  disabled={selectedTicket.status === "resolved"}
                  className="w-full py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 disabled:text-slate-500 text-white rounded-lg text-xs font-bold transition shadow-sm"
                >
                  Mark as Resolved
                </button>
                <button
                  onClick={() => handleToggleStatus("open")}
                  disabled={selectedTicket.status === "open"}
                  className="w-full py-2 bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:text-slate-500 text-white rounded-lg text-xs font-bold transition shadow-sm"
                >
                  Re-open Ticket
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}