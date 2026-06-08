// Paste June 03, 2026 - 2:45AM (Fixed Sidebar Height & Sticky Layout)

import React, { useEffect, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import ProblemsTab from "./tabs/ProblemsTab";
import SubmissionsTab from "./tabs/SubmissionsTab";
import UsersTab from "./tabs/UsersTab";
import ResearchTab from "./tabs/ResearchTab";
import ProfileTab from "./tabs/ProfileTab";
import MyRequestsTab from "./tabs/MyRequestsTab";
import AdminQueueTab from "./tabs/AdminQueueTab";

// IMPORT CÁC TAB MỚI CỦA PHẦN QUẢN LÝ CỘNG ĐỒNG
import BlogsTab from "./tabs/BlogsTab";
import TicketsTab from "./tabs/TicketsTab";

function IntroTab({ isLight, onNavigate, username, userRole }) {
  const [stats, setStats] = useState({ problems: 0, submissions: 0, users: 0, contests: 0 });
  const [loadingStats, setLoadingStats] = useState(true);

  const t = {
    pageBg: isLight ? "#eaf2f0" : "#090d16",
    surface: isLight ? "#ffffff" : "#131b2e",
    surfaceAlt: isLight ? "#f0faf6" : "#182239",
    border: isLight ? "rgba(15,23,42,0.10)" : "rgba(255,255,255,0.12)",
    accent: isLight ? "#059669" : "#10B981",
    accentDim: isLight ? "rgba(5,150,105,0.12)" : "rgba(16,185,129,0.12)",
    accentBorder: isLight ? "rgba(5,150,105,0.30)" : "rgba(16,185,129,0.35)",
    textPrimary: isLight ? "#0f172a" : "#F8FAFC",
    textSecondary: isLight ? "#475569" : "#94A3B8",
    textMuted: isLight ? "#94a3b8" : "#64748B",
    shadow: isLight ? "0 1px 10px rgba(15,23,42,0.08)" : "0 4px 20px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05)",
  };

  useEffect(() => {
    async function loadStats() {
      try {
        const token = localStorage.getItem("access_token");
        const headers = token ? { Authorization: `Bearer ${token}` } : {};
        const [probRes, subRes, userRes] = await Promise.allSettled([
          fetch("http://localhost:21081/api/problems/filter?filter_mode=all", { headers }),
          fetch("http://localhost:21081/api/submissions", { headers }),
          fetch("http://localhost:21081/api/admin/users", { headers }),
        ]);

        const prob = probRes.status === "fulfilled" && probRes.value.ok ? await probRes.value.json() : null;
        const subs = subRes.status === "fulfilled" && subRes.value.ok ? await subRes.value.json() : null;
        const usrs = userRes.status === "fulfilled" && userRes.value.ok ? await userRes.value.json() : null;

        const probCount = prob?.data ? prob.data.length : (Array.isArray(prob) ? prob.length : 0);
        const subsCount = subs?.data ? subs.data.length : (Array.isArray(subs) ? subs.length : 0);
        const usersCount = usrs?.data ? usrs.data.length : (Array.isArray(usrs) ? usrs.length : 0);

        setStats({
          problems: probCount,
          submissions: subsCount,
          users: usersCount,
          contests: 0,
        });
      } catch (_) { /* graceful silence */ }
      finally { setLoadingStats(false); }
    }
    loadStats();
  }, []);

  const hour = new Date().getHours();
  const greeting = hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";

  // Lọc chỉ giữ lại 4 đề mục nhanh khả dụng (đã bỏ Wiki và Contests)
  const quickCards = [
    {
      key: "PROBLEMS", label: "Problems", color: "#6366f1", bg: "rgba(99,102,241,0.10)",
      border: "rgba(99,102,241,0.25)",
      desc: "Solve machine learning & core algorithmic problems",
      icon: (
        <svg width="22" height="22" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.6}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414A1 1 0 0120 9.414V19a2 2 0 01-2 2z" />
        </svg>
      ),
    },
    {
      key: "SUBMISSIONS", label: "Submissions", color: "#10B981", bg: "rgba(16,185,129,0.10)",
      border: "rgba(16,185,129,0.25)",
      desc: "Track your code submission history and execution results",
      icon: (
        <svg width="22" height="22" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.6}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
        </svg>
      ),
    },
    {
      key: "BLOGS", label: "Blogs", color: "#ec4899", bg: "rgba(236,72,153,0.10)",
      border: "rgba(236,72,153,0.25)",
      desc: "Read & share technical expertise with the community",
      icon: (
        <svg width="22" height="22" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.6}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 4a2 2 0 012 2v5a2 2 0 01-2 2z" />
        </svg>
      ),
    },
    {
      key: "TICKETS", label: "Support Tickets", color: "#8b5cf6", bg: "rgba(139,92,246,0.10)",
      border: "rgba(139,92,246,0.25)",
      desc: "Raise technical assistance tickets to administrator staff",
      icon: (
        <svg width="22" height="22" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.6}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M18.364 5.636l-3.536 3.536m0 5.656l3.536 3.536M9.172 9.172L5.636 5.636m3.536 9.192l-3.536 3.536M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-5 0a4 4 0 11-8 0 4 4 0 018 0z" />
        </svg>
      ),
    },
  ];

  const statCards = [
    { label: "Problems", value: stats.problems, icon: "📝", color: "#6366f1", sub: "Total Problems" },
    { label: "Submissions", value: stats.submissions, icon: "🚀", color: "#10B981", sub: "Submissions Run" },
    { label: "Members", value: stats.users, icon: "👥", color: "#f59e0b", sub: "Registered Users" },
    { label: "Contests", value: stats.contests, icon: "🏆", color: "#ec4899", sub: "Active Contests" },
  ];

  return (
    <>
      <style>{`
        .intro-card { transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease; }
        .intro-card:hover { transform: translateY(-3px); }
        .intro-quick-card { transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease; cursor: pointer; }
        .intro-quick-card:hover { transform: translateY(-4px); }
        @keyframes heroPulse {
          0%,100% { opacity: 0.7; } 50% { opacity: 1; }
        }
        @keyframes floatDot {
          0%,100% { transform: translateY(0px); }
          50% { transform: translateY(-8px); }
        }
        .hero-dot { animation: floatDot 3s ease-in-out infinite; }
        .hero-dot:nth-child(2) { animation-delay: 0.5s; }
        .hero-dot:nth-child(3) { animation-delay: 1s; }
        .stat-shimmer { animation: heroPulse 2s ease-in-out infinite; }
      `}</style>

      <div style={{ maxWidth: 1100, margin: "0 auto", display: "flex", flexDirection: "column", gap: 32 }}>

        {/* ── Hero Banner ── */}
        <div
          style={{
            borderRadius: 20,
            overflow: "hidden",
            position: "relative",
            background: isLight
              ? "linear-gradient(135deg, #064e3b 0%, #065f46 40%, #0891b2 100%)"
              : "linear-gradient(135deg, #0a1628 0%, #0d2137 40%, #0a2540 100%)",
            border: `1px solid ${isLight ? "rgba(255,255,255,0.15)" : "rgba(16,185,129,0.15)"}`,
            boxShadow: isLight ? "0 8px 40px rgba(6,78,59,0.25)" : "0 8px 40px rgba(0,0,0,0.5)",
            padding: "48px 48px 40px",
            minHeight: 220,
          }}
        >
          <div style={{ position: "absolute", top: -40, right: -40, width: 260, height: 260, borderRadius: "50%", background: "radial-gradient(circle, rgba(16,185,129,0.18) 0%, transparent 70%)", pointerEvents: "none" }} />
          <div style={{ position: "absolute", bottom: -60, left: 120, width: 200, height: 200, borderRadius: "50%", background: "radial-gradient(circle, rgba(6,182,212,0.12) 0%, transparent 70%)", pointerEvents: "none" }} />

          <div style={{ position: "absolute", top: 32, right: 60, display: "flex", gap: 12 }}>
            {["#10B981", "#06b6d4", "#6366f1"].map((col, i) => (
              <div key={i} className="hero-dot" style={{ width: 10, height: 10, borderRadius: "50%", background: col, opacity: 0.7, animationDelay: `${i * 0.4}s` }} />
            ))}
          </div>

          <div style={{ display: "inline-flex", alignItems: "center", gap: 7, background: "rgba(16,185,129,0.18)", border: "1px solid rgba(16,185,129,0.35)", borderRadius: 20, padding: "4px 12px", marginBottom: 18 }}>
            <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#34d399", display: "inline-block" }} />
            <span style={{ fontSize: 11, fontWeight: 600, color: "#6ee7b7", letterSpacing: "0.08em", fontFamily: "'JetBrains Mono', monospace" }}>ML ONLINE JUDGE</span>
          </div>

          <h1 style={{ fontSize: 32, fontWeight: 700, color: "#fff", marginBottom: 10, lineHeight: 1.25, letterSpacing: "-0.5px" }}>
            {greeting}, <span style={{ color: "#34d399" }}>{username || "Researcher"}</span> 👋
          </h1>
          <p style={{ fontSize: 15, color: "rgba(255,255,255,0.65)", maxWidth: 520, lineHeight: 1.65, marginBottom: 28 }}>
            Welcome back to <strong style={{ color: "rgba(255,255,255,0.85)" }}>Judge Research</strong> — an AI-powered training & research platform designed for the AI community.
          </p>

          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            <button
              onClick={() => onNavigate && onNavigate("PROBLEMS")}
              style={{
                background: "linear-gradient(135deg, #10B981, #059669)",
                border: "none", borderRadius: 10, padding: "10px 22px",
                fontSize: 14, fontWeight: 600, color: "#fff", cursor: "pointer",
                boxShadow: "0 4px 18px rgba(16,185,129,0.35)",
                display: "flex", alignItems: "center", gap: 8,
              }}
            >
              <svg width="15" height="15" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.2}><path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
              Start Practicing
            </button>
            <button
              onClick={() => onNavigate && onNavigate("RESEARCH")}
              style={{
                background: "rgba(255,255,255,0.08)", border: "1px solid rgba(255,255,255,0.2)",
                borderRadius: 10, padding: "10px 22px",
                fontSize: 14, fontWeight: 500, color: "rgba(255,255,255,0.85)", cursor: "pointer",
                display: "flex", alignItems: "center", gap: 8,
              }}
            >
              <svg width="15" height="15" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" /></svg>
              {userRole === "user" ? "Research Roadmaps" : "Research Roadmap"}
            </button>
          </div>

          <div style={{ position: "absolute", top: 20, right: 20 }}>
            <span style={{
              background: "rgba(255,255,255,0.1)", border: "1px solid rgba(255,255,255,0.2)",
              borderRadius: 6, padding: "3px 10px", fontSize: 11, color: "rgba(255,255,255,0.7)",
              fontFamily: "'JetBrains Mono', monospace", fontWeight: 500, textTransform: "uppercase",
            }}>
              {userRole}
            </span>
          </div>
        </div>

        {/* ── Stats Cards ── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}>
          {statCards.map((s) => (
            <div
              key={s.label}
              className="intro-card"
              style={{
                background: t.surface, border: `1px solid ${t.border}`,
                borderRadius: 14, padding: "20px 22px",
                boxShadow: t.shadow,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", justify_content: "space-between", marginBottom: 12 }}>
                <span style={{ fontSize: 22 }}>{s.icon}</span>
                <div style={{
                  width: 8, height: 8, borderRadius: "50%",
                  background: s.color, boxShadow: `0 0 8px ${s.color}66`,
                }} />
              </div>
              <div style={{ fontSize: 28, fontWeight: 700, color: t.textPrimary, fontFamily: "'JetBrains Mono', monospace", lineHeight: 1 }}>
                {loadingStats ? <span className="stat-shimmer" style={{ color: t.textSecondary }}>—</span> : s.value.toLocaleString()}
              </div>
              <div style={{ fontSize: 12, color: t.textSecondary, marginTop: 6 }}>{s.sub}</div>
              <div style={{ marginTop: 10, height: 3, borderRadius: 2, background: t.border, overflow: "hidden" }}>
                <div style={{ height: "100%", width: s.value > 0 ? "100%" : "20%", background: `linear-gradient(90deg, ${s.color}66, ${s.color})`, borderRadius: 2, transition: "width 1s ease" }} />
              </div>
            </div>
          ))}
        </div>

        {/* ── Quick Navigation (Đã cấu hình lại lưới 2x2 cân xứng cho 4 thẻ) ── */}
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
            <div style={{ width: 3, height: 18, borderRadius: 2, background: "linear-gradient(180deg, #10B981, #059669)" }} />
            <span style={{ fontSize: 15, fontWeight: 600, color: t.textPrimary }}>Quick Navigation</span>
            <span style={{ fontSize: 12, color: t.textSecondary, marginLeft: 4 }}>— Choose a module to begin</span>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 14 }}>
            {quickCards.map((card) => (
              <div
                key={card.key}
                className="intro-quick-card"
                onClick={() => onNavigate && onNavigate(card.key)}
                style={{
                  background: t.surface, border: `1px solid ${t.border}`,
                  borderRadius: 14, padding: "20px 20px",
                  boxShadow: t.shadow, position: "relative", overflow: "hidden",
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.borderColor = card.border;
                  e.currentTarget.style.boxShadow = `0 8px 28px ${card.color}18`;
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.borderColor = t.border;
                  e.currentTarget.style.boxShadow = t.shadow;
                }}
              >
                <div style={{ position: "absolute", top: 0, right: 0, width: 80, height: 80, borderRadius: "0 14px 0 80px", background: card.bg, opacity: 0.6 }} />
                <div style={{ display: "flex", alignItems: "flex-start", gap: 14, position: "relative" }}>
                  <div style={{
                    width: 44, height: 44, borderRadius: 12,
                    background: card.bg, border: `1px solid ${card.border}`,
                    display: "flex", alignItems: "center", justify_content: "center",
                    color: card.color, flexShrink: 0,
                  }}>
                    {card.icon}
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 14, fontWeight: 600, color: t.textPrimary, marginBottom: 5 }}>{card.label}</div>
                    <div style={{ fontSize: 12, color: t.textSecondary, lineHeight: 1.5 }}>{card.desc}</div>
                  </div>
                </div>
                <div style={{ marginTop: 14, display: "flex", alignItems: "center", gap: 5, color: card.color, fontSize: 12, fontWeight: 500 }}>
                  <span>Open {card.label}</span>
                  <svg width="13" height="13" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.2}><path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" /></svg>
                </div>
              </div>
            ))}
          </div>
        </div>

      </div>
    </>
  );
}

const TABS = [
  {
    key: "HOME",
    label: "Home",
    icon: (
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 10.5L12 3l9 7.5V20a1 1 0 01-1 1h-5v-6H9v6H4a1 1 0 01-1-1v-9.5z" />
      </svg>
    ),
    component: IntroTab,
  },
  {
    key: "PROBLEMS",
    label: "Problems",
    icon: (
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414A1 1 0 0120 9.414V19a2 2 0 01-2 2z" />
      </svg>
    ),
    component: ProblemsTab,
  },
  {
    key: "SUBMISSIONS",
    label: "Submissions",
    icon: (
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
      </svg>
    ),
    component: SubmissionsTab,
  },
  {
    key: "BLOGS",
    label: "Blogs",
    icon: (
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 4a2 2 0 012 2v5a2 2 0 01-2 2z" />
      </svg>
    ),
    component: BlogsTab,
  },
  {
    key: "TICKETS",
    label: "Support Tickets",
    icon: (
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M18.364 5.636l-3.536 3.536m0 5.656l3.536 3.536M9.172 9.172L5.636 5.636m3.536 9.192l-3.536 3.536M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-5 0a4 4 0 11-8 0 4 4 0 018 0z" />
      </svg>
    ),
    component: TicketsTab,
  },
  {
    key: "MY_REQUESTS",
    label: "My Requests",
    allowedRoles: ["user", "contributor"],
    icon: (
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
      </svg>
    ),
    component: MyRequestsTab,
  },
  {
    key: "ADMIN_QUEUE",
    label: "Pending Queue",
    allowedRoles: ["admin"],
    icon: (
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
      </svg>
    ),
    component: AdminQueueTab,
  },
  {
    key: "PROFILE",
    label: "Profile",
    icon: (
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
      </svg>
    ),
    component: ProfileTab,
  },
  {
    key: "USERS",
    label: "Users",
    allowedRoles: ["admin"],
    icon: (
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
      </svg>
    ),
    component: UsersTab,
  },
  {
    key: "RESEARCH",
    label: "Research",
    allowedRoles: ["admin", "contributor", "user"],
    icon: (
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
    component: ResearchTab,
  },
];

export default function Home() {
  const navigate = useNavigate();
  const location = useLocation();

  const [username, setUsername] = useState("");
  const [avatarUrl, setAvatarUrl] = useState(() => localStorage.getItem("avatar_url") || "");
  const [activeTab, setActiveTab] = useState("HOME");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [theme, setTheme] = useState(() => localStorage.getItem("home_theme") || "dark");
  const [userRole, setUserRole] = useState(() => localStorage.getItem("user_role") || "user");

  const isLight = theme === "light";
  const c = (darkColor, lightColor) => (isLight ? lightColor : darkColor);

  const getAvatarUrl = (url) => {
    if (!url) return null;
    return url.startsWith("/") ? `http://localhost:21081${url}` : url;
  };

  const handleProfileUpdate = (newUsername, newAvatarUrl) => {
    setUsername(newUsername);
    setAvatarUrl(newAvatarUrl);
    localStorage.setItem("username", newUsername);
    if (newAvatarUrl) {
      localStorage.setItem("avatar_url", newAvatarUrl);
    } else {
      localStorage.removeItem("avatar_url");
    }
  };

  useEffect(() => {
    if (location.state?.activeTab) {
      setActiveTab(location.state.activeTab);
    }
  }, [location]);

  useEffect(() => {
    const token = localStorage.getItem("access_token");
    const storedUsername = localStorage.getItem("username");
    const storedRole = localStorage.getItem("user_role");

    if (!token) { navigate("/login"); return; }

    setUsername(storedUsername || "");
    if (storedRole) {
      setUserRole(storedRole);
    }
  }, [navigate]);

  const handleLogout = () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    localStorage.removeItem("username");
    localStorage.removeItem("user_role");
    localStorage.removeItem("user_id");
    localStorage.removeItem("avatar_url");
    navigate("/login");
  };

  const handleToggleTheme = () => {
    setTheme((prev) => {
      const next = prev === "dark" ? "light" : "dark";
      localStorage.setItem("home_theme", next);
      return next;
    });
  };

  const filteredTabs = TABS.filter((tab) => {
    if (tab.allowedRoles) {
      return tab.allowedRoles.includes(userRole);
    }
    return true;
  });

  const activeTabEntry = filteredTabs.find((tab) => tab.key === activeTab) || filteredTabs[0] || TABS[0];
  const ActiveTabComponent = activeTabEntry.component || ProblemsTab;

  const initials = username
    ? username.slice(0, 2).toUpperCase()
    : "JR";

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: ${isLight ? "#eaf2f0" : "#080C14"}; overflow: hidden; }
        .home-shell { font-family: 'Sora', sans-serif; }
        .sidebar-nav-btn { transition: background 0.15s, color 0.15s, border-color 0.15s; }
        .sidebar-nav-btn:hover:not(.active) { background: ${isLight ? "rgba(15,23,42,0.06)" : "rgba(255,255,255,0.05)"} !important; color: ${isLight ? "#0f172a" : "#CBD5E1"} !important; }
        .main-scroll::-webkit-scrollbar { width: 6px; }
        .main-scroll::-webkit-scrollbar-track { background: transparent; }
        .main-scroll::-webkit-scrollbar-thumb { background: ${isLight ? "rgba(15,23,42,0.18)" : "rgba(255,255,255,0.08)"}; border-radius: 3px; }
        .logout-btn { transition: background 0.15s, box-shadow 0.15s; }
        .logout-btn:hover { background: ${isLight ? "#e11d48" : "#E11D48"} !important; color: #fff !important; box-shadow: 0 0 12px rgba(225,29,72,0.35) !important; }
        @keyframes fadeSlide {
          from { opacity: 0; transform: translateY(10px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .tab-content { animation: fadeSlide 0.22s ease both; }
      `}</style>

      <div
        className="home-shell"
        style={{
          display: "flex",
          height: "100vh",
          width: "100vw",
          background: c("#090d16", "#eaf2f0"),
          color: c("#CBD5E1", "#0f172a"),
          overflow: "hidden",
        }}
      >
        <aside
          style={{
            width: sidebarOpen ? 232 : 64,
            minWidth: sidebarOpen ? 232 : 64,
            height: "100vh",
            background: c("#131b2e", "#f8fffc"),
            borderRight: c("1px solid rgba(255,255,255,0.12)", "1px solid rgba(15,23,42,0.12)"),
            display: "flex",
            flexDirection: "column",
            transition: "width 0.22s ease, min-width 0.22s ease",
            overflow: "hidden",
            zIndex: 10,
          }}
        >
          <div
            style={{
              padding: sidebarOpen ? "20px 20px 18px" : "20px 0 18px",
              borderBottom: c("1px solid rgba(255,255,255,0.12)", "1px solid rgba(15,23,42,0.12)"),
              display: "flex",
              alignItems: "center",
              gap: 10,
              justifyContent: sidebarOpen ? "flex-start" : "center",
              flexShrink: 0,
            }}
          >
            <div
              style={{
                width: 32,
                height: 32,
                borderRadius: 8,
                background: "linear-gradient(135deg, #10B981 0%, #059669 100%)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexShrink: 0,
                fontSize: 14,
                fontWeight: 700,
                color: "#fff",
                fontFamily: "'JetBrains Mono', monospace",
              }}
            >
              JR
            </div>
            {sidebarOpen && (
              <div style={{ overflow: "hidden" }}>
                <div style={{ fontSize: 14, fontWeight: 600, color: c("#F1F5F9", "#0f172a"), whiteSpace: "nowrap" }}>
                  Judge Research
                </div>
                <div style={{ fontSize: 11, color: "#475569", whiteSpace: "nowrap" }}>
                  ML Online Judge
                </div>
              </div>
            )}
          </div>

          <nav
            style={{
              flex: 1,
              padding: "12px 10px",
              display: "flex",
              flexDirection: "column",
              gap: 2,
              overflowY: "auto",
            }}
          >
            {filteredTabs.map((tab) => {
              const isActive = activeTab === tab.key;
              return (
                <button
                  key={tab.key}
                  className={`sidebar-nav-btn ${isActive ? "active" : ""}`}
                  onClick={() => setActiveTab(tab.key)}
                  title={!sidebarOpen ? (tab.key === "RESEARCH" && userRole === "user" ? "Roadmaps" : tab.label) : ""}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    padding: sidebarOpen ? "9px 12px" : "9px 0",
                    justifyContent: sidebarOpen ? "flex-start" : "center",
                    borderRadius: 8,
                    border: isActive
                      ? c("1px solid rgba(16,185,129,0.34)", "1px solid rgba(5,150,105,0.35)")
                      : "1px solid transparent",
                    background: isActive
                      ? c("rgba(16,185,129,0.13)", "rgba(16,185,129,0.15)")
                      : "transparent",
                    color: isActive ? c("#34D399", "#047857") : c("#64748B", "#475569"),
                    cursor: "pointer",
                    width: "100%",
                    textAlign: "left",
                    flexShrink: 0,
                  }}
                >
                  <span style={{ flexShrink: 0 }}>{tab.icon}</span>
                  {sidebarOpen && (
                    <span style={{ fontSize: 13, fontWeight: isActive ? 600 : 400, whiteSpace: "nowrap" }}>
                      {tab.key === "RESEARCH" && userRole === "user" ? "Roadmaps" : tab.label}
                    </span>
                  )}
                  {isActive && sidebarOpen && (
                    <span
                      style={{
                        marginLeft: "auto",
                        width: 6,
                        height: 6,
                        borderRadius: "50%",
                        background: c("#34D399", "#10B981"),
                        flexShrink: 0,
                      }}
                    />
                  )}
                </button>
              );
            })}
          </nav>

          <div
            style={{
              borderTop: c("1px solid rgba(255,255,255,0.12)", "1px solid rgba(15,23,42,0.12)"),
              padding: sidebarOpen ? "14px 16px" : "14px 0",
              display: "flex",
              alignItems: "center",
              gap: 10,
              justifyContent: sidebarOpen ? "flex-start" : "center",
              flexShrink: 0,
              marginTop: "auto",
              background: c("#131b2e", "#f8fffc"),
            }}
          >
            <div
              style={{
                width: 32,
                height: 32,
                borderRadius: "50%",
                background: "linear-gradient(135deg, #10B981 0%, #059669 100%)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 12,
                fontWeight: 700,
                color: "#fff",
                flexShrink: 0,
                fontFamily: "'JetBrains Mono', monospace",
                overflow: "hidden"
              }}
            >
              {avatarUrl ? <img src={getAvatarUrl(avatarUrl)} alt="Avatar" style={{ width: '100%', height: '100%', objectFit: 'cover' }} /> : initials}
            </div>
            {sidebarOpen && (
              <div style={{ flex: 1, overflow: "hidden" }}>
                <div style={{ fontSize: 13, fontWeight: 500, color: c("#E2E8F0", "#0f172a"), whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                  {username || "User"}
                </div>
                <div style={{ fontSize: 11, color: "#475569" }}>Member</div>
              </div>
            )}
          </div>
        </aside>

        <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, overflow: "hidden" }}>
          <header
            style={{
              height: 56,
              borderBottom: c("1px solid rgba(255,255,255,0.12)", "1px solid rgba(15,23,42,0.12)"),
              background: c("#131b2e", "#f8fffc"),
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: "0 24px",
              flexShrink: 0,
              gap: 16,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <button
                onClick={() => setSidebarOpen((v) => !v)}
                style={{
                  background: c("rgba(255,255,255,0.03)", "rgba(15,23,42,0.04)"),
                  border: c("1px solid rgba(255,255,255,0.14)", "1px solid rgba(15,23,42,0.16)"),
                  borderRadius: 6,
                  padding: "5px 7px",
                  color: c("#CBD5E1", "#0f172a"),
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                }}
                title="Toggle sidebar"
              >
                <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>

              <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13, color: c("#CBD5E1", "#334155") }}>
                <span>Dashboard</span>
                <span>›</span>
                <span style={{ color: c("#F8FAFC", "#0f172a"), fontWeight: 600 }}>
                  {activeTabEntry.key === "RESEARCH" && userRole === "user" ? "Roadmaps" : activeTabEntry.label}
                </span>
              </div>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <button
                type="button"
                onClick={handleToggleTheme}
                style={{
                  background: c("rgba(255,255,255,0.03)", "rgba(15,23,42,0.04)"),
                  border: c("1px solid rgba(255,255,255,0.14)", "1px solid rgba(15,23,42,0.16)"),
                  borderRadius: 8,
                  padding: "5px 10px",
                  fontSize: 12,
                  color: c("#E2E8F0", "#0f172a"),
                  cursor: "pointer",
                  fontFamily: "'JetBrains Mono', monospace",
                }}
              >
                {isLight ? "Dark" : "Light"}
              </button>

              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  background: c("rgba(16,185,129,0.13)", "rgba(16,185,129,0.12)"),
                  border: c("1px solid rgba(16,185,129,0.35)", "1px solid rgba(16,185,129,0.35)"),
                  borderRadius: 20,
                  padding: "4px 10px",
                  fontSize: 12,
                  color: c("#34D399", "#047857"),
                  fontFamily: "'JetBrains Mono', monospace",
                }}
              >
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: c("#34D399", "#10B981"), display: "inline-block" }} />
                Online
              </div>

              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 7,
                  background: c("rgba(255,255,255,0.07)", "rgba(15,23,42,0.04)"),
                  border: c("1px solid rgba(255,255,255,0.14)", "1px solid rgba(15,23,42,0.16)"),
                  borderRadius: 20,
                  padding: "4px 12px 4px 6px",
                  fontSize: 13,
                  color: c("#F1F5F9", "#0f172a"),
                  fontFamily: "'JetBrains Mono', monospace",
                }}
              >
                <div
                  style={{
                    width: 22,
                    height: 22,
                    borderRadius: "50%",
                    background: "linear-gradient(135deg, #10B981, #059669)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 10,
                    fontWeight: 700,
                    color: "#fff",
                    overflow: "hidden"
                  }}
                >
                  {avatarUrl ? <img src={getAvatarUrl(avatarUrl)} alt="Avatar" style={{ width: '100%', height: '100%', objectFit: 'cover' }} /> : initials}
                </div>
                @{username || "user"}
              </div>

              <button
                className="logout-btn"
                onClick={handleLogout}
                style={{
                  background: c("rgba(225,29,72,0.16)", "rgba(225,29,72,0.10)"),
                  border: c("1px solid rgba(225,29,72,0.44)", "1px solid rgba(225,29,72,0.30)"),
                  borderRadius: 7,
                  padding: "6px 14px",
                  fontSize: 13,
                  fontWeight: 500,
                  color: c("#FDA4AF", "#be123c"),
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                }}
              >
                <svg width="13" height="13" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                </svg>
                Log Out
              </button>
            </div>
          </header>

          <main
            className="main-scroll"
            style={{
              flex: 1,
              overflowY: "auto",
              overflowX: "hidden",
              background: c("#080C14", "#eaf2f0"),
              position: "relative",
              isolation: "isolate",
            }}
          >
            <div
              style={{
                position: "absolute",
                inset: 0,
                backgroundImage: "url('https://www.mun.ca/pharmacy/media/production/memorial/academic/school-of-pharmacy/media-library/images/content/abstract-digital-network-with-molecular-data-science-elements.jpg')",
                backgroundSize: "cover",
                backgroundPosition: "center",
                backgroundRepeat: "no-repeat",
                opacity: isLight ? 0.05 : 0.08,
                pointerEvents: "none",
                zIndex: 0,
              }}
            />
            <div
              style={{
                position: "absolute",
                inset: 0,
                background: isLight
                  ? "radial-gradient(ellipse at 60% 40%, rgba(16,185,129,0.07) 0%, transparent 60%), linear-gradient(180deg, rgba(234,242,240,0.75) 0%, rgba(234,242,240,0.88) 100%)"
                  : "radial-gradient(ellipse at 60% 40%, rgba(16,185,129,0.04) 0%, transparent 60%), linear-gradient(180deg, rgba(8,12,20,0.45) 0%, rgba(8,12,20,0.65) 100%)",
                pointerEvents: "none",
                zIndex: 1,
              }}
            />
            <div
              style={{
                position: "absolute",
                inset: 0,
                backgroundImage: isLight
                  ? "linear-gradient(rgba(15,23,42,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(15,23,42,0.05) 1px, transparent 1px)"
                  : "linear-gradient(rgba(255,255,255,0.01) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.01) 1px, transparent 1px)",
                backgroundSize: "48px 48px",
                pointerEvents: "none",
                zIndex: 2,
              }}
            />
            <div
              key={activeTab}
              className="tab-content"
              style={{ position: "relative", zIndex: 3, padding: "32px 32px 48px" }}
            >
              <ActiveTabComponent
                isLight={isLight}
                onProfileUpdate={handleProfileUpdate}
                onNavigate={setActiveTab}
                username={username}
                userRole={userRole}
              />
            </div>
          </main>
        </div>
      </div>
    </>
  );
}