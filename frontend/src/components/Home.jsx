import React, { useEffect, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom"; // Gộp chung các import từ react-router-dom
import ProblemsTab from "./tabs/ProblemsTab";
import SubmissionsTab from "./tabs/SubmissionsTab";
import UsersTab from "./tabs/UsersTab";
import ContestsTab from "./tabs/ContestsTab";
import WikiTab from "./tabs/WikiTab";
import ResearchTab from "./tabs/ResearchTab";
import ProfileTab from "./tabs/ProfileTab";
import MyRequestsTab from "./tabs/MyRequestsTab";
import AdminQueueTab from "./tabs/AdminQueueTab";

function IntroTab({ isLight }) {
  return (
    <section className={`space-y-4 ${isLight ? "text-slate-700" : "text-slate-200"}`}>
      <h2 className={`text-3xl font-bold tracking-wide ${isLight ? "text-slate-900" : "text-white"}`}>
        Welcome to Judge Research
      </h2>
      <p className={`text-base leading-7 ${isLight ? "text-slate-700" : "text-slate-300"}`}>
        This platform helps you practice machine learning problems, read theory/tutorial materials,
        and solve tasks directly in a live coding workspace.
      </p>
      <div className="grid gap-3 md:grid-cols-3">
        <article className={`rounded-lg border p-4 ${isLight ? "border-slate-300 bg-white/90" : "border-white/10 bg-slate-900/40"}`}>
          <h3 className={`text-sm font-semibold ${isLight ? "text-emerald-700" : "text-cyan-300"}`}>Problems</h3>
          <p className={`mt-2 text-sm ${isLight ? "text-slate-700" : "text-slate-300"}`}>Create and manage problem statements and resources.</p>
        </article>
        <article className={`rounded-lg border p-4 ${isLight ? "border-slate-300 bg-white/90" : "border-white/10 bg-slate-900/40"}`}>
          <h3 className={`text-sm font-semibold ${isLight ? "text-emerald-700" : "text-cyan-300"}`}>Live Coding</h3>
          <p className={`mt-2 text-sm ${isLight ? "text-slate-700" : "text-slate-300"}`}>Open a coding workspace and test solutions quickly.</p>
        </article>
        <article className={`rounded-lg border p-4 ${isLight ? "border-slate-300 bg-white/90" : "border-white/10 bg-slate-900/40"}`}>
          <h3 className={`text-sm font-semibold ${isLight ? "text-emerald-700" : "text-cyan-300"}`}>Research Flow</h3>
          <p className={`mt-2 text-sm ${isLight ? "text-slate-700" : "text-slate-300"}`}>Track submissions, users, and contest activities in one place.</p>
        </article>
      </div>
    </section>
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
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414A1 1 0 0120 9.414V19a2 2 0 01-2 2z"/>
      </svg>
    ),
    component: ProblemsTab,
  },
  {
    key: "SUBMISSIONS",
    label: "Submissions",
    icon: (
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/>
      </svg>
    ),
    component: SubmissionsTab,
  },
  {
    key: "MY_REQUESTS",
    label: "My Requests",
    role: "user", // Phân quyền: chỉ tài khoản user thường mới có tab này
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
    role: "admin", // Phân quyền: chỉ tài khoản admin mới có tab này
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
    icon: (
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z"/>
      </svg>
    ),
    component: UsersTab,
  },
  {
    key: "CONTESTS",
    label: "Contests",
    icon: (
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138z"/>
      </svg>
    ),
    component: ContestsTab,
  },
  {
    key: "WIKI",
    label: "Wiki",
    icon: (
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/>
      </svg>
    ),
    component: WikiTab,
  },
  {
    key: "RESEARCH",
    label: "Research",
    icon: (
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
      </svg>
    ),
    component: ResearchTab,
  },
];

export default function Home() {
  const navigate = useNavigate();
  const location = useLocation();

  // Khai báo các State sạch sẽ (Không lặp lại)
  const [username, setUsername] = useState("");
  const [avatarUrl, setAvatarUrl] = useState(() => localStorage.getItem("avatar_url") || ""); // <--- ADD THIS
  const [activeTab, setActiveTab] = useState("HOME");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [theme, setTheme] = useState(() => localStorage.getItem("home_theme") || "dark");
  const [userRole, setUserRole] = useState(() => localStorage.getItem("user_role") || "user");

  const isLight = theme === "light";
  const c = (darkColor, lightColor) => (isLight ? lightColor : darkColor);


// <--- ADD THIS HELPER FUNCTION --->
  const getAvatarUrl = (url) => {
    if (!url) return null;
    return url.startsWith("/") ? `http://localhost:21081${url}` : url;
  };
  // <--- ADD THIS UPDATE HANDLER --->
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
  // Phục hồi Tab hoạt động dựa vào điều hướng lịch sử ở trang khác
  
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
    navigate("/login");
  };

  const handleToggleTheme = () => {
    setTheme((prev) => {
      const next = prev === "dark" ? "light" : "dark";
      localStorage.setItem("home_theme", next);
      return next;
    });
  };

  // Lọc danh sách Tabs dựa theo phân quyền người dùng
  const filteredTabs = TABS.filter((tab) => {
    if (tab.role) {
      return tab.role === userRole;
    }
    return true;
  });

  const ActiveTabComponent =
    TABS.find((tab) => tab.key === activeTab)?.component || ProblemsTab;

  const initials = username
    ? username.slice(0, 2).toUpperCase()
    : "JR";

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: ${isLight ? "#eaf2f0" : "#080C14"}; }
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
          minHeight: "100vh",
          width: "100vw",
          background: c("#080C14", "#eaf2f0"),
          color: c("#CBD5E1", "#0f172a"),
          overflow: "hidden",
        }}
      >
        <aside
          style={{
            width: sidebarOpen ? 232 : 64,
            minWidth: sidebarOpen ? 232 : 64,
            background: c("#0D1117", "#f8fffc"),
            borderRight: c("1px solid rgba(255,255,255,0.06)", "1px solid rgba(15,23,42,0.12)"),
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
              borderBottom: c("1px solid rgba(255,255,255,0.06)", "1px solid rgba(15,23,42,0.12)"),
              display: "flex",
              alignItems: "center",
              gap: 10,
              justifyContent: sidebarOpen ? "flex-start" : "center",
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

          <nav style={{ flex: 1, padding: "12px 10px", display: "flex", flexDirection: "column", gap: 2 }}>
            {filteredTabs.map((tab) => {
              const isActive = activeTab === tab.key;
              return (
                <button
                  key={tab.key}
                  className={`sidebar-nav-btn ${isActive ? "active" : ""}`}
                  onClick={() => setActiveTab(tab.key)}
                  title={!sidebarOpen ? tab.label : ""}
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
                  }}
                >
                  <span style={{ flexShrink: 0 }}>{tab.icon}</span>
                  {sidebarOpen && (
                    <span style={{ fontSize: 13, fontWeight: isActive ? 600 : 400, whiteSpace: "nowrap" }}>
                      {tab.label}
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
              borderTop: c("1px solid rgba(255,255,255,0.06)", "1px solid rgba(15,23,42,0.12)"),
              padding: sidebarOpen ? "14px 16px" : "14px 0",
              display: "flex",
              alignItems: "center",
              gap: 10,
              justifyContent: sidebarOpen ? "flex-start" : "center",
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
                overflow: "hidden" // <--- ADD THIS
              }}
            >
                            {/* V MODIFY THIS LINE V */}
              {avatarUrl ? <img src={getAvatarUrl(avatarUrl)} alt="Avatar" style={{width: '100%', height: '100%', objectFit: 'cover'}}/> : initials}

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
              borderBottom: c("1px solid rgba(255,255,255,0.06)", "1px solid rgba(15,23,42,0.12)"),
              background: c("#0D1117", "#f8fffc"),
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
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16"/>
                </svg>
              </button>

              <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13, color: c("#CBD5E1", "#334155") }}>
                <span>Dashboard</span>
                <span>›</span>
                <span style={{ color: c("#F8FAFC", "#0f172a"), fontWeight: 600 }}>
                  {TABS.find((t) => t.key === activeTab)?.label}
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
                    overflow: "hidden" // <--- ADD THIS
                  }}
                >
                  {avatarUrl ? <img src={getAvatarUrl(avatarUrl)} alt="Avatar" style={{width: '100%', height: '100%', objectFit: 'cover'}}/> : initials}
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
                  <path strokeLinecap="round" strokeLinejoin="round" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"/>
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
              <ActiveTabComponent isLight={isLight} onProfileUpdate={handleProfileUpdate} />
            </div>
          </main>
        </div>
      </div>
    </>
  );
}