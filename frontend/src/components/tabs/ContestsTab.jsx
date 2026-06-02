import React from "react";

export default function ContestsTab({ isLight = false }) {
  const t = isLight ? {
    accent:       "#059669",
    textPrimary:  "#0f172a",
    textSecondary:"#475569",
    surface:      "#ffffff",
    border:       "#e2e8f0",
  } : {
    accent:       "#06b6d4",
    textPrimary:  "#f1f5f9",
    textSecondary:"#64748b",
    surface:      "#0f172a",
    border:       "rgba(255,255,255,0.07)",
  };

  return (
    <div style={{ fontFamily: "'Inter var', 'Inter', sans-serif" }}>
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
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={isLight ? "#fff" : "#22d3ee"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
              </svg>
            </div>
            <h2 style={{
              margin: 0, fontSize: 18, fontWeight: 700,
              color: t.textPrimary, letterSpacing: "-0.02em",
            }}>
              Contests
            </h2>
          </div>
          <p style={{ margin: 0, fontSize: 12, color: t.textSecondary, lineHeight: 1.5, paddingLeft: 40 }}>
            Compete with others in real-time or practice past competitive programming challenges.
          </p>
        </div>
      </div>

      <div style={{
        background: t.surface,
        border: `1px solid ${t.border}`,
        borderRadius: 12, padding: 24,
        color: t.textSecondary, fontSize: 13,
      }}>
        Contests tab content will be implemented here.
      </div>
    </div>
  );
}

