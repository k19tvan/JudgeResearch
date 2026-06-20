import React from "react";

export default function WikiTab({ isLight = false }) {
  const t = isLight ? {
    accent:       "#059669",
    textPrimary:  "#0f172a",
    textSecondary:"#475569",
    surface:      "#ffffff",
    border:       "#e2e8f0",
  } : {
    accent:       "#06b6d4",
    textPrimary:  "#f1f5f9",
    textSecondary:"#94a3b8",
    surface:      "#131b2e",
    border:       "rgba(255,255,255,0.12)",
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
                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
              </svg>
            </div>
            <h2 style={{
              margin: 0, fontSize: 18, fontWeight: 700,
              color: t.textPrimary, letterSpacing: "-0.02em",
            }}>
              Wiki
            </h2>
          </div>
          <p style={{ margin: 0, fontSize: 12, color: t.textSecondary, lineHeight: 1.5, paddingLeft: 40 }}>
            Read system guidelines, platform documentation, and study guides.
          </p>
        </div>
      </div>

      <div style={{
        background: t.surface,
        border: `1px solid ${t.border}`,
        borderRadius: 12, padding: 24,
        color: t.textSecondary, fontSize: 13,
      }}>
        Wiki tab content will be implemented here.
      </div>
    </div>
  );
}

