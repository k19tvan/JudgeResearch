import React from "react";

export default function WikiTab({ isLight = false }) {
  return (
    <section>
      <h2 className={`text-2xl font-bold tracking-wide ${isLight ? "text-slate-900" : "text-white"}`}>WIKI</h2>
      <p className={`mt-3 text-sm ${isLight ? "text-slate-700" : "text-slate-400"}`}>
        Wiki tab content will be implemented here.
      </p>
    </section>
  );
}
