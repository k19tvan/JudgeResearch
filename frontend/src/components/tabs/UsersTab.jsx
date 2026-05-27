import React from "react";

export default function UsersTab({ isLight = false }) {
  return (
    <section>
      <h2 className={`text-2xl font-bold tracking-wide ${isLight ? "text-slate-900" : "text-white"}`}>USERS</h2>
      <p className={`mt-3 text-sm ${isLight ? "text-slate-700" : "text-slate-400"}`}>
        Users tab content will be implemented here.
      </p>
    </section>
  );
}
