// src/components/Register.jsx
import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { registerUser } from "../api";

export default function Register() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    username: "",
    password: "",
    display_name: "",
    email: "",
  });
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [fieldErrors, setFieldErrors] = useState({});

  const inputBaseClass = "cosmic-input mt-1.5 w-full rounded-lg bg-slate-950/60 px-4 py-2 text-slate-100 placeholder-slate-600 focus:outline-none focus:ring-2 transition-all duration-300";
  const inputNormalClass = "border border-slate-800 focus:border-cyan-500 focus:ring-cyan-500/20";
  const inputErrorClass = "border border-red-500 focus:border-red-500 focus:ring-red-500/20";

  const getInputClass = (fieldName) =>
    `${inputBaseClass} ${fieldErrors[fieldName] ? inputErrorClass : inputNormalClass}`;

  const validatePassword = (password) => {
    const errors = [];
    if (password.length < 8) {
      errors.push("Password must be at least 8 characters long.");
    }
    if (!/\d/.test(password)) {
      errors.push("Password must contain at least one digit (0-9).");
    }
    if (/\s/.test(password)) {
      errors.push("Password must not contain whitespace characters.");
    }
    return errors;
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
    if (fieldErrors[name]) {
      setFieldErrors((prev) => {
        const next = { ...prev };
        delete next[name];
        return next;
      });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess("");
    setFieldErrors({});
    const trimmedData = {
      username: formData.username.trim(),
      display_name: formData.display_name.trim(),
      email: formData.email.trim(),
      password: formData.password,
    };
    const missing = [];
    const nextFieldErrors = {};

    if (!trimmedData.display_name) {
      missing.push("display name");
      nextFieldErrors.display_name = true;
    }
    if (!trimmedData.email) {
      missing.push("email");
      nextFieldErrors.email = true;
    }
    if (!trimmedData.username) {
      missing.push("username");
      nextFieldErrors.username = true;
    }
    if (!trimmedData.password || !trimmedData.password.trim()) {
      missing.push("password");
      nextFieldErrors.password = true;
    }

    if (missing.length > 0) {
      setFieldErrors(nextFieldErrors);
      const missingText = missing.join(", ");
      setError(`Missing required fields: ${missingText}.`);
      return;
    }

    const passwordErrors = validatePassword(trimmedData.password);
    if (passwordErrors.length > 0) {
      setFieldErrors({ password: true });
      setError(passwordErrors.join(" "));
      return;
    }
    try {
      await registerUser(trimmedData);
      setSuccess("Registration successful");
      setTimeout(() => navigate("/login"), 2000); // Chuyển sang trang đăng nhập sau 2 giây
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="flex min-h-screen w-screen bg-[#030014] text-slate-100 overflow-hidden">
      
      {/* Tích hợp trực tiếp style xử lý autofill và chuyển động tinh vân công nghệ */}
      <style>{`
        .cosmic-input:-webkit-autofill,
        .cosmic-input:-webkit-autofill:hover, 
        .cosmic-input:-webkit-autofill:focus, 
        .cosmic-input:-webkit-autofill:active {
          -webkit-box-shadow: 0 0 0 1000px #090d16 inset !important;
          -webkit-text-fill-color: #f1f5f9 !important;
          transition: background-color 5000s ease-in-out 0s;
        }
        @keyframes neuralPulse {
          0%, 100% {
            transform: scale(1) translate(0px, 0px);
            opacity: 0.12;
          }
          50% {
            transform: scale(1.18) translate(10px, -10px);
            opacity: 0.22;
          }
        }
        .neural-glow-1 {
          animation: neuralPulse 14s ease-in-out infinite alternate;
        }
        .neural-glow-2 {
          animation: neuralPulse 18s ease-in-out infinite alternate-reverse;
        }
      `}</style>

      {/* LEFT SIDE: Code & AI Concept Visuals */}
      <div className="relative hidden md:flex w-1/2 items-center justify-center overflow-hidden border-r border-white/5">
        <img 
          src="https://images.unsplash.com/photo-1635070041078-e363dbe005cb?auto=format&fit=crop&w=1200&q=80" 
          alt="AI Complex Architecture" 
          className="absolute inset-0 h-full w-full object-cover opacity-60 mix-blend-screen"
        />
        <div className="absolute inset-0 bg-gradient-to-r from-transparent to-[#030014]" />
        
        <div className="relative z-10 px-12 text-left max-w-lg">
          <span className="inline-block rounded-full bg-cyan-500/10 px-3 py-1 text-xs font-semibold tracking-wider text-cyan-400 border border-cyan-500/30">
            AI AGENT ROADMAPS
          </span>
          <h1 className="mt-4 text-4xl font-extrabold tracking-tight text-white lg:text-5xl">
            Design your path <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-indigo-300 to-emerald-400">
              in Machine Learning.
            </span>
          </h1>
          <p className="mt-4 text-slate-400 leading-relaxed">
            Register your workspace to unlock AI Agent support, practice Python in Monaco Editor, and submit ML algorithms against automated testcases in the sandbox.
          </p>
        </div>
      </div>

      {/* RIGHT SIDE: Terminal-themed Form Container */}
      <div className="relative flex w-full md:w-1/2 items-center justify-center px-6 py-12 overflow-y-auto">
        {/* Đèn nền tinh vân xanh và tím bảo mật */}
        <div className="neural-glow-1 absolute top-1/4 left-1/4 w-72 h-72 bg-indigo-600/10 rounded-full blur-[100px] pointer-events-none" />
        <div className="neural-glow-2 absolute bottom-1/4 right-1/4 w-80 h-80 bg-cyan-500/10 rounded-full blur-[120px] pointer-events-none" />
        
        <div className="relative z-10 w-full max-w-md rounded-2xl border border-white/10 bg-slate-950/40 p-8 shadow-[0_0_50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
          
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-slate-900/80 border border-slate-700/50 shadow-[0_0_15px_rgba(16,185,129,0.3)]">
            <span className="text-2xl animate-bounce">⚡</span>
          </div>

          <h2 className="mt-4 text-center text-3xl font-extrabold tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-indigo-200 to-emerald-400">
            SIGN UP
          </h2>
          <p className="mt-2 text-center text-sm text-slate-400">
            Registered on this server?{" "}
            <Link to="/login" className="font-semibold text-emerald-400 hover:text-emerald-300 transition duration-300 hover:underline">
              Access account
            </Link>
          </p>

          {error && (
            <div className="mt-4 rounded-lg bg-red-950/50 border border-red-500/30 p-3 text-sm text-red-400 shadow-[0_0_10px_rgba(239,68,68,0.2)]">
              ⚠️ System Error: {error}
            </div>
          )}
          {success && (
            <div className="mt-4 rounded-lg bg-emerald-950/50 border border-emerald-500/30 p-3 text-sm text-emerald-400 shadow-[0_0_10px_rgba(16,185,129,0.2)]">
              🚀 Workspace Init: {success}
            </div>
          )}

          <form onSubmit={handleSubmit} className="mt-6 space-y-4">
            <div>
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">Display Name</label>
              <input
                type="text"
                name="display_name"
                required
                className={getInputClass("display_name")}
                placeholder="E.g. Alan Turing"
                onChange={handleChange}
                aria-invalid={fieldErrors.display_name ? "true" : "false"}
              />
            </div>
            <div>
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">Email Address</label>
              <input
                type="email"
                name="email"
                required
                className={getInputClass("email")}
                placeholder="email@example.com"
                onChange={handleChange}
                aria-invalid={fieldErrors.email ? "true" : "false"}
              />
            </div>
            <div>
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">Username</label>
              <input
                type="text"
                name="username"
                required
                className={getInputClass("username")}
                placeholder="Set unique ID..."
                onChange={handleChange}
                aria-invalid={fieldErrors.username ? "true" : "false"}
              />
            </div>
            <div>
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">Password</label>
              <input
                type="password"
                name="password"
                required
                className={getInputClass("password")}
                placeholder="••••••••"
                onChange={handleChange}
                aria-invalid={fieldErrors.password ? "true" : "false"}
              />
              <p className="mt-1 text-xs text-slate-500">At least 8 characters, 1 digit, no spaces.</p>
            </div>

            <button
              type="submit"
              className="w-full mt-4 rounded-lg bg-gradient-to-r from-cyan-500 via-indigo-500 to-emerald-600 py-3 font-semibold text-white shadow-lg shadow-emerald-500/10 transition-all duration-300 hover:from-cyan-400 hover:via-indigo-400 hover:to-emerald-500 hover:shadow-emerald-500/30 transform hover:scale-[1.01] active:scale-[0.99]"
            >
              Register
            </button>
            <button
              type="button"
              onClick={() => navigate("/login")}
              className="mt-3 w-full rounded-lg border border-slate-700/60 py-2 text-sm font-semibold text-slate-300 transition-all duration-300 hover:border-slate-400 hover:text-white"
            >
              Back
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}