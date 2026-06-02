// src/components/Login.jsx
import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { loginUser } from "../api";

export default function Login() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({ username: "", password: "" });
  const [error, setError] = useState("");

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
      e.preventDefault();
      setError("");
      try {
        const data = await loginUser(formData);
        
        localStorage.setItem("access_token", data.access_token);
        localStorage.setItem("refresh_token", data.refresh_token);
        localStorage.setItem("username", formData.username);
        
        // BỔ SUNG: Lưu user_id vào localStorage để các component khác sử dụng
        localStorage.setItem("user_id", data.user_id); 
        localStorage.setItem("user_role", data.user_role);
        if (data.avatar_url) {
          localStorage.setItem("avatar_url", data.avatar_url);
        } else {
          localStorage.removeItem("avatar_url");
        }

        navigate("/");
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
          animation: neuralPulse 12s ease-in-out infinite alternate;
        }
        .neural-glow-2 {
          animation: neuralPulse 16s ease-in-out infinite alternate-reverse;
        }
      `}</style>

      {/* LEFT SIDE: Neural Network Visuals & Concept */}
      <div className="relative hidden md:flex w-1/2 items-center justify-center overflow-hidden border-r border-white/5">
        <img 
          src="https://images.unsplash.com/photo-1507668077129-56e32842fceb?auto=format&fit=crop&w=1200&q=80" 
          alt="Neural Network Abstract" 
          className="absolute inset-0 h-full w-full object-cover opacity-60 mix-blend-screen"
        />
        <div className="absolute inset-0 bg-gradient-to-r from-transparent to-[#030014]" />
        
        <div className="relative z-10 px-12 text-left max-w-lg">
          <span className="inline-block rounded-full bg-cyan-500/10 px-3 py-1 text-xs font-semibold tracking-wider text-cyan-400 border border-cyan-500/30">
            ML ONLINE JUDGE
          </span>
          <h1 className="mt-4 text-4xl font-extrabold tracking-tight text-white lg:text-5xl">
            Train & Evaluate <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-indigo-300 to-emerald-400">
              your models online.
            </span>
          </h1>
          <p className="mt-4 text-slate-400 leading-relaxed">
            Execute Python code inside our secure Docker sandbox environment. Leverage AI-driven suggestions and optimize your machine learning pipeline with real-time feedback.
          </p>
        </div>
      </div>

      {/* RIGHT SIDE: Terminal-themed Form Container */}
      <div className="relative flex w-full md:w-1/2 items-center justify-center px-6 py-12">
        {/* Đèn nền tinh vân tím và xanh lam đại diện cho xử lý AI */}
        <div className="neural-glow-1 absolute top-1/3 left-1/3 w-72 h-72 bg-indigo-600/10 rounded-full blur-[100px] pointer-events-none" />
        <div className="neural-glow-2 absolute bottom-1/3 right-1/3 w-80 h-80 bg-cyan-500/10 rounded-full blur-[120px] pointer-events-none" />
        
        <div className="relative z-10 w-full max-w-md rounded-2xl border border-white/10 bg-slate-950/40 p-8 shadow-[0_0_50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
          
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-slate-900/80 border border-slate-700/50 shadow-[0_0_15px_rgba(16,185,129,0.3)]">
            <span className="text-2xl animate-pulse">🤖</span>
          </div>

          <h2 className="mt-4 text-center text-3xl font-extrabold tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-indigo-200 to-emerald-400">
            SIGN IN
          </h2>
          <p className="mt-2 text-center text-sm text-slate-400">
            Ready to deploy?{" "}
            <Link to="/register" className="font-semibold text-emerald-400 hover:text-emerald-300 transition duration-300 hover:underline">
              Create an account
            </Link>
          </p>

          {error && (
            <div className="mt-4 rounded-lg bg-red-950/50 border border-red-500/30 p-3 text-sm text-red-400 shadow-[0_0_10px_rgba(239,68,68,0.2)]">
              ⚠️ System Error: {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="mt-6 space-y-5">
            <div>
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">Username</label>
              <input
                type="text"
                name="username"
                required
                className="cosmic-input mt-1.5 w-full rounded-lg border border-slate-800 bg-slate-950/60 px-4 py-2.5 text-slate-100 placeholder-slate-600 focus:border-cyan-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/20 transition-all duration-300"
                placeholder="Your username..."
                onChange={handleChange}
              />
            </div>
            <div>
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">Password</label>
              <input
                type="password"
                name="password"
                required
                className="cosmic-input mt-1.5 w-full rounded-lg border border-slate-800 bg-slate-950/60 px-4 py-2.5 text-slate-100 placeholder-slate-600 focus:border-cyan-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/20 transition-all duration-300"
                placeholder="••••••••"
                onChange={handleChange}
              />
            </div>

            <button
              type="submit"
              className="w-full mt-2 rounded-lg bg-gradient-to-r from-cyan-500 via-indigo-500 to-emerald-600 py-3 font-semibold text-white shadow-lg shadow-emerald-500/10 transition-all duration-300 hover:from-cyan-400 hover:via-indigo-400 hover:to-emerald-500 hover:shadow-emerald-500/30 transform hover:scale-[1.01] active:scale-[0.99]"
            >
              Initialize Session
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
