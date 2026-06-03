// src/App.jsx
import React from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import Login from "./components/Login";
import Register from "./components/Register";
import Home from "./components/Home";
import LiveCodingPage from "./components/LiveCodingPage";
import ResearchRoadmapPage from "./components/ResearchRoadmapPage";

function App() {
  return (
    <Router>
      <Routes>
        {/* Trang chủ */}
        <Route path="/" element={<Home />} />
        
        {/* Trang đăng nhập */}
        <Route path="/login" element={<Login />} />
        
        {/* Trang đăng ký */}
        <Route path="/register" element={<Register />} />

        {/* Trang live coding */}
        <Route path="/livecoding/:problemId" element={<LiveCodingPage />} />
        <Route path="/livecoding/draft/:stepId" element={<LiveCodingPage />} />

        {/* Trang chi tiết research roadmap */}
        <Route path="/research/draft/:sessionId" element={<ResearchRoadmapPage mode="draft" />} />
        <Route path="/research/roadmap/:roadmapId" element={<ResearchRoadmapPage mode="roadmap" />} />
        
        {/* Nếu người dùng vào bất cứ đường dẫn lạ nào, tự động đẩy về trang chủ */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
}

export default App;
