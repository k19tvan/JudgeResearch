const AUTH_API_URL = "http://localhost:21081/api/auth";
const PROBLEM_API_URL = "http://localhost:21081/api/problems";
const ROADMAP_API_URL = "http://localhost:21081/api/roadmaps";
const ROADMAP_STEP_API_URL = "http://localhost:21081/api/roadmap-steps"; // Thêm base URL cho roadmap-steps
const USER_API_URL = "http://localhost:21081/api/users";

export const registerUser = async (userData) => {
  const response = await fetch(`${AUTH_API_URL}/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(userData),
  });
  
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Đăng ký thất bại");
  }
  return response.json();
};

export const loginUser = async (credentials) => {
  const response = await fetch(`${AUTH_API_URL}/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(credentials),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Sai tài khoản hoặc mật khẩu");
  }
  return response.json(); 
};

export const createManualProblem = async (formData) => {
  // Không đặt Header 'Content-Type' vì trình duyệt sẽ tự động thiết lập ranh giới (boundary) cho FormData
  const response = await fetch(`${PROBLEM_API_URL}/create/manual`, {
    method: "POST",
    body: formData, 
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Create problem failed");
  }

  return response.json();
};

export const fetchProblems = async (userId = null) => {
  if (userId) {
    return filterProblems("all", userId);
  }
  return filterProblems("public");
};

export const fetchProblemContent = async (problemId) => {
  const response = await fetch(`${PROBLEM_API_URL}/${problemId}/content`, {
    method: "GET",
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Fetch problem content failed");
  }

  return response.json();
};

export const createProblemsFromRepo = async (payload) => {
  const response = await fetch(`${PROBLEM_API_URL}/problems_from_repo`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Create proposed list failed");
  }

  return response.json();
};

export const fetchDraftSessions = async (userId) => {
  const response = await fetch(`${PROBLEM_API_URL}/draft_sessions?user_id=${userId}`);

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Fetch draft sessions failed");
  }

  return response.json();
};

export const fetchDraftSessionDetail = async (sessionId) => {
  const response = await fetch(`${PROBLEM_API_URL}/draft_sessions/${sessionId}`);

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Fetch draft session failed");
  }

  return response.json();
};

export const updateDraftSessionFeedback = async (payload) => {
  const response = await fetch(`${PROBLEM_API_URL}/draft_sessions/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Update proposed list failed");
  }

  return response.json();
};

export const finalizeDraftSession = async (payload) => {
  const response = await fetch(`${PROBLEM_API_URL}/draft_sessions/finalize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Save roadmap failed");
  }

  return response.json();
};

export const fetchRoadmap = async (roadmapId) => {
  const response = await fetch(`${ROADMAP_API_URL}/${roadmapId}`);

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Fetch roadmap failed");
  }

  return response.json();
};

export const fetchRoadmaps = async (userId) => {
  const response = await fetch(`${ROADMAP_API_URL}?user_id=${userId}`);

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Fetch roadmaps failed");
  }

  return response.json();
};

// Đã cập nhật khớp hoàn toàn với endpoint roadmap-step mới của backend
export const createProblemDetailedly = async (stepId) => {
  const response = await fetch(`${ROADMAP_STEP_API_URL}/${stepId}/create_detailedly`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Create problem materials failed");
  }

  return response.json();
};

// Đã cập nhật khớp hoàn toàn với endpoint lưu chính thức mới của backend
export const saveStepToProblem = async (stepId) => {
  const response = await fetch(`${ROADMAP_STEP_API_URL}/${stepId}/save_to_problem`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Save step to problem failed");
  }

  return response.json();
};

export const runProblem = async (problemId, submitted_code) => {
  const response = await fetch(`${PROBLEM_API_URL}/${problemId}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ submitted_code }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Run failed");
  }

  return response.json();
};

export const submitProblem = async (problemId, user_id, submitted_code) => {
  const response = await fetch(`${PROBLEM_API_URL}/${problemId}/submit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id, submitted_code }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Submit failed");
  }

  return response.json();
};

export const fetchProblemSubmissions = async (problemId, userId) => {
  const response = await fetch(`${PROBLEM_API_URL}/${problemId}/submissions?user_id=${userId}`);
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Failed to fetch submissions");
  }
  return response.json();
};

// ================ USER PROFILE ENDPOINTS ================

export const fetchUserProfile = async (userId) => {
  const response = await fetch(`${USER_API_URL}/${userId}`);

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Fetch user profile failed");
  }

  return response.json();
};

export const updateUserProfile = async (userId, profileData) => {
  const response = await fetch(`${USER_API_URL}/${userId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(profileData),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Update user profile failed");
  }

  return response.json();
};

// ================ SUBMISSIONS / LIVE CODING ENDPOINTS ================

export const createSubmission = async (submissionData) => {
  const response = await fetch(`http://localhost:21081/api/submissions`, {
    method: "POST",
    headers: { 
      "Content-Type": "application/json",
    },
    body: JSON.stringify(submissionData),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Create submission failed");
  }

  return response.json();
};

export const fetchSubmission = async (submissionId) => {
  const response = await fetch(`http://localhost:21081/api/submissions/${submissionId}`);

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Fetch submission failed");
  }

  return response.json();
};

export const fetchUserSubmissions = async (userId, problemId = null) => {
  const url = problemId 
    ? `${USER_API_URL}/${userId}/submissions?problem_id=${problemId}`
    : `${USER_API_URL}/${userId}/submissions`;
  
  const response = await fetch(url);

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Fetch submissions failed");
  }

  return response.json();
};

// ================ PROBLEM FILTERING ENDPOINTS ================

export const filterProblems = async (filterMode = "public", userId = null) => {
  let url = `${PROBLEM_API_URL}/filter?filter_mode=${filterMode}`;
  if (userId) {
    url += `&user_id=${userId}`;
  }

  const response = await fetch(url);

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Filter problems failed");
  }

  return response.json();
};

// ================ APPROVAL WORKFLOW ENDPOINTS ================

export const requestProblemApproval = async (problemId) => {
  const response = await fetch(`${PROBLEM_API_URL}/${problemId}/request-approval`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Request approval failed");
  }

  return response.json();
};

export const approveProblem = async (problemId) => {
  const response = await fetch(`${PROBLEM_API_URL}/${problemId}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Approve problem failed");
  }

  return response.json();
};

export const rejectProblem = async (problemId) => {
  const response = await fetch(`${PROBLEM_API_URL}/${problemId}/reject`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Reject problem failed");
  }

  return response.json();
};