// src/api.js
import axios from "axios";
axios.defaults.withCredentials = true;

const AUTH_API_URL = "http://localhost:21081/api/auth";
const PROBLEM_API_URL = "http://localhost:21081/api/problems";
const ROADMAP_API_URL = "http://localhost:21081/api/roadmaps";
const ROADMAP_STEP_API_URL = "http://localhost:21081/api/roadmap-steps"; 
const USER_API_URL = "http://localhost:21081/api/users";

// ================ CORE SECURE FETCH FUNCTION (WITH SILENT REFRESH) ================

async function customFetch(url, options = {}) {
  let accessToken = localStorage.getItem("access_token");

  // Bắt buộc phải có để truyền nhận HttpOnly Cookie chéo cổng localhost
  options.credentials = "include"; 

  options.headers = {
    ...(accessToken ? { "Authorization": `Bearer ${accessToken}` } : {}),
    ...options.headers,
  };

  let response = await fetch(url, options);

  // Tự động xử lý âm thầm khi Access Token hết hạn (401)
  if (response.status === 401) {
    try {
      // Gọi API refresh (Không cần truyền body vì cookie tự động được đính kèm chéo cổng)
      const refreshResponse = await fetch("http://localhost:21081/api/auth/refresh", {
        method: "POST",
        credentials: "include",
      });

      if (refreshResponse.ok) {
        const refreshData = await refreshResponse.json();
        localStorage.setItem("access_token", refreshData.access_token);
        
        // Gắn access_token mới vào cấu hình và thực thi lại yêu cầu cũ
        options.headers["Authorization"] = `Bearer ${refreshData.access_token}`;
        response = await fetch(url, options);
      } else {
        handleSessionExpired();
      }
    } catch (err) {
      handleSessionExpired();
    }
  }

  return response;
}

function handleSessionExpired() {
  localStorage.removeItem("access_token");
  localStorage.removeItem("username");
  localStorage.removeItem("user_id");
  localStorage.removeItem("user_role");
  window.location.href = "/login";
}

// ================ AUTHENTICATION ENDPOINTS ================

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
    credentials: "include", // Đảm bảo nhận HttpOnly Cookie refresh_token từ máy chủ
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Sai tài khoản hoặc mật khẩu");
  }
  return response.json(); 
};

// ================ PROBLEMS ENDPOINTS ================

export const createManualProblem = async (formData) => {
  // customFetch tự động thêm credentials: "include" và Authorization header
  const response = await customFetch(`${PROBLEM_API_URL}/create/manual`, {
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

export const fetchProblemContent = async (problemId, userId) => {
  const url = userId 
    ? `http://localhost:21081/api/problems/${problemId}/content?user_id=${userId}`
    : `http://localhost:21081/api/problems/${problemId}/content`;
  const response = await customFetch(url);
  return response.json();
};

export const createProblemsFromRepo = async (payload) => {
  const response = await customFetch(`${PROBLEM_API_URL}/problems_from_repo`, {
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
  const response = await customFetch(`${PROBLEM_API_URL}/draft_sessions?user_id=${userId}`);

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Fetch draft sessions failed");
  }

  return response.json();
};

export const fetchDraftSessionDetail = async (sessionId) => {
  const response = await customFetch(`${PROBLEM_API_URL}/draft_sessions/${sessionId}`);

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Fetch draft session failed");
  }

  return response.json();
};

export const updateDraftSessionFeedback = async (payload) => {
  const response = await customFetch(`${PROBLEM_API_URL}/draft_sessions/feedback`, {
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
  const response = await customFetch(`${PROBLEM_API_URL}/draft_sessions/finalize`, {
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

// ================ ROADMAPS ENDPOINTS ================

export const fetchRoadmap = async (roadmapId) => {
  const response = await customFetch(`${ROADMAP_API_URL}/${roadmapId}`);

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Fetch roadmap failed");
  }

  return response.json();
};

export const fetchRoadmaps = async (userId = null, filterMode = null) => {
  let url = `${ROADMAP_API_URL}`;
  const params = [];
  if (userId) params.push(`user_id=${userId}`);
  if (filterMode) params.push(`filter_mode=${filterMode}`);
  if (params.length > 0) {
    url += `?${params.join("&")}`;
  }
  const response = await customFetch(url);

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Fetch roadmaps failed");
  }

  return response.json();
};

export const createProblemDetailedly = async (stepId) => {
  const response = await customFetch(`${ROADMAP_STEP_API_URL}/${stepId}/create_detailedly`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Create problem materials failed");
  }

  return response.json();
};

export const saveStepToProblem = async (stepId) => {
  const response = await customFetch(`${ROADMAP_STEP_API_URL}/${stepId}/save_to_problem`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Save step to problem failed");
  }

  return response.json();
};

// ================ CODE RUN / SUBMISSION ENDPOINTS ================

export const runProblem = async (problemId, submitted_code) => {
  const response = await customFetch(`${PROBLEM_API_URL}/${problemId}/run`, {
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
  const response = await customFetch(`${PROBLEM_API_URL}/${problemId}/submit`, {
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
  const response = await customFetch(`${PROBLEM_API_URL}/${problemId}/submissions?user_id=${userId}`);
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Failed to fetch submissions");
  }
  return response.json();
};

// ================ USER PROFILE ENDPOINTS ================

export const fetchUserProfile = async (userId) => {
  const response = await customFetch(`${USER_API_URL}/${userId}`);

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Fetch user profile failed");
  }

  return response.json();
};

export const updateUserProfile = async (userId, profileData) => {
  const headers = {};
  if (!(profileData instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  const response = await customFetch(`${USER_API_URL}/${userId}`, {
    method: "PUT",
    headers,
    body: profileData instanceof FormData ? profileData : JSON.stringify(profileData),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Update user profile failed");
  }

  const data = await response.json();
  if (data.access_token) {
    localStorage.setItem("access_token", data.access_token);
  }
  return data;
};

export const deactivateUserAccount = async (userId) => {
  const response = await customFetch(`${USER_API_URL}/${userId}/deactivate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirm: true }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Account deactivation failed");
  }

  return response.json();
};

// ================ ADMIN ACCOUNT MANAGEMENT ENDPOINTS ================

export const fetchManagedUsers = async (search = "") => {
  const query = search.trim() ? `?search=${encodeURIComponent(search.trim())}` : "";
  const response = await customFetch(`${USER_API_URL.replace("/users", "/admin/users")}${query}`);

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Fetch managed users failed");
  }

  return response.json();
};

export const fetchManagedUserDetails = async (userId) => {
  const response = await customFetch(`${USER_API_URL.replace("/users", "/admin/users")}/${userId}`);

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Fetch managed user failed");
  }

  return response.json();
};

export const updateManagedUser = async (userId, accountData) => {
  const response = await customFetch(`${USER_API_URL.replace("/users", "/admin/users")}/${userId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(accountData),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Update managed user failed");
  }

  return response.json();
};

// ================ GENERAL SUBMISSIONS ENDPOINTS ================

export const createSubmission = async (submissionData) => {
  const response = await customFetch(`http://localhost:21081/api/submissions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(submissionData),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Create submission failed");
  }

  return response.json();
};

export const fetchSubmission = async (submissionId) => {
  const response = await customFetch(`http://localhost:21081/api/submissions/${submissionId}`);

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
  
  const response = await customFetch(url);

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

  const response = await customFetch(url);

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Filter problems failed");
  }

  return response.json();
};

// ================ APPROVAL WORKFLOW ENDPOINTS ================

export const requestProblemApproval = async (problemId) => {
  const response = await customFetch(`${PROBLEM_API_URL}/${problemId}/request-approval`, {
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
  const response = await customFetch(`${PROBLEM_API_URL}/${problemId}/approve`, {
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
  const response = await customFetch(`${PROBLEM_API_URL}/${problemId}/reject`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Reject problem failed");
  }

  return response.json();
};

export const deleteRoadmap = async (roadmapId) => {
  const response = await customFetch(`${ROADMAP_API_URL}/${roadmapId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Delete roadmap failed");
  }
  return response.json();
};

export const requestRoadmapApproval = async (roadmapId) => {
  const response = await customFetch(`${ROADMAP_API_URL}/${roadmapId}/request-approval`, {
    method: "POST",
  });
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Request approval failed");
  }
  return response.json();
};

export const approveRoadmap = async (roadmapId) => {
  const response = await customFetch(`${ROADMAP_API_URL}/${roadmapId}/approve`, {
    method: "POST",
  });
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Approve roadmap failed");
  }
  return response.json();
};

export const rejectRoadmap = async (roadmapId) => {
  const response = await customFetch(`${ROADMAP_API_URL}/${roadmapId}/reject`, {
    method: "POST",
  });
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Reject roadmap failed");
  }
  return response.json();
};

export const publishRoadmapDirectly = async (roadmapId) => {
  const response = await customFetch(`${ROADMAP_API_URL}/${roadmapId}/publish`, {
    method: "POST",
  });
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Publish roadmap failed");
  }
  return response.json();
};

export const unpublishRoadmap = async (roadmapId) => {
  const response = await customFetch(`${ROADMAP_API_URL}/${roadmapId}/unpublish`, {
    method: "POST",
  });
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Unpublish roadmap failed");
  }
  return response.json();
};

export const fetchStepDraftPreview = async (stepId) => {
  const response = await customFetch(`${ROADMAP_STEP_API_URL}/${stepId}/preview`);
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Fetch step preview failed");
  }
  return response.json();
};