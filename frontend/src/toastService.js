// src/toastService.js

let listeners = [];

export const addToastListener = (listener) => {
  listeners.push(listener);
  return () => {
    listeners = listeners.filter((l) => l !== listener);
  };
};

export const showToast = (message, type = "info", duration = 5000) => {
  const id = Date.now() + Math.random().toString(36).substring(2, 9);
  listeners.forEach((listener) => listener({ id, message, type, duration }));
  return id;
};

// Override window.alert
if (typeof window !== "undefined") {
  window.alert = (message) => {
    if (message === undefined || message === null) return;
    
    const msgString = typeof message === "object" ? JSON.stringify(message) : String(message);
    
    // Auto-detect toast type
    let type = "info";
    const msgLower = msgString.toLowerCase();
    
    if (
      msgLower.includes("error") ||
      msgLower.includes("fail") ||
      msgLower.includes("lỗi") ||
      msgLower.includes("không được") ||
      msgLower.includes("thất bại") ||
      msgLower.includes("bác bỏ")
    ) {
      type = "error";
    } else if (
      msgLower.includes("success") ||
      msgLower.includes("thành công") ||
      msgLower.includes("đã duyệt")
    ) {
      type = "success";
    } else if (
      msgLower.includes("warning") ||
      msgLower.includes("cảnh báo") ||
      msgLower.includes("please") ||
      msgLower.includes("vui lòng") ||
      msgLower.includes("chưa")
    ) {
      type = "warning";
    }
    
    showToast(msgString, type);
  };
}
