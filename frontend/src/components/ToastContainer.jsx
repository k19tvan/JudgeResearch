import React, { useState, useEffect } from "react";
import { addToastListener } from "../toastService";

export default function ToastContainer() {
  const [toasts, setToasts] = useState([]);

  useEffect(() => {
    const removeListener = addToastListener((newToast) => {
      // Avoid exact duplicates showing up at the exact same moment
      setToasts((prev) => {
        if (prev.some((t) => t.message === newToast.message && Date.now() - t.id < 500)) {
          return prev;
        }
        return [...prev, newToast];
      });

      // Auto remove toast after duration
      setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== newToast.id));
      }, newToast.duration);
    });

    return () => {
      removeListener();
    };
  }, []);

  const handleClose = (id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  return (
    <div className="toast-container">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onClose={() => handleClose(toast.id)} />
      ))}
    </div>
  );
}

function ToastItem({ toast, onClose }) {
  const { message, type, duration } = toast;
  const [isExiting, setIsExiting] = useState(false);

  useEffect(() => {
    // Set exit animation slightly before the actual removal from state
    const timer = setTimeout(() => {
      setIsExiting(true);
    }, duration - 400);

    return () => clearTimeout(timer);
  }, [duration]);

  // Icons mapping
  const getIcon = () => {
    switch (type) {
      case "success":
        return (
          <svg className="toast-icon text-success" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
      case "error":
        return (
          <svg className="toast-icon text-error" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
      case "warning":
        return (
          <svg className="toast-icon text-warning" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        );
      default:
        return (
          <svg className="toast-icon text-info" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
    }
  };

  return (
    <div className={`toast-item ${type} ${isExiting ? "slide-out" : "slide-in"}`}>
      <div className="toast-content">
        <div className="toast-icon-wrapper">{getIcon()}</div>
        <div className="toast-message">{message}</div>
        <button className="toast-close-btn" onClick={onClose}>
          <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      <div className="toast-progress-bar-wrapper">
        <div 
          className="toast-progress-bar" 
          style={{ animationDuration: `${duration}ms` }}
        />
      </div>
    </div>
  );
}
