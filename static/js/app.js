/**
 * app.js — ResearchAI global JavaScript
 * Handles: dark mode toggle, sidebar, toasts, shared utilities
 */

'use strict';

// ── Dark mode ──────────────────────────────────────────────────────────────
(function initTheme() {
  const saved = localStorage.getItem('theme');
  if (saved === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
  }
})();

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('theme', theme);

  const isDark = theme === 'dark';
  const icon    = document.getElementById('themeIcon');
  const iconMob = document.getElementById('themeIconMobile');

  if (icon) {
    icon.className = isDark ? 'bi bi-sun-fill' : 'bi bi-moon-fill';
  }
  if (iconMob) {
    iconMob.className = isDark ? 'bi bi-sun-fill' : 'bi bi-moon-fill';
  }
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'light';
  applyTheme(current === 'dark' ? 'light' : 'dark');
}

document.addEventListener('DOMContentLoaded', () => {
  // Apply correct icon on load
  const current = document.documentElement.getAttribute('data-theme') || 'light';
  applyTheme(current);

  // Theme toggle buttons
  const themeBtn    = document.getElementById('themeToggle');
  const themeBtnMob = document.getElementById('themeToggleMobile');
  if (themeBtn)    themeBtn.addEventListener('click', toggleTheme);
  if (themeBtnMob) themeBtnMob.addEventListener('click', toggleTheme);

  // Sidebar mobile toggle
  const sidebarToggle = document.getElementById('sidebarToggle');
  const sidebar       = document.getElementById('sidebar');
  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener('click', () => {
      sidebar.classList.toggle('open');
    });
    // Close sidebar when clicking outside
    document.addEventListener('click', (e) => {
      if (!sidebar.contains(e.target) && !sidebarToggle.contains(e.target)) {
        sidebar.classList.remove('open');
      }
    });
  }
});

// ── Toast notifications ────────────────────────────────────────────────────
/**
 * showToast(message, type)
 * type: 'success' | 'danger' | 'warning' | 'info'
 */
function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const colorMap = {
    success: '#22c55e',
    danger:  '#ef4444',
    warning: '#f59e0b',
    info:    '#06b6d4',
  };
  const iconMap = {
    success: 'bi-check-circle-fill',
    danger:  'bi-x-circle-fill',
    warning: 'bi-exclamation-triangle-fill',
    info:    'bi-info-circle-fill',
  };

  const id = 'toast-' + Date.now();
  const color = colorMap[type] || colorMap.info;
  const icon  = iconMap[type]  || iconMap.info;

  const toastHtml = `
    <div id="${id}" class="toast toast-custom align-items-center show" role="alert" aria-live="assertive">
      <div class="d-flex align-items-center p-3 gap-2">
        <i class="bi ${icon}" style="color:${color};font-size:15px;flex-shrink:0"></i>
        <div class="flex-grow-1" style="font-size:13.5px">${escapeHtml(message)}</div>
        <button type="button" class="btn-close ms-2" onclick="this.closest('.toast').remove()" style="font-size:10px;"></button>
      </div>
    </div>`;

  container.insertAdjacentHTML('beforeend', toastHtml);

  // Auto-remove after 4 seconds
  setTimeout(() => {
    const el = document.getElementById(id);
    if (el) {
      el.style.transition = 'opacity 0.3s';
      el.style.opacity = '0';
      setTimeout(() => el.remove(), 300);
    }
  }, 4000);
}

// ── Shared escapeHtml ──────────────────────────────────────────────────────
function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// ── Shared renderMarkdown ─────────────────────────────────────────────────
function renderMarkdown(text) {
  if (!text) return '';
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/^#{3}\s+(.+)$/gm, '<h5 class="mt-2 mb-1">$1</h5>')
    .replace(/^#{2}\s+(.+)$/gm, '<h4 class="mt-2 mb-1">$1</h4>')
    .replace(/^#{1}\s+(.+)$/gm, '<h3 class="mt-2 mb-1">$1</h3>')
    .replace(/^\s*[-•]\s+(.+)$/gm, '<li>$1</li>')
    .replace(/(<li>[\s\S]*?<\/li>)/g, '<ul class="mb-2">$1</ul>')
    .replace(/\n{2,}/g, '</p><p>')
    .replace(/\n/g, '<br>');
}

// ── Global API helper ──────────────────────────────────────────────────────
async function apiPost(url, data) {
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ error: resp.statusText }));
    throw new Error(err.error || `HTTP ${resp.status}`);
  }
  return resp.json();
}

async function apiGet(url) {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}`);
  }
  return resp.json();
}
