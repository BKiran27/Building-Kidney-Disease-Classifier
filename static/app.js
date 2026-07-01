/**
 * KidneyScan AI — Frontend Application Logic
 * ============================================
 * Handles:
 *  - Drag & drop / file upload
 *  - API communication
 *  - Results rendering with animations
 *  - Model info fetch
 *  - Navbar scroll effect
 */

'use strict';

const API_BASE = window.location.origin;

// ─── Class Color Map ──────────────────────────────────────────────────────────
const CLASS_COLORS = {
  Normal: { fill: '#10b981', ring: '#10b981', severity: 'low',      badge: 'rgba(16,185,129,0.15)',  border: 'rgba(16,185,129,0.35)'  },
  Cyst:   { fill: '#06b6d4', ring: '#06b6d4', severity: 'moderate', badge: 'rgba(6,182,212,0.15)',   border: 'rgba(6,182,212,0.35)'   },
  Stone:  { fill: '#f59e0b', ring: '#f59e0b', severity: 'moderate', badge: 'rgba(245,158,11,0.15)',  border: 'rgba(245,158,11,0.35)'  },
  Tumor:  { fill: '#ef4444', ring: '#ef4444', severity: 'high',     badge: 'rgba(239,68,68,0.15)',   border: 'rgba(239,68,68,0.35)'   },
};

const SEVERITY_ICONS = {
  Normal: 'fa-circle-check',
  Cyst:   'fa-circle-dot',
  Stone:  'fa-diamond',
  Tumor:  'fa-triangle-exclamation',
};

// ─── DOM Refs ─────────────────────────────────────────────────────────────────
const uploadZone      = document.getElementById('upload-zone');
const fileInput       = document.getElementById('file-input');
const uploadIdle      = document.getElementById('upload-idle');
const uploadPreview   = document.getElementById('upload-preview');
const previewImg      = document.getElementById('preview-img');
const analyzeBtn      = document.getElementById('analyze-btn');
const btnText         = analyzeBtn.querySelector('.btn-text');
const btnLoading      = analyzeBtn.querySelector('.btn-loading');
const resultsIdle     = document.getElementById('results-idle');
const resultsContent  = document.getElementById('results-content');
const predBadge       = document.getElementById('prediction-badge');
const predIcon        = document.getElementById('pred-icon');
const predClass       = document.getElementById('pred-class');
const ringFill        = document.getElementById('ring-fill');
const ringPct         = document.getElementById('ring-pct');
const severityBar     = document.getElementById('severity-bar');
const severityText    = document.getElementById('severity-text');
const resultDesc      = document.getElementById('result-description');
const probBars        = document.getElementById('prob-bars');
const demoWarning     = document.getElementById('demo-warning');
const apiStatusBadge  = document.getElementById('api-status-badge');
const statusText      = apiStatusBadge.querySelector('.status-text');
const totalParams     = document.getElementById('total-params');
const navbar          = document.getElementById('navbar');

let selectedFile = null;

// ═══════════════════════════════════════════════════════════════════════════════
// Navbar Scroll
// ═══════════════════════════════════════════════════════════════════════════════
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 20);
});

// ═══════════════════════════════════════════════════════════════════════════════
// API Health Check & Model Info
// ═══════════════════════════════════════════════════════════════════════════════
async function checkApiHealth() {
  try {
    const [healthRes, infoRes] = await Promise.all([
      fetch(`${API_BASE}/api/health`),
      fetch(`${API_BASE}/api/model-info`),
    ]);

    if (healthRes.ok) {
      const health = await healthRes.json();
      setApiStatus('connected', health.model_loaded ? 'API Online' : 'Demo Mode');
    }

    if (infoRes.ok) {
      const info = await infoRes.json();
      if (info.total_params) {
        totalParams.textContent = Number(info.total_params).toLocaleString();
      } else {
        totalParams.textContent = 'Not trained';
      }
    }
  } catch (err) {
    setApiStatus('error', 'API Offline');
    console.warn('API health check failed:', err);
  }
}

function setApiStatus(state, text) {
  apiStatusBadge.className = 'api-status ' + state;
  statusText.textContent = text;
}

// ═══════════════════════════════════════════════════════════════════════════════
// File Upload / Drag & Drop
// ═══════════════════════════════════════════════════════════════════════════════
uploadZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

// Drag & Drop
uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('dragover');
});
uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('dragover');
});
uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    handleFile(file);
  }
});

function handleFile(file) {
  selectedFile = file;

  // Show preview
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    uploadIdle.classList.add('hidden');
    uploadPreview.classList.remove('hidden');
  };
  reader.readAsDataURL(file);

  // Enable analyze button
  analyzeBtn.disabled = false;

  // Reset results
  resetResults();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Analyze Button
// ═══════════════════════════════════════════════════════════════════════════════
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  setAnalyzing(true);

  try {
    const formData = new FormData();
    formData.append('image', selectedFile);

    const response = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || `Server error: ${response.status}`);
    }

    const result = await response.json();
    renderResults(result);

  } catch (err) {
    console.error('Prediction error:', err);
    renderError(err.message);
  } finally {
    setAnalyzing(false);
  }
});

function setAnalyzing(isAnalyzing) {
  analyzeBtn.disabled = isAnalyzing;
  btnText.classList.toggle('hidden', isAnalyzing);
  btnLoading.classList.toggle('hidden', !isAnalyzing);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Results Rendering
// ═══════════════════════════════════════════════════════════════════════════════
function renderResults(data) {
  const cls    = data.predicted_class;
  const conf   = data.confidence;
  const colors = CLASS_COLORS[cls] || CLASS_COLORS.Normal;

  // Show results panel
  resultsIdle.classList.add('hidden');
  resultsContent.classList.remove('hidden');

  // ── Prediction Badge ───────────────────────────────────────────────────────
  predIcon.className = `fa-solid ${SEVERITY_ICONS[cls]}`;
  predIcon.style.color = colors.fill;
  predClass.textContent = cls;
  predBadge.style.background = colors.badge;
  predBadge.style.borderColor = colors.border;
  predClass.style.color = colors.fill;

  // ── Confidence Ring ────────────────────────────────────────────────────────
  const pct  = Math.round(conf * 100);
  const circ = 201; // 2π×32
  const dash = circ * (1 - conf);

  ringFill.style.stroke = colors.ring;
  ringPct.textContent   = `${pct}%`;
  ringPct.style.color   = colors.ring;

  // Animate ring
  requestAnimationFrame(() => {
    ringFill.style.strokeDashoffset = dash;
  });

  // ── Severity Bar ───────────────────────────────────────────────────────────
  severityBar.className = `severity-bar ${colors.severity}`;
  severityText.textContent = data.severity;

  // ── Description ────────────────────────────────────────────────────────────
  resultDesc.textContent = data.description;

  // ── Probability Bars ───────────────────────────────────────────────────────
  probBars.innerHTML = '';
  const probs = data.probabilities;
  const sorted = Object.entries(probs).sort(([, a], [, b]) => b - a);

  sorted.forEach(([name, prob]) => {
    const color  = CLASS_COLORS[name]?.fill || '#6366f1';
    const pctStr = (prob * 100).toFixed(1);
    const isTop  = name === cls;

    const row = document.createElement('div');
    row.className = 'prob-row';
    row.innerHTML = `
      <span class="prob-cls" style="${isTop ? `color:${color};font-weight:800` : ''}">${name}</span>
      <div class="prob-track">
        <div class="prob-fill" style="width:0%;background:${color}" data-target="${prob * 100}"></div>
      </div>
      <span class="prob-pct" style="${isTop ? `color:${color}` : ''}">${pctStr}%</span>
    `;
    probBars.appendChild(row);
  });

  // Animate bars after a short delay
  setTimeout(() => {
    probBars.querySelectorAll('.prob-fill').forEach(fill => {
      fill.style.width = fill.dataset.target + '%';
    });
  }, 100);

  // ── Demo Warning ───────────────────────────────────────────────────────────
  if (data.demo_mode) {
    demoWarning.classList.remove('hidden');
  } else {
    demoWarning.classList.add('hidden');
  }

  // Scroll to results
  document.getElementById('results-panel').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderError(message) {
  resultsIdle.classList.add('hidden');
  resultsContent.classList.remove('hidden');
  resultsContent.innerHTML = `
    <div style="text-align:center;padding:60px 20px;color:#ef4444">
      <i class="fa-solid fa-circle-xmark" style="font-size:3rem;margin-bottom:16px;display:block"></i>
      <strong>Analysis Failed</strong>
      <p style="margin-top:8px;font-size:0.87rem;color:#94a3b8">${message}</p>
      <button onclick="resetResults()" style="margin-top:20px;padding:10px 24px;background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);border-radius:8px;color:#ef4444;cursor:pointer;font-size:0.87rem;font-family:inherit">
        Try Again
      </button>
    </div>
  `;
}

function resetResults() {
  resultsContent.classList.add('hidden');
  resultsIdle.classList.remove('hidden');
  ringFill.style.strokeDashoffset = '201';
  probBars.innerHTML = '';
  demoWarning.classList.add('hidden');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Smooth Scroll for nav links
// ═══════════════════════════════════════════════════════════════════════════════
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', (e) => {
    e.preventDefault();
    const target = document.querySelector(anchor.getAttribute('href'));
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Intersection Observer — fade-in animations
// ═══════════════════════════════════════════════════════════════════════════════
const observerConfig = { threshold: 0.15, rootMargin: '0px 0px -50px 0px' };
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.opacity    = '1';
      entry.target.style.transform  = 'translateY(0)';
      observer.unobserve(entry.target);
    }
  });
}, observerConfig);

// Observe animate-able elements
document.querySelectorAll('.step-card, .disease-card, .arch-block, .model-stat-card, .panel').forEach(el => {
  el.style.opacity   = '0';
  el.style.transform = 'translateY(30px)';
  el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
  observer.observe(el);
});

// ═══════════════════════════════════════════════════════════════════════════════
// Init
// ═══════════════════════════════════════════════════════════════════════════════
checkApiHealth();
