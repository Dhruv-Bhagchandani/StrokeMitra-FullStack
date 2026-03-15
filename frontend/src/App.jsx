import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Layout from './components/Layout';
import Disclaimer from './components/Disclaimer';
import LandingPage from './components/LandingPage';
import { Camera, Mic, Activity, ArrowRight } from 'lucide-react';
import './landing.css';

const CameraModule = lazy(() => import('./components/CameraModule'));
const VoiceModule = lazy(() => import('./components/VoiceModule'));
const MotionModule = lazy(() => import('./components/MotionModule'));

const LoadingFallback = () => (
  <div className="loading-spinner">
    <div className="spinner"></div>
    <p>Loading Module...</p>
    <style>{`
      .loading-spinner {
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        height: 300px; color: hsl(var(--text-muted));
      }
      .spinner {
        width: 40px; height: 40px;
        border: 4px solid hsl(var(--bg-app));
        border-top: 4px solid hsl(var(--color-primary));
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: var(--space-md);
      }
      @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    `}</style>
  </div>
);

const AppHome = () => (
  <div className="home-container">
    <div className="hero">
      <h1>Early Detection Saves Lives</h1>
      <p>Perform a quick self-check if you suspect symptoms.</p>
    </div>
    <div className="action-grid">
      <Link to="/face" className="action-card">
        <div className="icon-box"><Camera size={32} /></div>
        <h3>Face Analysis</h3>
        <p>Check for facial drooping</p>
        <ArrowRight size={20} className="arrow" />
      </Link>
      <Link to="/voice" className="action-card">
        <div className="icon-box"><Mic size={32} /></div>
        <h3>Speech Check</h3>
        <p>Analyze speech clarity</p>
        <ArrowRight size={20} className="arrow" />
      </Link>
      <Link to="/motion" className="action-card">
        <div className="icon-box"><Activity size={32} /></div>
        <h3>Motion Test</h3>
        <p>Assess arm stability</p>
        <ArrowRight size={20} className="arrow" />
      </Link>
    </div>
    <Disclaimer />
    <style>{`
      .home-container { padding-top: var(--space-md); }
      .hero { text-align: center; margin-bottom: var(--space-xl); }
      .action-grid { display: grid; gap: var(--space-md); }
      .action-card {
        background: hsl(var(--bg-card)); padding: var(--space-lg);
        border-radius: var(--radius-lg); text-decoration: none;
        color: hsl(var(--text-main)); display: grid;
        grid-template-columns: auto 1fr auto; align-items: center;
        gap: var(--space-md); box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        transition: transform 0.2s, box-shadow 0.2s; border: 1px solid transparent;
      }
      .action-card:hover { transform: translateY(-2px); box-shadow: 0 8px 15px rgba(0,0,0,0.05); border-color: hsl(var(--color-primary-light)); }
      .icon-box { background: hsl(var(--bg-app)); width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: hsl(var(--color-primary)); }
      .action-card h3 { font-size: 1.1rem; margin: 0; }
      .action-card p  { font-size: 0.85rem; margin: 0; color: hsl(var(--text-muted)); }
      .arrow { color: hsl(var(--text-muted)); opacity: 0.5; transition: opacity 0.2s; }
      .action-card:hover .arrow { opacity: 1; color: hsl(var(--color-primary)); }
    `}</style>
  </div>
);

export default function App() {
  return (
    <Router>
      <Routes>
        {/* Landing page — no Layout wrapper */}
        <Route path="/" element={<LandingPage />} />

        {/* App screens — wrapped in Layout */}
        <Route path="/app" element={<Layout><AppHome /></Layout>} />
        <Route path="/face" element={<Layout><Suspense fallback={<LoadingFallback />}><CameraModule /></Suspense></Layout>} />
        <Route path="/voice" element={<Layout><Suspense fallback={<LoadingFallback />}><VoiceModule /></Suspense></Layout>} />
        <Route path="/motion" element={<Layout><Suspense fallback={<LoadingFallback />}><MotionModule /></Suspense></Layout>} />
      </Routes>
    </Router>
  );
}
