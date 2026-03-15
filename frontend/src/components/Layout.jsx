import { Link, useLocation } from 'react-router-dom';
import { Home, Camera, Mic, Activity, Menu } from 'lucide-react';

export default function Layout({ children }) {
  const location = useLocation();

  const navItems = [
    { path: '/', icon: Home, label: 'Home' },
    { path: '/face', icon: Camera, label: 'Face' },
    { path: '/voice', icon: Mic, label: 'Voice' },
    { path: '/motion', icon: Activity, label: 'Motion' },
  ];

  return (
    <div className="layout">
      <header className="app-header">
        <div className="logo">Stroke Mitra<span className="beta-tag">BETA</span></div>
        <button className="menu-btn" aria-label="Menu"><Menu size={24} /></button>
      </header>

      <main className="app-content animate-fade-in">
        {children}
      </main>

      <nav className="bottom-nav">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <Link
              key={item.path}
              to={item.path}
              className={`nav-item ${isActive ? 'active' : ''}`}
            >
              <item.icon size={24} />
              <span className="nav-label">{item.label}</span>
            </Link>
          );
        })}
      </nav>

      <style>{`
        .layout {
          display: flex;
          flex-direction: column;
          height: 100vh;
          overflow: hidden;
        }

        .app-header {
          padding: var(--space-md);
          display: flex;
          justify-content: space-between;
          align-items: center;
          background: hsl(var(--bg-card));
          box-shadow: 0 1px 2px rgba(0,0,0,0.05);
          flex-shrink: 0;
        }

        .logo {
          font-weight: 700;
          font-size: 1.25rem;
          color: hsl(var(--color-primary));
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .beta-tag {
          font-size: 0.7rem;
          background: hsl(var(--color-secondary));
          color: white;
          padding: 2px 6px;
          border-radius: 4px;
          text-transform: uppercase;
        }

        .menu-btn {
          background: none;
          border: none;
          color: hsl(var(--text-main));
          cursor: pointer;
        }

        .app-content {
          flex: 1;
          overflow-y: auto;
          padding: var(--space-md);
          padding-bottom: 80px; /* Space for nav */
          max-width: 600px;
          margin: 0 auto;
          width: 100%;
        }

        .bottom-nav {
          position: fixed;
          bottom: 0;
          left: 0;
          right: 0;
          background: hsl(var(--bg-card));
          display: flex;
          justify-content: space-around;
          padding: var(--space-sm) 0;
          border-top: 1px solid rgba(0,0,0,0.05);
          box-shadow: 0 -2px 10px rgba(0,0,0,0.02);
          z-index: 100;
        }

        .nav-item {
          display: flex;
          flex-direction: column;
          align-items: center;
          text-decoration: none;
          color: hsl(var(--text-muted));
          padding: var(--space-xs);
          border-radius: var(--radius-md);
          transition: color 0.2s;
        }

        .nav-item.active {
          color: hsl(var(--color-primary));
        }

        .nav-label {
          font-size: 0.75rem;
          margin-top: 2px;
        }
      `}</style>
    </div>
  );
}
