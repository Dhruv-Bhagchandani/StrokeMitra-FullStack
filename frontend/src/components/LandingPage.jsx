import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Link } from 'react-router-dom';

/* ─────────────────────────────────────────────
   INTERSECTION OBSERVER HOOK
   ───────────────────────────────────────────── */
function useInView(options = {}) {
    const ref = useRef(null);
    const [inView, setInView] = useState(false);

    useEffect(() => {
        const el = ref.current;
        if (!el) return;
        const observer = new IntersectionObserver(
            ([entry]) => {
                if (entry.isIntersecting) {
                    setInView(true);
                    observer.unobserve(el);
                }
            },
            { threshold: 0.15, ...options }
        );
        observer.observe(el);
        return () => observer.disconnect();
    }, []);

    return [ref, inView];
}

/* ─────────────────────────────────────────────
   ANIMATED COUNTER HOOK
   ───────────────────────────────────────────── */
function useCounter(target, duration = 2000, start = false) {
    const [count, setCount] = useState(0);

    useEffect(() => {
        if (!start) return;
        let startTime = null;
        const step = (timestamp) => {
            if (!startTime) startTime = timestamp;
            const progress = Math.min((timestamp - startTime) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            setCount(Math.floor(eased * target));
            if (progress < 1) requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
    }, [start, target, duration]);

    return count;
}

/* ─────────────────────────────────────────────
   STAT COUNTER COMPONENT
   ───────────────────────────────────────────── */
function StatCounter({ value, suffix, label, started }) {
    const count = useCounter(value, 2200, started);
    return (
        <div className="lp-stat-item">
            <div className="lp-stat-number">
                {count.toLocaleString()}{suffix}
            </div>
            <div className="lp-stat-label">{label}</div>
        </div>
    );
}

/* ─────────────────────────────────────────────
   HERO SECTION
   ───────────────────────────────────────────── */
function HeroSection() {
    const [loaded, setLoaded] = useState(false);

    useEffect(() => {
        const t = setTimeout(() => setLoaded(true), 100);
        return () => clearTimeout(t);
    }, []);

    return (
        <section className="lp-hero" id="hero" aria-label="Hero">
            {/* Animated background blobs */}
            <div className="lp-hero-bg" aria-hidden="true">
                <div className="lp-blob lp-blob-1" />
                <div className="lp-blob lp-blob-2" />
                <div className="lp-blob lp-blob-3" />
                <div className="lp-pulse-ring lp-pulse-1" />
                <div className="lp-pulse-ring lp-pulse-2" />
                <div className="lp-pulse-ring lp-pulse-3" />
            </div>

            {/* Grid overlay */}
            <div className="lp-hero-grid" aria-hidden="true" />

            <div className="lp-hero-content">
                <div className={`lp-hero-badge ${loaded ? 'lp-fade-up' : ''}`} style={{ animationDelay: '0.1s' }}>
                    <span className="lp-badge-dot" />
                    AI-Powered Stroke Screening
                </div>

                <h1 className={`lp-hero-title ${loaded ? 'lp-fade-up' : ''}`} style={{ animationDelay: '0.25s' }}>
                    Detect Stroke Early.
                    <br />
                    <span className="lp-hero-title-accent">Save Lives.</span>
                </h1>

                <p className={`lp-hero-subtitle ${loaded ? 'lp-fade-up' : ''}`} style={{ animationDelay: '0.4s' }}>
                    Stroke Mitra uses your device's camera, microphone, and motion sensors
                    to screen for early stroke symptoms — in under 60 seconds.
                </p>

                <div className={`lp-hero-actions ${loaded ? 'lp-fade-up' : ''}`} style={{ animationDelay: '0.55s' }}>
                    <Link to="/face" className="lp-btn lp-btn-primary" id="hero-cta-check">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M9 12l2 2 4-4" /><circle cx="12" cy="12" r="10" /></svg>
                        Check Symptoms
                    </Link>
                    <a href="#how-it-works" className="lp-btn lp-btn-ghost" id="hero-cta-learn">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" /></svg>
                        Learn How It Works
                    </a>
                </div>

                <div className={`lp-hero-trust ${loaded ? 'lp-fade-up' : ''}`} style={{ animationDelay: '0.7s' }}>
                    <div className="lp-trust-item">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></svg>
                        100% Private
                    </div>
                    <div className="lp-trust-divider" />
                    <div className="lp-trust-item">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" /></svg>
                        Under 60 Seconds
                    </div>
                    <div className="lp-trust-divider" />
                    <div className="lp-trust-item">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07A19.5 19.5 0 013.07 9.81a19.79 19.79 0 01-3.07-8.67A2 2 0 012 .18h3a2 2 0 012 1.72c.127.96.361 1.903.7 2.81a2 2 0 01-.45 2.11L6.09 7.91a16 16 0 006 6l1.27-1.27a2 2 0 012.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0122 14.92z" /></svg>
                        No Data Stored
                    </div>
                </div>
            </div>

            {/* Floating visual card */}
            <div className={`lp-hero-visual ${loaded ? 'lp-fade-left' : ''}`} style={{ animationDelay: '0.5s' }} aria-hidden="true">
                <div className="lp-hero-card">
                    <div className="lp-hero-card-header">
                        <div className="lp-card-dot lp-dot-red" />
                        <div className="lp-card-dot lp-dot-yellow" />
                        <div className="lp-card-dot lp-dot-green" />
                        <span className="lp-card-title-bar">Stroke Mitra — Screening</span>
                    </div>
                    <div className="lp-hero-card-body">
                        <div className="lp-scan-face">
                            <div className="lp-face-outline">
                                <div className="lp-scan-line" />
                                <div className="lp-face-eye lp-eye-left" />
                                <div className="lp-face-eye lp-eye-right" />
                                <div className="lp-face-mouth" />
                            </div>
                            <div className="lp-scan-corner lp-corner-tl" />
                            <div className="lp-scan-corner lp-corner-tr" />
                            <div className="lp-scan-corner lp-corner-bl" />
                            <div className="lp-scan-corner lp-corner-br" />
                        </div>
                        <div className="lp-analysis-bars">
                            <div className="lp-bar-row">
                                <span>Facial Symmetry</span>
                                <div className="lp-bar-track"><div className="lp-bar-fill" style={{ width: '87%', animationDelay: '1s' }} /></div>
                                <span className="lp-bar-val">87%</span>
                            </div>
                            <div className="lp-bar-row">
                                <span>Speech Clarity</span>
                                <div className="lp-bar-track"><div className="lp-bar-fill lp-bar-orange" style={{ width: '72%', animationDelay: '1.2s' }} /></div>
                                <span className="lp-bar-val">72%</span>
                            </div>
                            <div className="lp-bar-row">
                                <span>Arm Stability</span>
                                <div className="lp-bar-track"><div className="lp-bar-fill lp-bar-green" style={{ width: '94%', animationDelay: '1.4s' }} /></div>
                                <span className="lp-bar-val">94%</span>
                            </div>
                        </div>
                        <div className="lp-result-badge lp-result-low">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M9 12l2 2 4-4" /><circle cx="12" cy="12" r="10" /></svg>
                            Low Risk Detected
                        </div>
                    </div>
                </div>
            </div>

            {/* Scroll indicator */}
            <a href="#what-is" className="lp-scroll-indicator" aria-label="Scroll down">
                <div className="lp-scroll-mouse">
                    <div className="lp-scroll-wheel" />
                </div>
            </a>
        </section>
    );
}

/* ─────────────────────────────────────────────
   WHAT IS STROKE MITRA
   ───────────────────────────────────────────── */
function WhatIsSection() {
    const [ref, inView] = useInView();

    const cards = [
        {
            icon: (
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></svg>
            ),
            title: 'Clinically Informed',
            desc: 'Built on the FAST (Face, Arms, Speech, Time) framework used by medical professionals worldwide.',
            color: 'blue',
        },
        {
            icon: (
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><rect x="2" y="3" width="20" height="14" rx="2" ry="2" /><line x1="8" y1="21" x2="16" y2="21" /><line x1="12" y1="17" x2="12" y2="21" /></svg>
            ),
            title: 'Device-Native AI',
            desc: 'Runs entirely on your device. No cloud uploads, no data retention. Your health data stays yours.',
            color: 'teal',
        },
        {
            icon: (
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2" /><circle cx="9" cy="7" r="4" /><path d="M23 21v-2a4 4 0 00-3-3.87" /><path d="M16 3.13a4 4 0 010 7.75" /></svg>
            ),
            title: 'For Everyone',
            desc: 'Designed for patients, caregivers, and healthcare workers. No medical training required.',
            color: 'orange',
        },
    ];

    return (
        <section className="lp-section lp-section-alt" id="what-is" aria-label="What is Stroke Mitra">
            <div className="lp-container" ref={ref}>
                <div className={`lp-section-header ${inView ? 'lp-fade-up lp-animated' : ''}`}>
                    <div className="lp-section-tag">About</div>
                    <h2 className="lp-section-title">What is Stroke Mitra?</h2>
                    <p className="lp-section-subtitle">
                        Stroke Mitra is an AI-powered screening tool that helps identify early warning signs of stroke
                        using your smartphone's built-in sensors — no special equipment needed.
                    </p>
                </div>

                <div className="lp-what-grid">
                    {cards.map((card, i) => (
                        <div
                            key={card.title}
                            className={`lp-what-card lp-what-card-${card.color} ${inView ? 'lp-fade-up lp-animated' : ''}`}
                            style={{ animationDelay: `${0.15 + i * 0.15}s` }}
                        >
                            <div className={`lp-what-icon lp-icon-${card.color}`}>{card.icon}</div>
                            <h3 className="lp-what-card-title">{card.title}</h3>
                            <p className="lp-what-card-desc">{card.desc}</p>
                        </div>
                    ))}
                </div>

                <div className={`lp-disclaimer-banner ${inView ? 'lp-fade-up lp-animated' : ''}`} style={{ animationDelay: '0.6s' }}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" /></svg>
                    <p>
                        <strong>Medical Disclaimer:</strong> Stroke Mitra is a screening aid, not a diagnostic tool.
                        Always call emergency services (112) immediately if you suspect a stroke.
                    </p>
                </div>
            </div>
        </section>
    );
}

/* ─────────────────────────────────────────────
   HOW IT WORKS
   ───────────────────────────────────────────── */
function HowItWorksSection() {
    const [ref, inView] = useInView();

    const steps = [
        {
            num: '01',
            icon: (
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z" />
                    <circle cx="12" cy="13" r="4" />
                </svg>
            ),
            title: 'Face Detection',
            desc: 'Your camera analyzes facial symmetry in real-time, checking for drooping or asymmetry — a key stroke indicator.',
            detail: 'AI scans 68 facial landmarks',
            color: 'blue',
        },
        {
            num: '02',
            icon: (
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" />
                    <path d="M19 10v2a7 7 0 01-14 0v-2" />
                    <line x1="12" y1="19" x2="12" y2="23" />
                    <line x1="8" y1="23" x2="16" y2="23" />
                </svg>
            ),
            title: 'Voice Analysis',
            desc: 'Speak a simple phrase. Our model detects slurring, hesitation, and speech irregularities associated with stroke.',
            detail: 'NLP + audio pattern analysis',
            color: 'teal',
        },
        {
            num: '03',
            icon: (
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                </svg>
            ),
            title: 'Motion & Coordination',
            desc: 'Using your device\'s gyroscope and accelerometer, we assess arm stability and coordination.',
            detail: 'Gyroscope + accelerometer data',
            color: 'orange',
        },
        {
            num: '04',
            icon: (
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
                    <polyline points="14 2 14 8 20 8" />
                    <line x1="16" y1="13" x2="8" y2="13" />
                    <line x1="16" y1="17" x2="8" y2="17" />
                    <polyline points="10 9 9 9 8 9" />
                </svg>
            ),
            title: 'Instant Results',
            desc: 'Receive a risk assessment in seconds with clear guidance on next steps — including when to call emergency services.',
            detail: 'Results in under 60 seconds',
            color: 'green',
        },
    ];

    return (
        <section className="lp-section" id="how-it-works" aria-label="How Stroke Mitra Works">
            <div className="lp-container" ref={ref}>
                <div className={`lp-section-header ${inView ? 'lp-fade-up lp-animated' : ''}`}>
                    <div className="lp-section-tag">Process</div>
                    <h2 className="lp-section-title">How Stroke Mitra Works</h2>
                    <p className="lp-section-subtitle">
                        Four simple steps. No medical expertise required. Guided by AI every step of the way.
                    </p>
                </div>

                <div className="lp-steps-container">
                    {steps.map((step, i) => (
                        <div
                            key={step.num}
                            className={`lp-step ${inView ? 'lp-fade-up lp-animated' : ''}`}
                            style={{ animationDelay: `${0.1 + i * 0.15}s` }}
                        >
                            <div className={`lp-step-icon-wrap lp-step-icon-${step.color}`}>
                                {step.icon}
                                <div className="lp-step-num">{step.num}</div>
                            </div>
                            {i < steps.length - 1 && <div className="lp-step-connector" aria-hidden="true" />}
                            <div className="lp-step-content">
                                <h3 className="lp-step-title">{step.title}</h3>
                                <p className="lp-step-desc">{step.desc}</p>
                                <div className={`lp-step-detail lp-detail-${step.color}`}>
                                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><polyline points="9 18 15 12 9 6" /></svg>
                                    {step.detail}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
}

/* ─────────────────────────────────────────────
   KEY FEATURES
   ───────────────────────────────────────────── */
function FeaturesSection() {
    const [ref, inView] = useInView();

    const features = [
        {
            icon: (
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z" />
                    <circle cx="12" cy="13" r="4" />
                </svg>
            ),
            title: 'Camera-Based Facial Analysis',
            desc: 'Real-time AI detection of facial asymmetry using your front camera. No special hardware needed.',
            tag: 'Computer Vision',
        },
        {
            icon: (
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" />
                    <path d="M19 10v2a7 7 0 01-14 0v-2" />
                    <line x1="12" y1="19" x2="12" y2="23" />
                    <line x1="8" y1="23" x2="16" y2="23" />
                </svg>
            ),
            title: 'Voice Recording & Speech Detection',
            desc: 'Advanced NLP models analyze your speech for slurring, word-finding difficulty, and incoherence.',
            tag: 'NLP + Audio AI',
        },
        {
            icon: (
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                </svg>
            ),
            title: 'Motion & Coordination Sensing',
            desc: 'Gyroscope and accelerometer data assess arm drift and coordination — key neurological indicators.',
            tag: 'Sensor Fusion',
        },
        {
            icon: (
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <circle cx="12" cy="12" r="10" />
                    <polyline points="12 6 12 12 16 14" />
                </svg>
            ),
            title: 'Results in Under 60 Seconds',
            desc: 'The entire screening process takes less than a minute, giving you fast answers when every second counts.',
            tag: 'Real-Time',
        },
        {
            icon: (
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                    <path d="M7 11V7a5 5 0 0110 0v4" />
                </svg>
            ),
            title: 'Fully Private & Secure',
            desc: 'All processing happens on your device. No video, audio, or personal data is ever uploaded or stored.',
            tag: 'On-Device AI',
        },
        {
            icon: (
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                </svg>
            ),
            title: 'Clinically Guided Framework',
            desc: 'Based on the medically validated FAST protocol, trusted by emergency responders globally.',
            tag: 'Evidence-Based',
        },
    ];

    return (
        <section className="lp-section lp-section-alt" id="features" aria-label="Key Features">
            <div className="lp-container" ref={ref}>
                <div className={`lp-section-header ${inView ? 'lp-fade-up lp-animated' : ''}`}>
                    <div className="lp-section-tag">Features</div>
                    <h2 className="lp-section-title">Built for Speed. Designed for Trust.</h2>
                    <p className="lp-section-subtitle">
                        Every feature is purpose-built to deliver accurate, fast, and private stroke screening.
                    </p>
                </div>

                <div className="lp-features-grid">
                    {features.map((feat, i) => (
                        <div
                            key={feat.title}
                            className={`lp-feature-card ${inView ? 'lp-fade-up lp-animated' : ''}`}
                            style={{ animationDelay: `${0.1 + i * 0.1}s` }}
                            id={`feature-card-${i}`}
                        >
                            <div className="lp-feature-icon">{feat.icon}</div>
                            <div className="lp-feature-tag">{feat.tag}</div>
                            <h3 className="lp-feature-title">{feat.title}</h3>
                            <p className="lp-feature-desc">{feat.desc}</p>
                            <div className="lp-feature-hover-line" aria-hidden="true" />
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
}

/* ─────────────────────────────────────────────
   WHY EARLY DETECTION MATTERS
   ───────────────────────────────────────────── */
function StatsSection() {
    const [ref, inView] = useInView();

    const stats = [
        { value: 15000000, suffix: '+', label: 'Strokes occur globally each year', color: 'blue' },
        { value: 80, suffix: '%', label: 'Of strokes are preventable with early action', color: 'teal' },
        { value: 1900000, suffix: '', label: 'Brain cells lost every minute untreated', color: 'orange' },
        { value: 3, suffix: 'x', label: 'Better outcomes with treatment in first hour', color: 'green' },
    ];

    return (
        <section className="lp-section lp-section-dark" id="why-early-detection" aria-label="Why Early Detection Matters">
            <div className="lp-stats-bg" aria-hidden="true">
                <div className="lp-stats-blob-1" />
                <div className="lp-stats-blob-2" />
            </div>
            <div className="lp-container" ref={ref}>
                <div className={`lp-section-header lp-header-light ${inView ? 'lp-fade-up lp-animated' : ''}`}>
                    <div className="lp-section-tag lp-tag-light">The Stakes</div>
                    <h2 className="lp-section-title lp-title-light">Why Every Second Counts</h2>
                    <p className="lp-section-subtitle lp-subtitle-light">
                        Stroke is the second leading cause of death worldwide. Early detection dramatically changes outcomes.
                    </p>
                </div>

                <div className="lp-stats-grid">
                    {stats.map((stat, i) => (
                        <div
                            key={stat.label}
                            className={`lp-stat-card lp-stat-card-${stat.color} ${inView ? 'lp-fade-up lp-animated' : ''}`}
                            style={{ animationDelay: `${0.15 + i * 0.15}s` }}
                        >
                            <StatCounter
                                value={stat.value}
                                suffix={stat.suffix}
                                label={stat.label}
                                started={inView}
                            />
                        </div>
                    ))}
                </div>

                <div className={`lp-fast-banner ${inView ? 'lp-fade-up lp-animated' : ''}`} style={{ animationDelay: '0.75s' }}>
                    <h3 className="lp-fast-title">Remember <span className="lp-fast-acronym">F.A.S.T.</span></h3>
                    <div className="lp-fast-grid">
                        {[
                            { letter: 'F', word: 'Face', desc: 'Is one side drooping?' },
                            { letter: 'A', word: 'Arms', desc: 'Can they raise both arms?' },
                            { letter: 'S', word: 'Speech', desc: 'Is speech slurred or strange?' },
                            { letter: 'T', word: 'Time', desc: 'Call 112 immediately!' },
                        ].map((item) => (
                            <div key={item.letter} className="lp-fast-item">
                                <div className="lp-fast-letter">{item.letter}</div>
                                <div className="lp-fast-word">{item.word}</div>
                                <div className="lp-fast-desc">{item.desc}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </section>
    );
}

/* ─────────────────────────────────────────────
   CTA SECTION
   ───────────────────────────────────────────── */
function CTASection() {
    const [ref, inView] = useInView();

    return (
        <section className="lp-section lp-section-cta" id="cta" aria-label="Call to Action">
            <div className="lp-cta-bg" aria-hidden="true">
                <div className="lp-cta-blob-1" />
                <div className="lp-cta-blob-2" />
                <div className="lp-cta-rings">
                    <div className="lp-cta-ring lp-cta-ring-1" />
                    <div className="lp-cta-ring lp-cta-ring-2" />
                    <div className="lp-cta-ring lp-cta-ring-3" />
                </div>
            </div>
            <div className="lp-container" ref={ref}>
                <div className={`lp-cta-content ${inView ? 'lp-fade-up lp-animated' : ''}`}>
                    <div className="lp-cta-icon-wrap" aria-hidden="true">
                        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07A19.5 19.5 0 013.07 9.81a19.79 19.79 0 01-3.07-8.67A2 2 0 012 .18h3a2 2 0 012 1.72c.127.96.361 1.903.7 2.81a2 2 0 01-.45 2.11L6.09 7.91a16 16 0 006 6l1.27-1.27a2 2 0 012.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0122 14.92z" />
                        </svg>
                    </div>
                    <h2 className="lp-cta-title">
                        Don't Wait. <span className="lp-cta-accent">Act Now.</span>
                    </h2>
                    <p className="lp-cta-subtitle">
                        If you or someone near you shows stroke symptoms, every second matters.
                        Start the Stroke Mitra check right now — it could save a life.
                    </p>
                    <div className="lp-cta-actions">
                        <Link to="/face" className="lp-btn lp-btn-cta-primary" id="cta-start-check">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M9 12l2 2 4-4" /><circle cx="12" cy="12" r="10" /></svg>
                            Start Stroke Check Now
                        </Link>
                        <a href="tel:112" className="lp-btn lp-btn-emergency" id="cta-emergency">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07A19.5 19.5 0 013.07 9.81a19.79 19.79 0 01-3.07-8.67A2 2 0 012 .18h3a2 2 0 012 1.72c.127.96.361 1.903.7 2.81a2 2 0 01-.45 2.11L6.09 7.91a16 16 0 006 6l1.27-1.27a2 2 0 012.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0122 14.92z" /></svg>
                            Call 112 Emergency
                        </a>
                    </div>
                    <p className="lp-cta-note">
                        Free to use · No registration · Works on any modern smartphone
                    </p>
                </div>
            </div>
        </section>
    );
}

/* ─────────────────────────────────────────────
   FOOTER
   ───────────────────────────────────────────── */
function Footer() {
    return (
        <footer className="lp-footer" aria-label="Footer">
            <div className="lp-container">
                <div className="lp-footer-top">
                    <div className="lp-footer-brand">
                        <div className="lp-footer-logo">
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M22 12h-4l-3 9L9 3l-3 9H2" /></svg>
                            Stroke Mitra
                        </div>
                        <p className="lp-footer-tagline">
                            Early detection. Better outcomes. Every second counts.
                        </p>
                    </div>

                    <div className="lp-footer-links">
                        <div className="lp-footer-col">
                            <h4 className="lp-footer-col-title">Screening</h4>
                            <Link to="/face" className="lp-footer-link">Face Analysis</Link>
                            <Link to="/voice" className="lp-footer-link">Speech Check</Link>
                            <Link to="/motion" className="lp-footer-link">Motion Test</Link>
                        </div>
                        <div className="lp-footer-col">
                            <h4 className="lp-footer-col-title">Learn</h4>
                            <a href="#what-is" className="lp-footer-link">What is Stroke Mitra</a>
                            <a href="#how-it-works" className="lp-footer-link">How It Works</a>
                            <a href="#why-early-detection" className="lp-footer-link">Why Early Detection</a>
                        </div>
                        <div className="lp-footer-col">
                            <h4 className="lp-footer-col-title">Emergency</h4>
                            <a href="tel:112" className="lp-footer-link lp-link-emergency">🚨 Call 112</a>
                            <a href="tel:108" className="lp-footer-link lp-link-emergency">🚑 Ambulance 108</a>
                        </div>
                    </div>
                </div>

                <div className="lp-footer-bottom">
                    <p className="lp-footer-copy">
                        © 2025 Stroke Mitra. Built with care for public health awareness.
                    </p>
                    <p className="lp-footer-disclaimer">
                        This tool is for screening purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
                    </p>
                </div>
            </div>
        </footer>
    );
}

/* ─────────────────────────────────────────────
   NAVBAR
   ───────────────────────────────────────────── */
function LandingNav() {
    const [scrolled, setScrolled] = useState(false);
    const [menuOpen, setMenuOpen] = useState(false);

    useEffect(() => {
        const onScroll = () => setScrolled(window.scrollY > 20);
        window.addEventListener('scroll', onScroll, { passive: true });
        return () => window.removeEventListener('scroll', onScroll);
    }, []);

    const closeMenu = useCallback(() => setMenuOpen(false), []);

    return (
        <nav className={`lp-nav ${scrolled ? 'lp-nav-scrolled' : ''}`} role="navigation" aria-label="Main navigation">
            <div className="lp-nav-inner">
                <a href="#hero" className="lp-nav-logo" aria-label="Stroke Mitra home">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M22 12h-4l-3 9L9 3l-3 9H2" /></svg>
                    Stroke Mitra
                </a>

                <div className={`lp-nav-links ${menuOpen ? 'lp-nav-open' : ''}`}>
                    <a href="#what-is" className="lp-nav-link" onClick={closeMenu}>About</a>
                    <a href="#how-it-works" className="lp-nav-link" onClick={closeMenu}>How It Works</a>
                    <a href="#features" className="lp-nav-link" onClick={closeMenu}>Features</a>
                    <a href="#why-early-detection" className="lp-nav-link" onClick={closeMenu}>Why It Matters</a>
                    <Link to="/face" className="lp-btn lp-btn-nav" onClick={closeMenu} id="nav-cta">Check Symptoms</Link>
                </div>

                <button
                    className="lp-nav-hamburger"
                    onClick={() => setMenuOpen(!menuOpen)}
                    aria-label={menuOpen ? 'Close menu' : 'Open menu'}
                    aria-expanded={menuOpen}
                >
                    <span className={`lp-ham-line ${menuOpen ? 'lp-ham-open' : ''}`} />
                    <span className={`lp-ham-line ${menuOpen ? 'lp-ham-open' : ''}`} />
                    <span className={`lp-ham-line ${menuOpen ? 'lp-ham-open' : ''}`} />
                </button>
            </div>
        </nav>
    );
}

/* ─────────────────────────────────────────────
   MAIN LANDING PAGE
   ───────────────────────────────────────────── */
export default function LandingPage() {
    // Smooth scroll for anchor links
    useEffect(() => {
        const handleClick = (e) => {
            const target = e.target.closest('a[href^="#"]');
            if (!target) return;
            e.preventDefault();
            const id = target.getAttribute('href').slice(1);
            const el = document.getElementById(id);
            if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        };
        document.addEventListener('click', handleClick);
        return () => document.removeEventListener('click', handleClick);
    }, []);

    return (
        <div className="lp-root">
            <LandingNav />
            <HeroSection />
            <WhatIsSection />
            <HowItWorksSection />
            <FeaturesSection />
            <StatsSection />
            <CTASection />
            <Footer />
        </div>
    );
}
