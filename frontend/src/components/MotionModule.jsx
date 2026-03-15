import { useState, useEffect, useRef } from 'react';
import { Activity, Play, Octagon } from 'lucide-react';

const MotionModule = () => {
    const [isTracking, setIsTracking] = useState(false);
    const isTrackingRef = useRef(false);
    const [coords, setCoords] = useState({ x: 0, y: 0, z: 0 });
    const ballRef = useRef(null);

    const handleMotion = (event) => {
        if (!isTrackingRef.current) return;

        // Acceleration including gravity gives a "tilt" feel effectively for 1g
        const acc = event.accelerationIncludingGravity;
        if (!acc) return;

        const x = acc.x || 0;
        const y = acc.y || 0;
        const z = acc.z || 0;

        setCoords({
            x: x.toFixed(2),
            y: y.toFixed(2),
            z: z.toFixed(2)
        });

        // Visual feedback - simple tilt visualization
        if (ballRef.current) {
            // Sensitivity
            const moveX = x * 10;
            const moveY = y * 10;
            // Limit movement
            const limitedX = Math.max(-80, Math.min(80, moveX));
            const limitedY = Math.max(-80, Math.min(80, moveY));
            ballRef.current.style.transform = `translate(${limitedX}px, ${-limitedY}px)`;
        }
    };

    const requestPermission = async () => {
        if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
            try {
                const response = await DeviceMotionEvent.requestPermission();
                if (response === 'granted') {
                    startTracking();
                } else {
                    alert('Permission needed for motion analysis.');
                }
            } catch (e) {
                console.error(e);
            }
        } else {
            startTracking();
        }
    };

    const startTracking = () => {
        setIsTracking(true);
        isTrackingRef.current = true;
        window.addEventListener('devicemotion', handleMotion);
    };

    const stopTracking = () => {
        setIsTracking(false);
        isTrackingRef.current = false;
        window.removeEventListener('devicemotion', handleMotion);
        setCoords({ x: 0, y: 0, z: 0 });
        if (ballRef.current) ballRef.current.style.transform = 'translate(0,0)';
    };

    useEffect(() => {
        return () => {
            window.removeEventListener('devicemotion', handleMotion);
            isTrackingRef.current = false;
        };
    }, []);

    return (
        <div className="motion-container">
            <h2><Activity size={24} className="inline-icon" /> Motion Analysis</h2>
            <p className="description">
                Hold your device flat. Press start and gently tilt your device.
            </p>

            <div className="feedback-area">
                <div className="center-target"></div>
                <div className="moving-ball" ref={ballRef}></div>
            </div>

            <div className="metrics-grid">
                <div className="metric-box">
                    <span className="label">X - Axis</span>
                    <span className="value">{coords.x}</span>
                </div>
                <div className="metric-box">
                    <span className="label">Y - Axis</span>
                    <span className="value">{coords.y}</span>
                </div>
                <div className="metric-box">
                    <span className="label">Z - Axis</span>
                    <span className="value">{coords.z}</span>
                </div>
            </div>

            <div className="controls">
                {!isTracking ? (
                    <button className="btn btn-primary btn-record" onClick={requestPermission}>
                        <Play size={20} /> Start Test
                    </button>
                ) : (
                    <button className="btn btn-secondary btn-stop" onClick={stopTracking}>
                        <Octagon size={20} /> Stop
                    </button>
                )}
            </div>

            <style>{`
                .motion-container {
                    display: flex; 
                    flex-direction: column;
                    align-items: center;
                }
                .inline-icon { vertical-align: middle; margin-right: 8px; color: hsl(var(--color-primary)); }
                .description { font-size: 0.9rem; margin-bottom: var(--space-lg); color: hsl(var(--text-muted)); }

                .feedback-area {
                    width: 200px;
                    height: 200px;
                    border-radius: 50%;
                    border: 4px solid #eee;
                    position: relative;
                    margin: var(--space-lg) 0;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: hsl(var(--bg-card));
                    box-shadow: inset 0 2px 10px rgba(0,0,0,0.05);
                }
                .center-target {
                    width: 12px;
                    height: 12px;
                    background: #ddd;
                    border-radius: 50%;
                }
                .moving-ball {
                    width: 24px;
                    height: 24px;
                    background: hsl(var(--color-primary));
                    border-radius: 50%;
                    position: absolute;
                    transition: transform 0.1s linear; 
                    box-shadow: 0 4px 10px rgba(46, 134, 176, 0.4);
                }
                
                .metrics-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr 1fr;
                    gap: var(--space-md);
                    width: 100%;
                    margin-bottom: var(--space-lg);
                }
                .metric-box {
                    background: hsl(var(--bg-card));
                    padding: var(--space-md);
                    border-radius: var(--radius-md);
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
                    border: 1px solid rgba(0,0,0,0.03);
                }
                .metric-box .label { display: block; font-size: 0.7rem; color: hsl(var(--text-muted)); margin-bottom: 4px; }
                .metric-box .value { font-size: 1.1rem; font-weight: 700; color: hsl(var(--color-secondary)); font-family: monospace; }
                
                .btn-record { width: 100%; max-width: 300px; padding: var(--space-lg); border-radius: 50px; }
                .btn-stop { width: 100%; max-width: 300px; padding: var(--space-lg); border-radius: 50px; color: hsl(var(--status-error)); border-color: hsl(var(--status-error)); }
                .btn-stop:hover { background: hsl(var(--status-error), 0.1); }
            `}</style>
        </div>
    );
};

export default MotionModule;
