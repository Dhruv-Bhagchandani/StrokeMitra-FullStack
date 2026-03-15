import { useState, useRef, useEffect } from 'react';
import { Camera, RefreshCw, CheckCircle, AlertTriangle } from 'lucide-react';
import { submitData } from '../api';

const CameraModule = () => {
    const videoRef = useRef(null);
    const streamRef = useRef(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [permissionGranted, setPermissionGranted] = useState(false);

    const startCamera = async () => {
        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }
            });
            streamRef.current = mediaStream;
            if (videoRef.current) {
                videoRef.current.srcObject = mediaStream;
            }
            setPermissionGranted(true);
            setError(null);
        } catch (err) {
            console.error(err);
            setError('Camera access denied. Please allow camera permissions to proceed.');
        }
    };

    const stopCamera = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
            setPermissionGranted(false);
        }
    };

    const analyzeFace = () => {
        setIsAnalyzing(true);
        // Mock analysis - wait 2 seconds
        setTimeout(() => {
            const mockResult = {
                symmetry: 0.98,
                status: 'Normal',
                message: 'No significant asymmetry detected.',
                confidence: 0.95
            };
            setResult(mockResult);
            setIsAnalyzing(false);
            submitData('temp-session', 'face', mockResult); // Fire and forget
        }, 2000);
    };

    const reset = () => {
        setResult(null);
        setIsAnalyzing(false);
    };

    useEffect(() => {
        return () => {
            stopCamera();
        };
    }, []);

    return (
        <div className="module-container">
            <h2><Camera size={24} className="inline-icon" /> Face Analysis</h2>
            <p className="description">Align your face within the frame. Maintain a neutral expression.</p>

            <div className="camera-frame-container">
                {!permissionGranted ? (
                    <div className="camera-placeholder" onClick={startCamera}>
                        <Camera size={48} />
                        <span>Tap to Enable Camera</span>
                    </div>
                ) : (
                    <>
                        <video
                            ref={videoRef}
                            autoPlay
                            playsInline
                            className={`camera-feed ${isAnalyzing ? 'scanning' : ''}`}
                        />
                        <div className="guide-overlay">
                            <div className="face-oval"></div>
                            <div className="grid-lines"></div>
                        </div>
                    </>
                )}
            </div>

            {error && <div className="error-message"><AlertTriangle size={16} /> {error}</div>}

            <div className="controls">
                {!permissionGranted ? (
                    <button className="btn btn-primary" onClick={startCamera}>Grant Access</button>
                ) : !result ? (
                    <button
                        className="btn btn-primary action-btn"
                        onClick={analyzeFace}
                        disabled={isAnalyzing}
                    >
                        {isAnalyzing ? 'Analyzing...' : 'Capture & Analyze'}
                    </button>
                ) : (
                    <div className="result-card animate-fade-in">
                        <div className="result-header">
                            <CheckCircle size={24} color="var(--status-success)" />
                            <h3>Analysis Complete</h3>
                        </div>
                        <div className="result-body">
                            <div className="metric">
                                <span className="label">Symmetry Score</span>
                                <span className="value">{(result.symmetry * 100).toFixed(0)}%</span>
                            </div>
                            <p className="result-message">{result.message}</p>
                        </div>
                        <button className="btn btn-secondary" onClick={reset}>
                            <RefreshCw size={16} /> Retake
                        </button>
                    </div>
                )}
            </div>

            <style>{`
                .module-container {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-md);
                }
                .inline-icon { vertical-align: middle; margin-right: 8px; color: hsl(var(--color-primary)); }
                .description { font-size: 0.9rem; margin-bottom: var(--space-lg); }

                .camera-frame-container {
                    position: relative;
                    width: 100%;
                    aspect-ratio: 4/3;
                    background: #eee;
                    border-radius: var(--radius-lg);
                    overflow: hidden;
                    box-shadow: inset 0 0 20px rgba(0,0,0,0.1);
                }

                .camera-placeholder {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 100%;
                    color: hsl(var(--text-muted));
                    cursor: pointer;
                    background: #f0f2f5;
                    transition: background 0.2s;
                }
                .camera-placeholder:hover { background: #e5e8eb; }

                .camera-feed {
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                    transform: scaleX(-1); /* Mirror effect */
                }
                
                .scanning {
                    filter: sepia(0.2) contrast(1.1);
                    transition: filter 2s;
                }

                .guide-overlay {
                    position: absolute;
                    top: 0; left: 0; right: 0; bottom: 0;
                    pointer-events: none;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }

                .face-oval {
                    width: 50%;
                    height: 70%;
                    border: 2px dashed rgba(255, 255, 255, 0.7);
                    border-radius: 50%;
                    box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.5); /* Dim outside */
                }

                .error-message {
                    color: hsl(var(--status-error));
                    background: rgba(255,0,0,0.1);
                    padding: var(--space-sm);
                    border-radius: var(--radius-sm);
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    font-size: 0.9rem;
                }

                .controls {
                    display: flex;
                    justify-content: center;
                    margin-top: var(--space-md);
                }

                .action-btn { width: 100%; padding: var(--space-lg); font-size: 1.1rem; }

                .result-card {
                    background: hsl(var(--bg-card));
                    border: 1px solid rgba(0,0,0,0.05);
                    padding: var(--space-md);
                    border-radius: var(--radius-md);
                    width: 100%;
                    text-align: center;
                }
                .result-header {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 8px;
                    margin-bottom: var(--space-sm);
                    color: hsl(var(--status-success));
                }
                .metric {
                    display: flex;
                    justify-content: space-between;
                    padding: var(--space-sm) 0;
                    border-bottom: 1px solid #eee;
                    margin-bottom: var(--space-sm);
                }
                .value { font-weight: 700; color: hsl(var(--text-main)); }
            `}</style>
        </div>
    );
};

export default CameraModule;
