import { useState, useRef } from 'react';
import { Mic, Square, BarChart2, Loader, AlertTriangle, CheckCircle } from 'lucide-react';
import { analyzeSpeech } from '../api';

const VoiceModule = () => {
    const [isRecording, setIsRecording] = useState(false);
    const [audioUrl, setAudioUrl] = useState(null);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [analysisError, setAnalysisError] = useState(null);
    const mediaRecorderRef = useRef(null);
    const chunksRef = useRef([]);
    const audioBlobRef = useRef(null);

    const startRun = async () => {
        try {
            // Reset previous results
            setAnalysisResult(null);
            setAnalysisError(null);

            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            // Try different MIME types in order of preference
            const mimeTypes = [
                'audio/webm;codecs=opus',
                'audio/webm',
                'audio/ogg;codecs=opus',
                'audio/mp4'
            ];

            let selectedMimeType = 'audio/webm';
            for (const mimeType of mimeTypes) {
                if (MediaRecorder.isTypeSupported(mimeType)) {
                    selectedMimeType = mimeType;
                    break;
                }
            }

            console.log('Using MIME type:', selectedMimeType);
            mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: selectedMimeType });

            // Clear any previous chunks
            chunksRef.current = [];

            mediaRecorderRef.current.ondataavailable = (e) => {
                if (e.data && e.data.size > 0) {
                    console.log('Chunk received:', e.data.size, 'bytes');
                    chunksRef.current.push(e.data);
                }
            };

            mediaRecorderRef.current.onstop = async () => {
                console.log('Recording stopped. Total chunks:', chunksRef.current.length);
                const blob = new Blob(chunksRef.current, { type: selectedMimeType });
                console.log('Final blob size:', blob.size, 'bytes');

                audioBlobRef.current = blob;
                const url = URL.createObjectURL(blob);
                setAudioUrl(url);
                chunksRef.current = [];

                // Automatically analyze after recording stops
                await handleAnalysis(blob);
            };

            // Start recording and request data every 100ms to ensure we capture everything
            mediaRecorderRef.current.start(100);
            setIsRecording(true);
        } catch (err) {
            console.error(err);
            alert('Microphone access denied.');
        }
    };

    const stopRun = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
            // Stop tracks
            mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
        }
    };

    const handleAnalysis = async (audioBlob) => {
        setIsAnalyzing(true);
        setAnalysisError(null);

        try {
            // Call the slurry_speech API
            const result = await analyzeSpeech(audioBlob);
            setAnalysisResult(result);
        } catch (error) {
            console.error('Analysis failed:', error);
            setAnalysisError(error.message || 'Analysis failed. Please try again.');
        } finally {
            setIsAnalyzing(false);
        }
    };

    const getSeverityColor = (severity) => {
        const colors = {
            'none': '#10b981',
            'mild': '#f59e0b',
            'moderate': '#ef4444',
            'severe': '#dc2626'
        };
        return colors[severity] || '#6b7280';
    };

    const getRiskColor = (riskTier) => {
        const colors = {
            'low': '#10b981',
            'moderate': '#f59e0b',
            'high': '#ef4444',
            'critical': '#dc2626'
        };
        return colors[riskTier] || '#6b7280';
    };

    return (
        <div className="voice-container">
            <h2><Mic size={24} className="inline-icon" /> Voice Check - AI Analysis</h2>
            <p className="description">Read the following sentence clearly for 5-10 seconds:</p>

            <div className="prompt-card">
                "The quick brown fox jumps over the lazy dog. Peter Piper picked a peck of pickled peppers."
            </div>

            <div className={`visualizer ${isRecording ? 'active' : ''}`}>
                <div className="bar"></div>
                <div className="bar"></div>
                <div className="bar"></div>
                <div className="bar"></div>
                <div className="bar"></div>
            </div>

            <div className="controls">
                {!isRecording ? (
                    <button className="btn btn-primary btn-record" onClick={startRun} disabled={isAnalyzing}>
                        <Mic size={20} /> {audioUrl ? 'Record Again' : 'Start Recording'}
                    </button>
                ) : (
                    <button className="btn btn-danger btn-stop" onClick={stopRun}>
                        <Square size={20} /> Stop Recording
                    </button>
                )}
            </div>

            {audioUrl && (
                <div className="playback-area animate-fade-in">
                    <audio controls src={audioUrl} className="audio-player" />
                </div>
            )}

            {isAnalyzing && (
                <div className="analyzing-indicator animate-fade-in">
                    <Loader size={20} className="spinner" />
                    <span>Analyzing speech with AI model...</span>
                </div>
            )}

            {analysisError && (
                <div className="error-message animate-fade-in">
                    <AlertTriangle size={20} />
                    <span>{analysisError}</span>
                </div>
            )}

            {analysisResult && !isAnalyzing && (
                <div className="analysis-results animate-fade-in">
                    <h3><BarChart2 size={20} /> Analysis Results</h3>

                    {/* Emergency Alert */}
                    {analysisResult.emergency_alert && (
                        <div className="emergency-alert">
                            <AlertTriangle size={24} />
                            <div>
                                <strong>CRITICAL: Immediate Medical Attention Required</strong>
                                <p>Please seek emergency medical care immediately</p>
                            </div>
                        </div>
                    )}

                    {/* Slurring Score */}
                    <div className="result-card">
                        <div className="result-label">Slurring Score</div>
                        <div className="score-display">
                            <div className="score-value" style={{ color: getSeverityColor(analysisResult.severity) }}>
                                {analysisResult.slurring_score.toFixed(1)}/100
                            </div>
                            <div className="score-bar">
                                <div
                                    className="score-fill"
                                    style={{
                                        width: `${analysisResult.slurring_score}%`,
                                        backgroundColor: getSeverityColor(analysisResult.severity)
                                    }}
                                />
                            </div>
                        </div>
                        <div className="severity-badge" style={{ backgroundColor: getSeverityColor(analysisResult.severity) }}>
                            {analysisResult.severity.toUpperCase()}
                        </div>
                    </div>

                    {/* Risk Assessment */}
                    <div className="result-card">
                        <div className="result-label">Risk Assessment</div>
                        <div className="risk-display">
                            <div className="risk-score" style={{ color: getRiskColor(analysisResult.risk_tier) }}>
                                {analysisResult.risk_score.toFixed(1)}/100
                            </div>
                            <div className="risk-badge" style={{ backgroundColor: getRiskColor(analysisResult.risk_tier) }}>
                                {analysisResult.risk_tier.toUpperCase()} RISK
                            </div>
                        </div>
                    </div>

                    {/* Acoustic Summary */}
                    {analysisResult.acoustic_summary && (
                        <div className="acoustic-details">
                            <div className="detail-label">Acoustic Analysis:</div>
                            <div className="details-grid">
                                <div className="detail-item">
                                    <span className="detail-name">Speaking Rate:</span>
                                    <span className="detail-value">
                                        {analysisResult.acoustic_summary.speaking_rate_syllables_per_sec?.toFixed(2)} syl/s
                                    </span>
                                </div>
                                <div className="detail-item">
                                    <span className="detail-name">Pitch (Mean):</span>
                                    <span className="detail-value">
                                        {analysisResult.acoustic_summary.pitch_mean_hz?.toFixed(1)} Hz
                                    </span>
                                </div>
                                <div className="detail-item">
                                    <span className="detail-name">Confidence:</span>
                                    <span className="detail-value">
                                        {(analysisResult.confidence * 100).toFixed(1)}%
                                    </span>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Model Info */}
                    <div className="model-info">
                        <CheckCircle size={14} />
                        <span>Analyzed with {analysisResult.model_version} ({analysisResult.processing_time_ms}ms)</span>
                    </div>
                </div>
            )}

            <style>{`
                .voice-container {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    text-align: center;
                    max-width: 600px;
                    margin: 0 auto;
                }
                .prompt-card {
                    background: hsl(var(--bg-card));
                    padding: var(--space-lg);
                    border-radius: var(--radius-md);
                    font-size: 1.1rem;
                    font-weight: 500;
                    margin: var(--space-md) 0;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                    border-left: 4px solid hsl(var(--color-primary));
                }

                .visualizer {
                    height: 60px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 6px;
                    margin: var(--space-lg) 0;
                    opacity: 0.3;
                    transition: opacity 0.3s;
                }
                .visualizer.active { opacity: 1; }

                .bar {
                    width: 8px;
                    background: hsl(var(--color-secondary));
                    height: 20%;
                    border-radius: 4px;
                    animation: bounce 1s infinite ease-in-out;
                }
                .active .bar:nth-child(1) { animation-delay: 0.1s; height: 60%; }
                .active .bar:nth-child(2) { animation-delay: 0.2s; height: 90%; }
                .active .bar:nth-child(3) { animation-delay: 0.3s; height: 100%; }
                .active .bar:nth-child(4) { animation-delay: 0.4s; height: 80%; }
                .active .bar:nth-child(5) { animation-delay: 0.5s; height: 50%; }

                @keyframes bounce {
                    0%, 100% { transform: scaleY(0.5); }
                    50% { transform: scaleY(1.2); }
                }

                .btn-record {
                    width: 100%;
                    max-width: 300px;
                    padding: var(--space-lg);
                    border-radius: 50px;
                }
                .btn-record:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                }
                .btn-stop {
                    background: hsl(var(--status-error));
                    color: white;
                    width: 100%;
                    max-width: 300px;
                    padding: var(--space-lg);
                    border-radius: 50px;
                    border: none;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 8px;
                    font-size: 1rem;
                    font-weight: 500;
                }

                .playback-area {
                    margin-top: var(--space-lg);
                    width: 100%;
                    background: hsl(var(--bg-card));
                    padding: var(--space-md);
                    border-radius: var(--radius-md);
                }
                .audio-player { width: 100%; margin-bottom: var(--space-sm); }

                .analyzing-indicator {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    padding: var(--space-md);
                    background: #f0f9ff;
                    border-radius: var(--radius-md);
                    margin-top: var(--space-md);
                    color: #0369a1;
                    font-weight: 500;
                }

                .spinner {
                    animation: spin 1s linear infinite;
                }

                @keyframes spin {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }

                .error-message {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    padding: var(--space-md);
                    background: #fef2f2;
                    border-radius: var(--radius-md);
                    margin-top: var(--space-md);
                    color: #dc2626;
                    font-weight: 500;
                }

                .analysis-results {
                    width: 100%;
                    margin-top: var(--space-lg);
                    background: hsl(var(--bg-card));
                    padding: var(--space-lg);
                    border-radius: var(--radius-md);
                    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
                }

                .analysis-results h3 {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    margin-bottom: var(--space-md);
                    font-size: 1.2rem;
                }

                .emergency-alert {
                    background: #fef2f2;
                    border: 2px solid #dc2626;
                    border-radius: var(--radius-md);
                    padding: var(--space-md);
                    display: flex;
                    align-items: flex-start;
                    gap: 12px;
                    margin-bottom: var(--space-md);
                    color: #dc2626;
                    animation: pulse 2s ease-in-out infinite;
                }

                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.8; }
                }

                .emergency-alert strong {
                    display: block;
                    font-size: 1.1rem;
                }

                .result-card {
                    background: #f9fafb;
                    padding: var(--space-md);
                    border-radius: var(--radius-md);
                    margin-bottom: var(--space-md);
                }

                .result-label {
                    font-size: 0.9rem;
                    color: #6b7280;
                    margin-bottom: 8px;
                    text-align: left;
                }

                .score-display {
                    margin: 12px 0;
                }

                .score-value {
                    font-size: 2rem;
                    font-weight: 700;
                    margin-bottom: 8px;
                }

                .score-bar {
                    width: 100%;
                    height: 12px;
                    background: #e5e7eb;
                    border-radius: 6px;
                    overflow: hidden;
                }

                .score-fill {
                    height: 100%;
                    transition: width 0.5s ease;
                    border-radius: 6px;
                }

                .severity-badge, .risk-badge {
                    display: inline-block;
                    color: white;
                    padding: 6px 16px;
                    border-radius: 20px;
                    font-size: 0.85rem;
                    font-weight: 600;
                    margin-top: 8px;
                }

                .risk-display {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    gap: 16px;
                }

                .risk-score {
                    font-size: 1.8rem;
                    font-weight: 700;
                }

                .acoustic-details {
                    background: #f0f9ff;
                    padding: var(--space-md);
                    border-radius: var(--radius-md);
                    margin-top: var(--space-md);
                    text-align: left;
                }

                .detail-label {
                    font-weight: 600;
                    margin-bottom: 8px;
                    color: #0369a1;
                }

                .details-grid {
                    display: grid;
                    grid-template-columns: 1fr;
                    gap: 8px;
                }

                .detail-item {
                    display: flex;
                    justify-content: space-between;
                    padding: 6px 0;
                    border-bottom: 1px solid #e0f2fe;
                }

                .detail-name {
                    font-size: 0.9rem;
                    color: #374151;
                }

                .detail-value {
                    font-weight: 600;
                    color: #0369a1;
                }

                .model-info {
                    margin-top: var(--space-md);
                    padding-top: var(--space-md);
                    border-top: 1px solid #e5e7eb;
                    font-size: 0.85rem;
                    color: #6b7280;
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    justify-content: center;
                }

                @keyframes fade-in {
                    from { opacity: 0; transform: translateY(10px); }
                    to { opacity: 1; transform: translateY(0); }
                }

                .animate-fade-in {
                    animation: fade-in 0.4s ease;
                }
            `}</style>
        </div>
    );
};

export default VoiceModule;
