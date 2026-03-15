import { useState, useRef, useEffect } from 'react';
import { Mic, Square, Play, BarChart2 } from 'lucide-react';

const VoiceModule = () => {
    const [isRecording, setIsRecording] = useState(false);
    const [audioUrl, setAudioUrl] = useState(null);
    const mediaRecorderRef = useRef(null);
    const chunksRef = useRef([]);

    const startRun = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorderRef.current = new MediaRecorder(stream);

            mediaRecorderRef.current.ondataavailable = (e) => {
                if (e.data.size > 0) chunksRef.current.push(e.data);
            };

            mediaRecorderRef.current.onstop = () => {
                const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
                const url = URL.createObjectURL(blob);
                setAudioUrl(url);
                chunksRef.current = [];
            };

            mediaRecorderRef.current.start();
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

    return (
        <div className="voice-container">
            <h2><Mic size={24} className="inline-icon" /> Voice Check</h2>
            <p className="description">Read the following sentence clearly:</p>

            <div className="prompt-card">
                "The quick brown fox jumps over the lazy dog."
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
                    <button className="btn btn-primary btn-record" onClick={startRun}>
                        <Mic size={20} /> {audioUrl ? 'Record Again' : 'Start Recording'}
                    </button>
                ) : (
                    <button className="btn btn-danger btn-stop" onClick={stopRun}>
                        <Square size={20} /> Stop
                    </button>
                )}
            </div>

            {audioUrl && (
                <div className="playback-area animate-fade-in">
                    <audio controls src={audioUrl} className="audio-player" />
                    <div className="analysis-stub">
                        <BarChart2 size={16} /> Analysis: Clarity 96% (Normal)
                    </div>
                </div>
            )}

            <style>{`
                .voice-container {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    text-align: center;
                }
                .prompt-card {
                    background: hsl(var(--bg-card));
                    padding: var(--space-lg);
                    border-radius: var(--radius-md);
                    font-size: 1.2rem;
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

                .btn-record { width: 100%; max-width: 300px; padding: var(--space-lg); border-radius: 50px; }
                .btn-stop { background: hsl(var(--status-error)); color: white; width: 100%; max-width: 300px; padding: var(--space-lg); border-radius: 50px; border: none; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 8px; font-size: 1rem; font-weight: 500;}

                .playback-area {
                    margin-top: var(--space-lg);
                    width: 100%;
                    background: hsl(var(--bg-card));
                    padding: var(--space-md);
                    border-radius: var(--radius-md);
                }
                .audio-player { width: 100%; margin-bottom: var(--space-sm); }
                .analysis-stub {
                    text-align: left;
                    font-size: 0.9rem;
                    color: hsl(var(--color-secondary));
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
            `}</style>
        </div>
    );
};

export default VoiceModule;
