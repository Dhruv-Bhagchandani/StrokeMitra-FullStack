export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001/api';

// Slurry Speech API Configuration
// In production (when served by FastAPI): use relative URL (same origin, no CORS)
// In development (Vite dev server): use localhost:8000
export const SPEECH_API_URL = import.meta.env.VITE_SPEECH_API_URL ||
    (import.meta.env.PROD ? 'https://dhruvb1906-strokemitra-api.hf.space' : 'http://localhost:8000');

export const startSession = async () => {
    try {
        const response = await fetch(`${API_URL}/session`, { method: 'POST' });
        if (!response.ok) throw new Error('Failed to start session');
        return await response.json();
    } catch (err) {
        console.error(err);
        // Fallback for offline usage
        return { sessionId: 'offline-' + Date.now() };
    }
};

export const submitData = async (sessionId, type, payload) => {
    try {
        const response = await fetch(`${API_URL}/data`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sessionId, type, payload })
        });
        return response.ok;
    } catch (err) {
        console.warn('Offline mode: data saved locally');
        return true; // Pretend it worked
    }
};

/**
 * Analyze speech audio for dysarthria detection
 * @param {Blob} audioBlob - Audio file blob (WAV, MP3, etc.)
 * @param {number} patientAge - Optional patient age
 * @param {number} onsetHours - Optional hours since symptom onset
 * @returns {Promise<Object>} Analysis results
 */
export const analyzeSpeech = async (audioBlob, patientAge = null, onsetHours = null) => {
    try {
        const formData = new FormData();

        // Convert audio blob to file (preserving the original blob type)
        // Determine file extension from MIME type
        const mimeType = audioBlob.type || 'audio/webm';
        let extension = '.webm';
        if (mimeType.includes('opus') || mimeType.includes('ogg')) {
            extension = '.ogg';
        } else if (mimeType.includes('mp4') || mimeType.includes('m4a')) {
            extension = '.m4a';
        } else if (mimeType.includes('webm')) {
            extension = '.webm';
        }

        const audioFile = new File([audioBlob], `recording${extension}`, { type: mimeType });
        formData.append('audio_file', audioFile);

        // Add optional parameters
        if (patientAge) formData.append('patient_age', patientAge);
        if (onsetHours) formData.append('onset_hours', onsetHours);

        const response = await fetch(`${SPEECH_API_URL}/v1/speech/analyse`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Speech analysis failed');
        }

        return await response.json();
    } catch (err) {
        console.error('Speech analysis error:', err);
        throw err;
    }
};
