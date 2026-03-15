import express from 'express';
import cors from 'cors';

// Simple ID generator
const generateId = () => Date.now().toString(36) + Math.random().toString(36).substr(2);

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// In-memory "database"
const sessions = {};

// Routes
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date() });
});

// Start a new screening session
app.post('/api/session', (req, res) => {
    const sessionId = generateId();
    sessions[sessionId] = {
        startTime: new Date(),
        data: {},
        completed: false
    };
    res.json({ sessionId });
});

// Submit results/data (simulated upload)
app.post('/api/data', (req, res) => {
    const { sessionId, type, payload } = req.body;
    if (!sessions[sessionId]) {
        // For simple testing, allow creation if missing (or return error)
        // return res.status(404).json({ error: 'Session not found' });
        sessions[sessionId] = { data: {} };
    }

    // Store data
    if (!sessions[sessionId].data[type]) {
        sessions[sessionId].data[type] = [];
    }
    sessions[sessionId].data[type].push({
        payload,
        timestamp: new Date()
    });

    console.log(`[${type}] Data received for session ${sessionId}`);
    res.json({ success: true });
});

// Mark session as complete
app.post('/api/complete', (req, res) => {
    const { sessionId } = req.body;
    if (!sessions[sessionId]) {
        return res.status(404).json({ error: 'Session not found' });
    }
    sessions[sessionId].completed = true;
    sessions[sessionId].endTime = new Date();
    res.json({ success: true });
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
