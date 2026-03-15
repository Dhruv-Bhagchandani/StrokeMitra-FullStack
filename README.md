# StrokeMitra - Full-Stack Stroke Detection System

AI-powered early stroke detection using voice analysis.

## 🏗️ Architecture
- **Backend** (`/backend`): Python FastAPI + HuBERT ML model
- **Frontend** (`/frontend`): React + Vite

## 🚀 Deployment
Single-service deployment to Render.com:
1. Backend serves the API at `/v1/speech/analyse`
2. Backend also serves the React app at root `/`
3. No CORS issues (same origin)

See `MONOREPO_DEPLOYMENT.md` for detailed instructions.

## 💻 Local Development

### Terminal 1 - Backend:
```bash
cd backend
uvicorn api.main:app --reload --port 8000
```

### Terminal 2 - Frontend:
```bash
cd frontend
npm run dev
```

## 📊 Features
- Real-time voice analysis for dysarthria detection
- 97.44% test AUC accuracy
- Clinical-grade risk assessment
- Instant results with explainability

