# 🚀 Monorepo Deployment Guide: Single Repository for Full-Stack App

## Overview
Deploy both frontend and backend from **ONE GitHub repository** to **ONE Render service**.

```
StrokeMitra-FullStack/
├── backend/           # FastAPI (slurry_speech)
│   ├── api/
│   ├── src/
│   ├── models/
│   ├── requirements.txt
│   └── render.yaml
├── frontend/          # React (StrokeMitra)
│   ├── src/
│   ├── public/
│   └── package.json
└── README.md
```

**Benefits:**
- ✅ Single deployment
- ✅ No CORS issues
- ✅ Simpler management
- ✅ Same origin for API calls
- ✅ One GitHub repo

---

## Step 1: Merge Projects into One Repo (5 minutes)

```bash
# Create new monorepo
cd /Users/dhruv/Desktop
mkdir StrokeMitra-FullStack
cd StrokeMitra-FullStack

# Copy backend
mkdir backend
cp -r ../slurry_speech/* backend/

# Copy frontend
mkdir frontend  
cp -r ../StrokeMitra/* frontend/

# Create root README
cat > README.md << 'MDEOF'
# StrokeMitra - Stroke Detection System

Full-stack application for early stroke detection using voice analysis.

## Structure
- `/backend` - Python FastAPI + ML models
- `/frontend` - React frontend

## Quick Start
See deployment guide in `/backend/DEPLOYMENT.md`
MDEOF

# Initialize git
git init
git add .
git commit -m "Initial commit: Full-stack stroke detection app"
```

---

## Step 2: Update Build Script (Already Done!)

The backend will:
1. Install Python dependencies
2. Build React frontend
3. Copy built files to `backend/static/`
4. Serve React app at root URL
5. API available at `/v1/speech/analyse`

---

## Step 3: Push to GitHub

```bash
# Create repo on GitHub: StrokeMitra-FullStack
# Then:
git remote add origin https://github.com/YOUR_USERNAME/StrokeMitra-FullStack.git
git branch -M main
git push -u origin main
```

---

## Step 4: Deploy to Render (10 minutes)

1. **Go to Render Dashboard**
   - https://dashboard.render.com
   - Click "New +" → "Web Service"

2. **Connect Repository**
   - Connect GitHub: `StrokeMitra-FullStack`
   - **Root Directory**: `backend`  ⚠️ Important!
   - Render will detect `render.yaml`

3. **Configuration** (Auto-detected from render.yaml):
   - **Build Command**:
     ```bash
     pip install -r requirements.txt
     cd ../frontend
     npm install
     npm run build
     mkdir -p ../backend/static
     cp -r dist/* ../backend/static/
     cd ../backend
     ```
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - **Python Version**: 3.11.8

4. **Click "Create Web Service"**
   - Build takes ~10-15 minutes (downloads ML models)
   - Watch the logs for progress

5. **Get Your URL**
   - After deployment: `https://your-app.onrender.com`
   - This serves BOTH frontend and backend!

---

## Step 5: Test Deployment

### Test Backend API:
```bash
curl https://your-app.onrender.com/healthz
```
Should return: `{"status":"healthy"}`

### Test Frontend:
- Visit: `https://your-app.onrender.com`
- Should load the React app!

### Test Full Integration:
1. Navigate to Voice Module
2. Record voice (minimum 5 seconds)
3. Click "Analyze"
4. See AI results!

---

## How It Works

### URL Structure:
```
https://your-app.onrender.com/
├── /                          → React app (index.html)
├── /voice-module              → React routing
├── /v1/speech/analyse         → API endpoint
├── /healthz                   → Health check
├── /docs                      → API documentation
└── /assets/*                  → React static files (CSS, JS)
```

### Request Flow:
1. User visits `https://your-app.onrender.com`
2. FastAPI serves `static/index.html` (React app)
3. React loads in browser
4. User records voice
5. React sends audio to `/v1/speech/analyse` (same origin!)
6. FastAPI processes audio, returns results
7. React displays results

**No CORS needed** - Same origin for everything!

---

## Local Development

### Terminal 1 - Backend:
```bash
cd /Users/dhruv/Desktop/StrokeMitra-FullStack/backend
uvicorn api.main:app --reload --port 8000
```

### Terminal 2 - Frontend:
```bash
cd /Users/dhruv/Desktop/StrokeMitra-FullStack/frontend
npm run dev
```

Frontend runs on `:5173`, talks to backend on `:8000`

---

## Updating After Changes

### Update Frontend:
```bash
cd /Users/dhruv/Desktop/StrokeMitra-FullStack
git add frontend/
git commit -m "Update frontend"
git push
```
Render auto-rebuilds in ~5 minutes

### Update Backend:
```bash
cd /Users/dhruv/Desktop/StrokeMitra-FullStack
git add backend/
git commit -m "Update API"
git push
```
Render auto-rebuilds in ~3 minutes

---

## Alternative: Using Existing Separate Repos

If you want to keep repos separate but still deploy as one:

### Option A: Git Submodules
```bash
mkdir StrokeMitra-FullStack
cd StrokeMitra-FullStack

git init
git submodule add https://github.com/YOU/slurry-speech backend
git submodule add https://github.com/YOU/StrokeMitra frontend

git add .
git commit -m "Add submodules"
git push
```

### Option B: Monorepo Build Script
Create `build.sh` in root:
```bash
#!/bin/bash
# Install backend deps
cd backend && pip install -r requirements.txt

# Build frontend
cd ../frontend
npm install
npm run build

# Copy to backend static
mkdir -p ../backend/static
cp -r dist/* ../backend/static/

echo "✅ Build complete!"
```

---

## Troubleshooting

### Build fails - npm not found:
Render uses Python buildpack by default. Add to `render.yaml`:
```yaml
buildCommand: |
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
  export NVM_DIR="$HOME/.nvm"
  [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
  nvm install 18
  pip install -r requirements.txt
  cd ../frontend && npm install && npm run build
  mkdir -p ../backend/static && cp -r dist/* ../backend/static/
  cd ../backend
```

### Frontend shows 404:
- Check `backend/static/` has files after build
- Check FastAPI logs for static mount errors
- Verify `index.html` exists in `backend/static/`

### API calls fail:
- Check browser console for errors
- Verify `SPEECH_API_URL` uses relative path in production
- Check FastAPI logs in Render dashboard

---

## Costs

**Render Free Tier:**
- 750 hours/month
- 512MB RAM (might be tight for ML models)
- Spins down after 15min inactivity
- Cold start: ~30-60 seconds

**Render Starter ($7/month):**
- 24/7 uptime
- 2GB RAM (recommended for ML)
- No cold starts
- Better for production

---

## Next Steps

1. **Custom Domain**: Settings → Custom Domain
2. **Environment Vars**: Settings → Environment
3. **Monitoring**: Dashboard shows logs, metrics
4. **Scaling**: Upgrade plan if needed

---

## Summary

✅ **What you get:**
- Single URL for everything
- No CORS configuration needed
- One deployment to manage
- Free hosting (with limits)
- Auto-deploy from GitHub

🎉 **Your app is now live at one URL!**

