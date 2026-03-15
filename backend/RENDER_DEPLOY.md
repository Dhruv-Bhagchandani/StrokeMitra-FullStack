# 🚀 Render.com Deployment (Free, Always Online)

## ✅ **Deploy in 10 Minutes - Your API Will Run 24/7**

---

## 📋 **Prerequisites**

- ✅ GitHub account (free)
- ✅ Render.com account (free)
- ✅ Your code (already ready!)

---

## 🔧 **Step 1: Handle Large Model File**

Your trained model (`models/hubert_fast_best.pt`) is 362MB - too large for regular Git.

### **Option A: Use Git LFS** (Recommended)

```bash
cd /Users/dhruv/Desktop/slurry_speech

# Install Git LFS (if not installed)
brew install git-lfs

# Initialize Git LFS
git lfs install

# Track model files
git lfs track "*.pt"
git lfs track "models/*.pt"

# Add .gitattributes
git add .gitattributes

# Commit
git add models/hubert_fast_best.pt
git commit -m "Add model with Git LFS"
```

### **Option B: Upload to Cloud** (Alternative)

If Git LFS doesn't work, upload model to:
- Google Drive (get public link)
- Dropbox (get public link)
- Hugging Face Hub (free model hosting)

Then download on startup (see Step 4).

---

## 🐙 **Step 2: Push to GitHub**

```bash
cd /Users/dhruv/Desktop/slurry_speech

# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit - Dysarthria API with trained model"

# Create GitHub repository
# 1. Go to github.com
# 2. Click "New repository"
# 3. Name it: slurry-speech-api
# 4. Don't initialize with README (you already have files)
# 5. Click "Create repository"

# Add remote and push
git remote add origin https://github.com/YOUR_USERNAME/slurry-speech-api.git
git branch -M main
git push -u origin main
```

---

## 🌐 **Step 3: Deploy to Render**

### **A. Sign Up**
1. Go to: https://render.com
2. Click "Get Started for Free"
3. Sign up with GitHub

### **B. Create Web Service**
1. Click "New +"
2. Select "Web Service"
3. Connect your GitHub repository: `slurry-speech-api`
4. Click "Connect"

### **C. Configure Service**

**Name:** `slurry-speech-api`

**Environment:** `Python 3`

**Region:** `Oregon (US West)` (free tier)

**Branch:** `main`

**Build Command:**
```bash
pip install -r requirements.txt
```

**Start Command:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

**Plan:** `Free`

### **D. Deploy**
1. Click "Create Web Service"
2. Wait ~5-10 minutes (first deployment)
3. Watch the logs for completion

---

## ⚙️ **Step 4: If Model Download Needed**

If you used Option B (cloud storage), create this file:

```bash
# Create download_model.py
```

```python
#!/usr/bin/env python3
"""Download model if not present."""
import requests
from pathlib import Path

MODEL_PATH = Path("models/hubert_fast_best.pt")
MODEL_URL = "YOUR_GOOGLE_DRIVE_OR_DROPBOX_LINK"

if not MODEL_PATH.exists():
    print("Downloading model...")
    response = requests.get(MODEL_URL, stream=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    with open(MODEL_PATH, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("✓ Model downloaded")
```

Update `render.yaml` build command:
```yaml
buildCommand: pip install -r requirements.txt && python download_model.py
```

---

## 🎯 **Step 5: Get Your Permanent URL**

After deployment completes:

**Your permanent URL will be:**
```
https://slurry-speech-api.onrender.com
```

**API Endpoint:**
```
https://slurry-speech-api.onrender.com/v1/speech/analyse
```

**Documentation:**
```
https://slurry-speech-api.onrender.com/docs
```

---

## ✅ **Step 6: Test Your Deployment**

```bash
# Health check
curl https://slurry-speech-api.onrender.com/healthz

# Should return:
# {"status":"healthy","timestamp":"...","version":"1.0.0"}
```

---

## 📊 **Render Free Tier Details**

### **What You Get (FREE):**
✅ 750 hours/month (basically always-on)
✅ 512MB RAM
✅ Public URL (HTTPS included)
✅ Auto-deploy on git push
✅ Build time: ~10 minutes
✅ Logs and monitoring

### **Limitations:**
⚠️ Spins down after 15 min of inactivity
⚠️ Cold start: ~30 seconds first request
⚠️ 512MB RAM (might be tight for model)
⚠️ CPU-only (no GPU)

### **Performance:**
- Cold start: 30-60 seconds (model loading)
- Warm requests: 4-5 seconds
- Perfect for development and testing

---

## 🔄 **Auto-Deploy on Changes**

**After initial setup:**
```bash
# Make changes to your code
git add .
git commit -m "Update API"
git push

# Render auto-deploys!
# Watch: https://dashboard.render.com
```

---

## 🚨 **Troubleshooting**

### **Build Failed - Out of Memory**
**Solution:** Model too large for free tier
- Use Git LFS properly
- Or use cloud storage download
- Or upgrade to paid tier ($7/mo)

### **Service Won't Start**
**Check logs in Render dashboard:**
- Look for missing dependencies
- Check model file exists
- Verify Python version

### **Slow/Timeout**
- First request after idle: Expected (cold start)
- Subsequent requests: Should be ~4-5 seconds
- If always slow: May need paid tier

---

## 💡 **Tips**

**Keep Service Warm:**
Create a cron job to ping every 10 minutes:
```bash
# Use cron-job.org (free)
# URL: https://slurry-speech-api.onrender.com/healthz
# Interval: Every 10 minutes
```

**Monitor Logs:**
```
https://dashboard.render.com → Your service → Logs
```

**View Metrics:**
```
https://dashboard.render.com → Your service → Metrics
```

---

## 📝 **Summary: What Happens**

1. **You push code to GitHub** → Render detects changes
2. **Render builds your app** → Installs dependencies
3. **Render starts your API** → Runs 24/7
4. **You get permanent URL** → Share with frontend team
5. **Auto-deploys on updates** → Just push to GitHub

**Your computer can be off** - API still runs! ✅

---

## 🎉 **Next Steps**

After deployment:
1. ✅ Get your permanent URL
2. ✅ Update `FRONTEND_INTEGRATION.md` with new URL
3. ✅ Share with frontend team
4. ✅ Test from their frontend
5. ✅ Monitor in Render dashboard

**Your API will now run 24/7, independently of your computer!**
