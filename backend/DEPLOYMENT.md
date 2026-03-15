# 🚀 Free Deployment Guide for Frontend Team Access

## ✅ **Option 1: Render.com (Recommended - Free Tier)**

### **Step 1: Prepare Your Code**

Your deployment files are ready:
- ✅ `render.yaml` - Render configuration
- ✅ `requirements.txt` - Python dependencies
- ✅ `.gitignore` - Git ignore rules

### **Step 2: Handle Large Model File**

**Problem:** The trained model (`models/hubert_fast_best.pt`) is 362MB - too large for Git.

**Solution A: Git LFS** (Recommended)
```bash
# Install Git LFS
brew install git-lfs  # Mac
# or: sudo apt install git-lfs  # Linux

# Initialize Git LFS
git lfs install

# Track model files
git lfs track "*.pt"
git add .gitattributes
git add models/hubert_fast_best.pt
git commit -m "Add model with Git LFS"
```

**Solution B: Download on Startup**
Upload model to cloud storage (Google Drive, Dropbox, S3) and download on app startup.

**Solution C: Hugging Face Hub**
Upload model to Hugging Face and download in code.

### **Step 3: Push to GitHub**

```bash
# Initialize git (if not already done)
cd /Users/dhruv/Desktop/slurry_speech
git init
git add .
git commit -m "Initial commit - Dysarthria detection API"

# Create GitHub repo and push
# Go to github.com, create new repository
git remote add origin YOUR_GITHUB_REPO_URL
git branch -M main
git push -u origin main
```

### **Step 4: Deploy to Render**

1. **Go to:** https://render.com
2. **Sign up** with GitHub (free)
3. **New Web Service** → Connect your GitHub repo
4. **Render will auto-detect** `render.yaml`
5. **Click "Create Web Service"**
6. **Wait ~5-10 minutes** for deployment

### **Step 5: Get Your URL**

After deployment:
- URL: `https://slurry-speech-api.onrender.com`
- API Docs: `https://slurry-speech-api.onrender.com/docs`
- Health: `https://slurry-speech-api.onrender.com/healthz`

**Share this URL with frontend team!**

---

## ✅ **Option 2: Railway.app (Free $5 Credit)**

### **Quick Deploy:**

```bash
# Install Railway CLI
npm install -g railway

# Login and deploy
railway login
railway init
railway up
```

Railway gives you:
- Free $5/month credit
- Automatic deployments from GitHub
- Public URL provided

---

## ✅ **Option 3: ngrok (Quick Testing - Temporary URL)**

**Fastest way for immediate testing:**

```bash
# Install ngrok
brew install ngrok  # Mac

# Start your API locally
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Expose it publicly
ngrok http 8000
```

You'll get: `https://abc123.ngrok.io`

**Pros:**
- ✅ Immediate (30 seconds)
- ✅ No deployment needed
- ✅ Free

**Cons:**
- ❌ URL changes every restart
- ❌ API only runs when your computer is on
- ❌ For testing only

---

## 📊 **Free Tier Limitations**

### **Render.com Free Tier:**
- ✅ 750 hours/month (always on)
- ✅ 512MB RAM
- ✅ Public URL
- ⚠️ Spins down after 15 min inactivity
- ⚠️ Cold starts (~30 seconds)

### **Railway Free Tier:**
- ✅ $5/month credit
- ✅ 8GB RAM
- ✅ Always on
- ⚠️ Credit expires if unused

---

## 🔧 **Deployment Checklist**

Before deploying:

- [ ] Model file handled (Git LFS or cloud storage)
- [ ] `requirements.txt` up to date
- [ ] Environment variables set (if any)
- [ ] `.gitignore` configured
- [ ] Code pushed to GitHub
- [ ] Service deployed
- [ ] URL tested
- [ ] API docs accessible
- [ ] Frontend team notified with URL

---

## 🎯 **Quick Start (Recommended Path)**

### **For Immediate Testing (Today):**
```bash
# Use ngrok for instant public URL
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
ngrok http 8000
# Share the ngrok URL with frontend team
```

### **For Persistent Development (This Week):**
1. Push code to GitHub
2. Deploy to Render.com (free)
3. Share permanent URL with frontend team

### **For Production (Later):**
1. Move to paid tier or cloud (AWS/GCP/Azure)
2. Add authentication
3. Set up monitoring

---

## 📝 **What Frontend Team Gets**

After deployment, share:

**Base URL:**
```
https://your-api.onrender.com
```

**API Endpoint:**
```
POST https://your-api.onrender.com/v1/speech/analyse
```

**Documentation:**
```
https://your-api.onrender.com/docs
```

**Example Request:**
```javascript
const formData = new FormData();
formData.append('audio_file', audioBlob, 'recording.wav');
formData.append('patient_age', '65');

fetch('https://your-api.onrender.com/v1/speech/analyse', {
  method: 'POST',
  body: formData
})
.then(res => res.json())
.then(data => console.log(data));
```

---

## 🚨 **Important Notes**

**Model Size:** The trained model (362MB) may cause issues on free tiers:
- **If deployment fails:** Use smaller model or cloud storage
- **If too slow:** Cold starts may take 30-60 seconds
- **If out of memory:** May need to optimize or use paid tier

**Performance on Free Tier:**
- First request: ~30-60 seconds (cold start + model load)
- Subsequent requests: ~4-5 seconds
- After 15 min idle: Spins down, next request is slow again

**Recommended for:**
- ✅ Development and testing
- ✅ Frontend integration
- ✅ Demos and prototypes

**Not recommended for:**
- ❌ Production traffic
- ❌ High-volume usage
- ❌ Critical applications

---

## ✅ **Next Step**

Choose your deployment method:

1. **Quick Test (5 min):** Use ngrok
2. **Development (1 hour):** Deploy to Render
3. **Production (later):** Upgrade to paid cloud

Ready to deploy? Start with Option 1 (Render.com) for the best free persistent deployment!
