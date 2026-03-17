# 🚀 Fake News Detector - Running Guide

## Quick Start (Local Access)

To run locally at **http://127.0.0.1:5000**:

```bash
cd "C:/Users/sahit/OneDrive/Desktop/fakenews detector"
python app.py
```

---

## 🔗 Creating a Public URL

To access your app from anywhere using a public URL, you have these options:

### Option 1: Using ngrok (Recommended)

1. **Download ngrok**: https://ngrok.com/download
2. **Extract and run**:
   ```bash
   ngrok http 5000
   ```
3. **Copy the URL** shown (e.g., `https://abc123.ngrok.io`)

---

### Option 2: Using localtunnel (Requires Node.js)

1. **Install Node.js**: https://nodejs.org
2. **Run these commands**:
   ```bash
   cd "C:/Users/sahit/OneDrive/Desktop/fakenews detector"
   npx localtunnel --port 5000
   ```
3. **Click the URL** shown (e.g., `https://your-url.loca.lt`)
   - Note: You'll need to click "Click to Continue" on the page

---

### Option 3: Using Cloudflare Tunnel (Free)

1. **Install cloudflared**:
   ```bash
   winget install cloudflare.cloudflared
   ```
2. **Run**:
   ```bash
   cloudflared tunnel --url http://localhost:5000
   ```

---

## 📋 All-in-One Startup Script

Double-click `start_with_url.bat` in the project folder. It will:
1. Start the Flask server
2. Attempt to create a public URL using localtunnel
3. Show you the URL to use

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Module not found" | Run: `pip install flask pandas numpy scikit-learn requests beautifulsoup4` |
| "Model files not found" | Run: `python train_model.py` first |
| Port 5000 in use | Change port in app.py: `app.run(debug=True, port=5001)` |

---

## 🌐 Access Your App

**Local**: http://127.0.0.1:5000

**Public**: Use ngrok or localtunnel as shown above

---

## Features

✅ ML Detection (TF-IDF + Logistic Regression)
✅ Live News Verification  
✅ Google News Check
✅ Twitter/X Trending Check
✅ Credibility Score (0-100%)
✅ Rumor Detection Labels

---

**Model Accuracy**: 86.67%


