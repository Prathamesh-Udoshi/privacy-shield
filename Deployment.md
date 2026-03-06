# 🚀 Deployment Guide: Privacy Shield on Render

This guide explains how to deploy the Privacy Shield project (FastAPI Backend + Next.js Frontend) to [Render](https://render.com).

## 📋 Architecture Overview

- **Backend**: Python (FastAPI) - Web Service
- **Frontend**: Node.js (Next.js) - Static Site (or Web Service)

---

## 🛠️ Step 1: Deploy the Backend (Web Service)

1.  **Create a New Web Service**: In the Render dashboard, click **New +** and select **Web Service**.
2.  **Connect Repository**: Link your GitHub repository.
3.  **Configure Settings**:
    *   **Name**: `privacy-shield-api`
    *   **Environment**: `Python 3`
    *   **Root Directory**: `.` (leave empty to use project root)
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `python -m uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4.  **Environment Variables**:
    *   Click the **Advanced** button and add:
        *   `PYTHONPATH`: `.`
        *   `OPENAI_API_KEY`: `your-key-here` (Optional, for AI Diagnostics)

> [!NOTE]
> Once deployed, Render will provide a URL like `https://privacy-shield-api.onrender.com`. Save this for the frontend configuration.

---

## 🌐 Step 2: Deploy the Frontend (Static Site or Web Service)

Since it's a Next.js app, we recommend deploying as a **Web Service** to support Server-Side Rendering (SSR) or as a **Static Site** if only using Static Site Generation (SSG).

### Option A: As a Web Service (Recommended)

1.  **Create a New Web Service**: Click **New +** and select **Web Service**.
2.  **Connect Repository**: Link the same GitHub repository.
3.  **Configure Settings**:
    *   **Name**: `privacy-shield-app`
    *   **Environment**: `Node`
    *   **Root Directory**: `frontend`
    *   **Build Command**: `npm install && npm run build`
    *   **Start Command**: `npm run start`
4.  **Environment Variables**:
    *   Add this crucial variable:
        *   `NEXT_PUBLIC_API_URL`: `https://privacy-shield-api.onrender.com` (Use your actual backend URL)

---

## 🧩 Step 3: Configure CORS (Backend)

By default, the backend only allows `localhost`. You need to update `backend/main.py` before pushing to allow your Render frontend URL.

**Edit `backend/main.py`:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://privacy-shield-app.onrender.com", # Add your frontend URL here
    ],
    # ... rest of config
)
```

---

## 💎 Using Render Blueprints (`render.yaml`)

For a more automated setup, you can create a `render.yaml` in the root directory:

```yaml
services:
  # Backend FastAPI Service
  - type: web
    name: privacy-shield-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python -m uvicorn backend.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHONPATH
        value: .
      - key: OPENAI_API_KEY
        sync: false # Set via Render dashboard

  # Frontend Next.js Service
  - type: web
    name: privacy-shield-app
    env: node
    rootDir: frontend
    buildCommand: npm install && npm run build
    startCommand: npm run start
    envVars:
      - key: NEXT_PUBLIC_API_URL
        fromService:
          name: privacy-shield-api
          type: web
          property: host
```

## 🧪 Post-Deployment Verification

1.  Visit your frontend URL.
2.  Check the "Health Audit" on the homepage.
3.  Upload a sample file (`examples/users.csv`) to verify the full anonymization pipeline.
