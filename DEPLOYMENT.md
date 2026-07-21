# Deployment Guide — AI-Powered Research Agent
## IBM Cloud Lite + IBM watsonx.ai + Granite

---

## Prerequisites

| Tool | Version | Required For |
|------|---------|-------------|
| Python | 3.11+ | Backend |
| pip | Latest | Dependencies |
| Git | Any | Version control |
| IBM Cloud account | Lite tier | watsonx.ai |

---

## Step 1 — Create an IBM Cloud Account (Lite tier)

1. Go to **https://cloud.ibm.com/registration** and create a free Lite account.
2. Verify your email address.
3. Log in to the IBM Cloud console.

---

## Step 2 — Provision IBM watsonx.ai

1. In the IBM Cloud console, search for **"watsonx.ai"** in the catalog.
2. Click **watsonx.ai** → Select the **Lite** plan (free tier).
3. Choose a region (e.g., `us-south`). Note the URL — it will be  
   `https://us-south.ml.cloud.ibm.com` for US South.
4. Click **Create**.
5. Once provisioned, click **Launch watsonx.ai** to open the studio.

---

## Step 3 — Create a watsonx.ai Project

1. In the watsonx.ai studio, click **New project** → **Create an empty project**.
2. Give it a name (e.g., `research-agent`).
3. Click **Create**.
4. Open the project → Go to **Manage** tab → **General** section.
5. Copy the **Project ID** — you will need this as `WATSONX_PROJECT_ID`.

---

## Step 4 — Get Your IBM Cloud API Key

1. In IBM Cloud console, click your avatar (top-right) → **Profile and settings**.
2. Go to **IBM Cloud API keys** → **Create an IBM Cloud API key**.
3. Give it a name (e.g., `research-agent-key`) and click **Create**.
4. **Copy and save the key immediately** — it won't be shown again.
5. This becomes your `WATSONX_API_KEY`.

---

## Step 5 — Verify Granite Model Access

1. In watsonx.ai studio, open your project → **Assets** → **New asset** → **Work with models**.
2. In the Foundation Model Library, search for **"Granite"**.
3. Confirm `ibm/granite-3-3-8b-instruct` (or similar) is available on Lite tier.
4. Note the exact model ID — update `GRANITE_CONFIG["model_id"]` in `agent_config.py` if needed.

> **Lite tier note:** The Granite-3-3-8b-instruct model is available on the Lite plan.
> Token limits apply (50,000 tokens/month free). Monitor usage in your IBM Cloud account.

---

## Step 6 — Local Setup

```bash
# 1. Clone / download the project
git clone <your-repo-url>
cd research-agent

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your actual credentials:
# WATSONX_API_KEY=your_key_here
# WATSONX_PROJECT_ID=your_project_id_here
# WATSONX_URL=https://us-south.ml.cloud.ibm.com
# FLASK_SECRET_KEY=any_long_random_string

# 5. Run the application
python app.py
```

Open your browser at **http://localhost:5000**

---

## Step 7 — Verify the Connection

Navigate to `http://localhost:5000/api/health`  
You should see: `{"status": "ok", "sample": "OK"}`

If you see an error:
- Double-check `WATSONX_API_KEY` and `WATSONX_PROJECT_ID` in `.env`
- Verify the Granite model ID in `agent_config.py` matches a model available in your region
- Check that your Lite quota hasn't been exhausted

---

## Step 8 (Optional) — Deploy to IBM Code Engine

IBM Code Engine is a fully managed serverless platform — ideal for Flask apps.

### Prerequisites
- Install [IBM Cloud CLI](https://cloud.ibm.com/docs/cli): `curl -fsSL https://clis.cloud.ibm.com/install/linux | sh`
- Install Code Engine plugin: `ibmcloud plugin install code-engine`

### Deploy Steps

```bash
# 1. Log in to IBM Cloud
ibmcloud login --sso

# 2. Target your resource group
ibmcloud target -g Default

# 3. Create a Code Engine project (one-time)
ibmcloud ce project create --name research-agent-project

# 4. Select the project
ibmcloud ce project select --name research-agent-project

# 5. Create a container registry (or use Docker Hub)
# Build your container image first:
# Add a Procfile:
echo "web: python app.py" > Procfile

# Create a Dockerfile:
cat > Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
EOF

# Build and push to IBM Container Registry (or Docker Hub)
docker build -t research-agent:latest .
# (Tag and push to your registry of choice)

# 6. Create Code Engine application
ibmcloud ce application create \
  --name research-agent \
  --image <your-registry>/research-agent:latest \
  --port 5000 \
  --min-scale 0 \
  --max-scale 3 \
  --env WATSONX_API_KEY=<your_key> \
  --env WATSONX_PROJECT_ID=<your_project_id> \
  --env WATSONX_URL=https://us-south.ml.cloud.ibm.com \
  --env FLASK_SECRET_KEY=<random_string>

# 7. Get the public URL
ibmcloud ce application get --name research-agent --output url
```

Your app will be accessible at the printed URL.

---

## Step 9 (Optional) — Deploy to Cloud Foundry

```bash
# 1. Install CF CLI plugin
ibmcloud cf install

# 2. Push the app (uses Procfile or manifest.yml)
cat > manifest.yml << 'EOF'
applications:
  - name: research-agent
    memory: 1G
    instances: 1
    buildpack: python_buildpack
    env:
      WATSONX_API_KEY: <your_key>
      WATSONX_PROJECT_ID: <your_project_id>
      WATSONX_URL: https://us-south.ml.cloud.ibm.com
      FLASK_SECRET_KEY: <random_string>
EOF

ibmcloud cf push
```

---

## Environment Variables Reference

| Variable | Description | Required |
|----------|-------------|----------|
| `WATSONX_API_KEY` | IBM Cloud API key with watsonx.ai access | ✅ Yes |
| `WATSONX_PROJECT_ID` | Your watsonx.ai project UUID | ✅ Yes |
| `WATSONX_URL` | Regional watsonx.ai endpoint URL | ✅ Yes |
| `FLASK_SECRET_KEY` | Secret key for Flask sessions | ✅ Yes |
| `FLASK_DEBUG` | Set to `true` for dev mode | ❌ No |
| `PORT` | Port to listen on (default: 5000) | ❌ No |
| `DATABASE_URL` | SQLAlchemy DB URL (default: sqlite) | ❌ No |

---

## Customizing the Agent

Edit `agent_config.py` to change the persona, tone, citation style, and safety rules:

```python
AGENT_INSTRUCTIONS = {
    "role": "You are a specialized biomedical research assistant...",
    "tone": "precise, clinical",
    "domain_specialization": ["biology", "genomics"],
    "citation_style": "APA",
    ...
}
```

No other code changes are needed — the new instructions flow into every Granite prompt automatically.

---

## Swapping the Vector Store

Edit `VECTOR_STORE_CONFIG["backend"]` in `agent_config.py`:

```python
VECTOR_STORE_CONFIG = {
    "backend": "faiss",  # Change from "chroma" to "faiss"
    ...
}
```

Make sure `faiss-cpu` is installed: `pip install faiss-cpu`  
No changes to routes or `app.py` are required.

---

## Troubleshooting

| Symptom | Solution |
|---------|---------|
| `EnvironmentError: WATSONX_API_KEY is not set` | Copy `.env.example` → `.env` and fill credentials |
| `RuntimeError: Granite generation error: 403` | Check API key permissions and Lite quota |
| `chromadb` import error | Run `pip install chromadb` |
| `sentence-transformers` slow on first load | Model downloads once; subsequent loads use cache |
| Port 5000 already in use | Set `PORT=5001` in `.env` |

---

*Built for the IBM watsonx.ai Challenge — uses IBM Granite exclusively for all NLG tasks.*
