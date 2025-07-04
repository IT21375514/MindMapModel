# Mindmap FastAPI Service

This repository contains a FastAPI application for generating mind maps using a fine-tuned Mistral-7B model. It includes setup instructions, deployment steps, and how to manage the service with systemd and Nginx.

---

## 🚀 Features

* **FastAPI** server exposing `/generate`, `/extend`, and `/simplify` endpoints.
* **Uvicorn** as the ASGI server.
* **Systemd** unit file for daemonizing the service.
* **Nginx** as a reverse-proxy on port 80.
* Environment variable support via **python-dotenv**.

---

## 📦 Requirements

A minimal `requirements.txt`:

```
fastapi
uvicorn[standard]
transformers
torch
nest_asyncio
pydantic
protobuf==3.20.3
peft
sentencepiece
python-dotenv
requests
```

Install via:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Local Development

1. **Clone** this repo and navigate into it:

   ```bash
   git clone git@github.com:youruser/mindmap-service.git
   cd mindmap-service
   ```

2. **Create a virtual environment** and activate it:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure environment**:

   * Copy `.env.example` to `.env`:

     ```bash
     cp .env.example .env
     ```
   * Edit `.env` with your Hugging Face token:

     ```ini
     HF_TOKEN=hf_xxx
     ```

5. **Run the app**:

   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

Access the interactive docs at `http://localhost:8000/docs`.

---

## 🛠️ Production Deployment

### 1. Setup on VM

```bash
# SSH into your VM
ssh azureuser@<VM_IP> -i path/to/key.pem

# Update and install system packages
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3 python3-pip python3-venv nginx
```

### 2. Clone & Install

```bash
cd /opt
sudo git clone git@github.com:youruser/mindmap-service.git mindmap
sudo chown -R $USER:$USER mindmap
cd mindmap

# Create venv & install deps
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Environment Variables

Ensure `.env` is present in `/opt/mindmap` with your `HF_TOKEN`.

### 4. Systemd Service

Create `/etc/systemd/system/mindmap.service`:

```ini
[Unit]
Description=Mindmap FastAPI Service
After=network.target

[Service]
User=azureuser
WorkingDirectory=/opt/mindmap
EnvironmentFile=/opt/mindmap/.env
ExecStart=/opt/mindmap/venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000 --workers 4
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable & start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable mindmap
sudo systemctl start mindmap
sudo systemctl status mindmap
```

### 5. Nginx Reverse Proxy

Create `/etc/nginx/sites-available/fastapi`:

```nginx
server {
    listen 80;
    server_name <YOUR_VM_IP>;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 3600;
        proxy_send_timeout 3600;
        proxy_read_timeout 3600;
        send_timeout 3600;
    }
}
```

Enable & reload:

```bash
sudo ln -s /etc/nginx/sites-available/fastapi /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## 🔄 Updating the Service

After pushing new commits:

```bash
cd /opt/mindmap
git pull origin main
source venv/bin/activate
pip install -r requirements.txt  # if deps changed
sudo systemctl restart mindmap
```

---

## 📂 Repository Structure

```
├── app.py           # FastAPI application
├── requirements.txt # Python dependencies
├── .env.example     # Sample environment variables
├── README.md        # This file
└── ...              # other modules and utilities
```

---
