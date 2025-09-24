# RealEstate Search — Passo 1

## Subindo Alterações
git add -A
git commit -m "Testando GitHub Actions"
git push

---

Infra mínima: Qdrant + Search Service (FastAPI) + CI/CD (GitHub Actions -> VPS via SSH).

## Pré-requisitos na VPS (apenas 1x)
```bash
sudo apt update && sudo apt install -y ca-certificates curl gnupg
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
sudo apt install -y docker-compose-plugin || true
mkdir -p /opt/realestate-search && cd /opt/realestate-search
# O .env será criado pelo deploy se não existir; edite depois.