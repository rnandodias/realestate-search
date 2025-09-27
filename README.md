## Subindo Alterações
git add -A
git commit -m "Adicionando MongoDB Atlas no processo"
git push

---

# RealEstate Search — Passo 1 (Atlas)

## Observações importantes para bases grandes (~139k docs)
- **Sanitização de payload**: o ETL converte `NaN`/`+/-Inf` em `null` para evitar `InvalidJSONError` no envio ao Qdrant.
- **Batching**: tamanho padrão `BATCH_UPSERT=128` (pode ajustar via `.env`).
- **Custo/latência**: se quiser reduzir, troque `EMBEDDING_MODEL` para `text-embedding-3-small` e `VECTOR_SIZE=1536`.

## Busca — corpo de requisição (schema Rodrigo)
```json
POST /search
{
  "query_text": "terreno no Mendanha em Campo Grande RJ até 80 mil",
  "filters": {
    "city": "Rio de Janeiro",
    "neighborhoods": ["Campo Grande"],
    "unitType": "ALLOTMENT_LAND",
    "usageType": "RESIDENTIAL",
    "price": {"min": 0, "max": 80000}
  },
  "top_k": 10
}

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