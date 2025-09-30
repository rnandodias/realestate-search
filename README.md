# Subindo Alterações
```bash
git add -A
git commit -m "Passo 1 do processo no n8n"
git push

```

# RealEstate Search — Passo 1 (Atlas)

## Observações importantes para bases grandes (~139k docs)
- **Sanitização de payload**: o ETL converte `NaN`/`+/-Inf` em `null` para evitar `InvalidJSONError` no envio ao Qdrant.
- **Batching**: tamanho padrão `BATCH_UPSERT=128` (pode ajustar via `.env`).
- **Custo/latência**: se quiser reduzir, troque `EMBEDDING_MODEL` para `text-embedding-3-small` e `VECTOR_SIZE=1536`.

# ETL — Como rodar

```bash
cd /opt/realestate-search

# (opcional) limpar a coleção no Qdrant
export RESET_QDRANT=true

# pending = só quem não tem embedding
export MODE=pending

# amostra (opcional)
export SAMPLE_SIZE=5000

# lotes
export BATCH_EMBED=32
export BATCH_UPSERT=128

docker run --rm -it \
  -v /opt/realestate-search:/work -w /work \
  --network host python:3.11 \
  bash -lc "pip install -r etl/requirements.txt && python etl/etl_embeddings.py"

# Outros modos
# MODE=updated   -> reindexa docs com updatedAt > embedded_at
# MODE=all       -> passa em todos (reconstrução)
# MODE=ids_file  -> reprocessa apenas os _ids do arquivo IDS_FILE
```

# Busca — corpo de requisição (schema Rodrigo)

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
```



curl -s -X POST http://127.0.0.1:8080/search \
    -H 'Content-Type: application/json' \
    -d '{"query_text":"Quero apê 2 quartos na Barra da Tijuca próximo do metrô","filters":{"city":"Rio de Janeiro","neighborhoods": ["Barra da Tijuca"],"propertyType": ["Apartamento"]},"top_k":5}' | jq

{"query_text": "Apartamento de três quartos na Avenida General San Martín, no Leblon.","filters": {"city": "rio de janeiro","neighborhoods": ["Leblon"],"propertyType": ["Apartamento"],"bedrooms": {"min": 3,"max": 3}},"top_k": 5}

search_example = {
  "query_text": "quero um apartamento residencial com 3 quartos e 1 suíte na Barra Olímpica, até 700 mil, com elevador e piscina",
  "filters": {
    "city": "rio de janeiro",
    "neighborhoods": ["Barra Olímpica"],
    "propertyType": ["Unidade"],
    "unitType": ["Apartamento"],
    "usageType": ["Residencial"],
    "price": { "min": None, "max": 700000 },
    "usableArea": { "min": 70, "max": 100 },
    "totalArea": { "min": None, "max": None },
    "bedrooms": { "min": 3, "max": 3 },
    "bathrooms": { "min": 2, "max": None },
    "suites": { "min": 1, "max": None },
    "parkingSpaces": { "min": 1, "max": None },
    "amenities": ["elevador", "piscina adulto"],
    "status": ["active"],
    "sellerTier": ["gold", "diamond"]
  },
  "top_k": 10
}

Infra mínima: Qdrant + Search Service (FastAPI) + CI/CD (GitHub Actions -> VPS via SSH).

# Utils

```bash
# Verificar quantos pontos temos na coleção do Qdrant
curl -s http://127.0.0.1:6333/collections/imoveis_v1 | jq
```

```bash
# Apagando a coleção do Qdrant
curl -X DELETE "http://127.0.0.1:6333/collections/imoveis_v1"
```

## Pré-requisitos na VPS (apenas 1x)

```bash
sudo apt update && sudo apt install -y ca-certificates curl gnupg
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
sudo apt install -y docker-compose-plugin || true
mkdir -p /opt/realestate-search && cd /opt/realestate-search
# O .env será criado pelo deploy se não existir; edite depois.
```