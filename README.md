# Token Optimizer — DeepSeek API

Analyse et optimisation des coûts de tokens pour l'API DeepSeek.

## Pourquoi

DeepSeek propose un **context caching** automatique qui réduit le coût d'entrée de **98%** pour les cache hits. Ce projet aide à maximiser ce taux de cache hit.

## Stack

- Python 3.7+ stdlib seulement
- 0 dépendances pip
- Pricing sourcé depuis la doc officielle DeepSeek (Mai 2026)

## Quick Start

```bash
git clone https://github.com/hexapost-studio/token-optimizer.git
cd token-optimizer

# Voir les prix
python3 optimizer.py pricing

# Simuler le coût d'un prompt
python3 optimizer.py simulate "Ton prompt ici" --model deepseek-v4-pro

# Comparer 2 variantes
python3 optimizer.py compare "Version courte" "Version longue détaillée"

# Voir les stratégies d'optimisation
python3 optimizer.py strategies

# Analyser des logs d'appels API (format JSONL)
python3 optimizer.py analyze api-calls.jsonl
```

## Prix DeepSeek (par 1M tokens)

| Modèle | Input (cache hit) | Input (cache miss) | Output |
|--------|-------------------|--------------------|--------|
| deepseek-v4-flash | $0.0028 | $0.14 | $0.28 |
| deepseek-v4-pro | $0.0036* | $0.435* | $0.87* |

\* Prix promotionnels (-75%), retour au prix normal après 31 Mai 2026

## Stratégies clés

1. **Ne jamais modifier le system prompt** — il est caché intégralement
2. **Static avant variable** — `"Instructions: [fixes]. Question: {variable}"` → tout le préfixe est cachable
3. **Désactiver le thinking** si pas nécessaire (les tokens de raisonnement ne sont pas cachables)
4. **Flash pour 80% des appels**, Pro pour 20% (tâches complexes)
5. **Batching** : 1 appel de 2000 tokens plutôt que 3 de 1000

## Sources

- [DeepSeek API Docs](https://api-docs.deepseek.com)
- [Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing)
- [Context Caching](https://api-docs.deepseek.com/guides/kv_cache)
- [Token Usage](https://api-docs.deepseek.com/quick_start/token_usage)
