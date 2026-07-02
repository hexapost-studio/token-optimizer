# Token Optimizer — DeepSeek API + Hermes Agent

Analyse et optimisation des coûts de tokens pour l'API DeepSeek, intégré avec
Hermes Agent pour un routage intelligent des modèles.

## Architecture du système

```
┌─────────────────────────────────────────────────────────┐
│                    HERMES AGENT                          │
│                                                          │
│  ┌──────────────┐              ┌──────────────┐         │
│  │ default      │              │ eco          │         │
│  │ Pro+thinking │  80% du      │ Flash        │         │
│  │ medium       │  quotidien   │ no thinking  │         │
│  └──────┬───────┘              └──────┬───────┘         │
│         │                             │                 │
│         │  20% tâches                 │  80% tâches     │
│         │  complexes                  │  courantes      │
│         ▼                             ▼                 │
│  ┌──────────────────────────────────────────────┐       │
│  │  Tâches auxiliaires → deepseek-v4-flash     │       │
│  │  (vision, compression, search, curator...)  │       │
│  │  Subagents → deepseek-v4-flash              │       │
│  └──────────────────────────────────────────────┘       │
│                                                          │
│  ┌──────────────────────────────────────────────┐       │
│  │  optimizer.py                                 │       │
│  │  ├─ count    → comptage exact (tokenizer V3) │       │
│  │  ├─ simulate → coût estimé d'un prompt       │       │
│  │  ├─ compare  → comparer 2 variantes          │       │
│  │  ├─ analyze  → analyser logs d'appels        │       │
│  │  └─ strategies → guide complet               │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

## Prix DeepSeek V4 (par 1M tokens)

| Modèle | Input cache hit | Input cache miss | Output |
|--------|-----------------|------------------|--------|
| deepseek-v4-flash | $0.0028 | $0.14 | $0.28 |
| deepseek-v4-pro (promo) | $0.0036 | $0.435 | $0.87 |
| deepseek-v4-pro (normal) | $0.0145 | $1.74 | $3.48 |

- Promo -75% sur Pro jusqu'au 31/05/2026
- Output coûte 4x l'input → priorité à la réduction de l'output
- Cache hit = 50-120x moins cher que cache miss
- Thinking tokens = non cachables, même prix que output
- reasoning_effort "low"/"medium"/"high" → tous mappés à "high" (même coût)
- reasoning_effort "none"/"minimal" → thinking désactivé
- reasoning_effort "xhigh"/"max" → reasoning maximal

## Quick Start

```bash
cd ~/Documents/token-optimizer

# Voir les prix
python3 optimizer.py pricing

# Compter les tokens EXACTS (tokenizer officiel DeepSeek V3)
python3 optimizer.py count "Ton texte ici"
python3 optimizer.py count ~/.hermes/SOUL.md

# Simuler le coût d'un prompt
python3 optimizer.py simulate "Explique-moi le théorème de Pythagore" --model deepseek-v4-flash

# Comparer 2 variantes
python3 optimizer.py compare "Version courte" "Version longue détaillée"

# Guide complet des stratégies
python3 optimizer.py strategies

# Analyser des logs d'appels API (JSONL)
python3 optimizer.py analyze api-calls.jsonl
```

## Tokenizer officiel DeepSeek V3

Le tokenizer est dans `~/Downloads/deepseek_v3_tokenizer/` (7.5 MB).

Il permet un comptage **exact** des tokens (vs ±30% d'erreur avec l'estimation
char-based). Le script le détecte automatiquement :

```bash
# Test
python3 optimizer.py count "Bonjour le monde"
# → Tokens (EXACT) : 4 (ratio 3.6 char/tok en français)

# Benchmark: écart entre estimation et réalité
python3 optimizer.py count "Un paragraphe plus long pour voir l'erreur..."
# → Tokens (EXACT) : 28
# → Tokens (estimé) : 30 (erreur: 7.1%)
```

## Profils Hermes

### Configuration

```bash
# Créer le profil eco
hermes profile create eco --clone

# Configurer eco : Flash + no thinking
hermes -p eco config set model.default deepseek-v4-flash
hermes -p eco config set agent.reasoning_effort none

# Configurer les tâches auxiliaires (sur les DEUX profils)
for task in vision web_extract compression session_search skills_hub \
            approval title_generation curator flush_memories mcp; do
  hermes config set auxiliary.$task.provider deepseek
  hermes config set auxiliary.$task.model deepseek-v4-flash
done

# Subagents
hermes config set delegation.provider deepseek
hermes config set delegation.model deepseek-v4-flash
```

### Utilisation quotidienne

```bash
hermes                     # Pro + thinking (debug, raisonnement complexe)
hermes -p eco              # Flash sans thinking (80% du quotidien, -68% de coût)
hermes -p eco -m deepseek-v4-pro  # Flash → Pro ponctuellement

# Alias recommandé (dans ~/.zshrc)
alias hermes-eco='hermes -p eco'
```

### Résultat

| Commande | Modèle | Thinking | Usage |
|----------|--------|----------|-------|
| `hermes` | Pro | ON (high) | Tâches complexes, debug, raisonnement |
| `hermes -p eco` | Flash | OFF | 80% du quotidien |
| Auxiliaires | Flash | OFF | Vision, compression, search... |
| Subagents | Flash | OFF | Toutes les délégations |

## Stratégies d'optimisation (classées par impact)

### Tier 1 — Appliqué (HIGH impact, LOW effort)

| # | Stratégie | Économie estimée | Statut |
|---|-----------|-----------------|--------|
| 1 | Flash par défaut (eco profile) | 60-70% total | ✅ |
| 2 | Auxiliaires sur Flash | 60-80% sur 30-40% du volume | ✅ |
| 3 | Thinking off par défaut | 30-80% output | ✅ (eco) |
| 4 | Compression automatique | 50-90% sessions longues | ✅ (déjà actif) |
| 5 | Tokenizer exact | Qualité des décisions | ✅ |

### Tier 2 — Non appliqué (MEDIUM effort)

| # | Stratégie | Économie estimée | Pourquoi pas |
|---|-----------|-----------------|-------------|
| 6 | Logger cache hit tokens | Données réelles | Script à écrire |
| 7 | Structured output partout | 30-50% output | Partiellement fait (tool calling) |
| 8 | Few-shot trimming | 40-70% exemples | Audit manuel du SOUL.md |

### Tier 3 — Non prioritaire (HIGH effort)

| # | Stratégie | Économie estimée | Pourquoi pas |
|---|-----------|-----------------|-------------|
| 9 | LLMLingua-2 compression | 50-80% prompts longs | Code à écrire |
| 10 | Cache sémantique local | 20-40% patterns | Single-user → hit rate bas |
| 11 | Batch API (50% réduction) | 50% tâches batch | 24h turnaround |

## Estimation d'économies

```
Session type (10 tours) :
  Avant  (100% Pro)           : $2.80
  Après  (80% eco + aux Flash): $0.90
  Économie                    : 68%

Après fin promo (1er juin 2026) :
  Avant  (100% Pro, normal)   : $11.20
  Après  (80% eco + aux Flash): $1.70
  Économie                    : 85%
```

## Analyse critique (pièges du cache)

1. **Pro avec cache hit PEUT coûter plus cher que Flash sans cache**
   - Output de Pro ($0.87/M) > Flash complet ($0.14/M + $0.28/M)
   - Si ta réponse est longue, Flash sans cache bat Pro avec cache

2. **Changer 1 mot au milieu peut tuer tout le cache**
   - Cache = all-or-nothing sur les cache units
   - Pas de partial credit

3. **Disk cache ≠ instantané**
   - Construction = secondes
   - Appels trop rapides → cache pas prêt

4. **Thinking = anti-cache**
   - Tokens de raisonnement jamais cachés
   - 10K thinking + 2K visible = thinking = 83% du coût output

5. **Best-effort : pas de garantie**
   - Hit rate probablement 40-60% (pas 80%)
   - Dépend de la charge serveur DeepSeek

## Rigueur scientifique

Pour passer de 4/10 à 8/10 :

- [x] Sources documentées (doc officielle DeepSeek)
- [x] Hypothèses explicites
- [x] Tokenizer officiel pour comptage exact
- [ ] Logger prompt_cache_hit_tokens sur 100+ appels réels
- [ ] Calculer le hit rate moyen ± écart-type
- [ ] Tester A/B : même prompt, Pro vs Flash, mesurer qualité vs coût
- [ ] Publier les résultats (gist GitHub)

## Sources

- [DeepSeek API Docs](https://api-docs.deepseek.com)
- [Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing)
- [Context Caching (KV Cache)](https://api-docs.deepseek.com/guides/kv_cache)
- [Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode)
- [Token Usage](https://api-docs.deepseek.com/quick_start/token_usage)
- [Hermes Agent Docs](https://hermes-agent.nousresearch.com/docs/)
- [LLMLingua-2 (ACL 2024)](https://arxiv.org/abs/2403.12966)
- [RouteLLM (Anyscale 2024)](https://arxiv.org/abs/2406.18665)
- [MemGPT (2024)](https://arxiv.org/abs/2310.08560)

## Fichiers du projet

```
~/Documents/token-optimizer/
├── optimizer.py          # Outil d'analyse et simulation
├── README.md             # Cette documentation

~/Downloads/deepseek_v3_tokenizer/
├── tokenizer.json        # Tokenizer officiel DeepSeek V3 (7.5 MB)
├── tokenizer_config.json # Configuration
└── deepseek_tokenizer.py # Script de test

~/.hermes/
├── config.yaml           # Profil default (Pro + thinking)
└── profiles/eco/
    └── config.yaml       # Profil eco (Flash + no thinking)

~/llm_cost_reduction_research.md  # Recherche complète (10 stratégies, refs)
```
