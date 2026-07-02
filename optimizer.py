#!/usr/bin/env python3
"""
Token Optimizer — DeepSeek API cost analysis & optimization.
Analyse les appels API, mesure le cache hit rate, suggère des optimisations.

Usage:
  python3 optimizer.py analyze <logfile>     # Analyse un fichier de logs API
  python3 optimizer.py simulate <prompt>     # Simule le coût d'un prompt
  python3 optimizer.py compare <a> <b>      # Compare 2 variantes de prompt
  python3 optimizer.py count <text|file>     # Comptage exact de tokens
  python3 optimizer.py strategies            # Affiche les stratégies

Pricing source: https://api-docs.deepseek.com/quick_start/pricing
Cache docs: https://api-docs.deepseek.com/guides/kv_cache
"""

import json
import sys
import time
import os
import re
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════════════════
# OFFICIAL DEEPSEEK TOKENIZER (exact token counting)
# Download: pip install transformers
# Tokenizer files: ~/Downloads/deepseek_v3_tokenizer/
# ══════════════════════════════════════════════════════════════════════════════

_TOKENIZER = None
_TOKENIZER_DIR = os.path.expanduser("~/Downloads/deepseek_v3_tokenizer")

def _load_tokenizer():
    """Lazy-load the official DeepSeek V3 tokenizer."""
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    try:
        import transformers
        if os.path.isdir(_TOKENIZER_DIR):
            _TOKENIZER = transformers.AutoTokenizer.from_pretrained(
                _TOKENIZER_DIR, trust_remote_code=True
            )
        else:
            _TOKENIZER = False  # Sentinel: tried, not found
    except Exception:
        _TOKENIZER = False
    return _TOKENIZER

def count_tokens(text: str) -> int:
    """Exact token count using the official DeepSeek tokenizer.
    Falls back to character-based estimation if tokenizer unavailable."""
    tok = _load_tokenizer()
    if tok:
        return len(tok.encode(text))
    return estimate_tokens(text)

# ══════════════════════════════════════════════════════════════════════════════
# PRICING — DeepSeek API (per 1M tokens)
# Source: api-docs.deepseek.com/quick_start/pricing — May 2026
# ══════════════════════════════════════════════════════════════════════════════

PRICING = {
    "deepseek-v4-flash": {
        "input_cache_hit":  0.0028,    # Cache hit  — 98% cheaper
        "input_cache_miss": 0.14,      # Cache miss — full price
        "output":           0.28,
        "context":          1_000_000,  # 1M tokens
        "max_output":       384_000,
    },
    "deepseek-v4-pro": {
        "input_cache_hit":  0.003625,   # Promo -75% (normal: $0.0145)
        "input_cache_miss": 0.435,      # Promo -75% (normal: $1.74)
        "output":           0.87,       # Promo -75% (normal: $3.48)
        "context":          1_000_000,
        "max_output":       384_000,
    },
}

# Prix NORMaux (post-promo, après le 31/05/2026)
PRICING_NORMAL = {
    "deepseek-v4-flash": {
        "input_cache_hit":  0.0028,    # Inchangé
        "input_cache_miss": 0.14,      # Inchangé
        "output":           0.28,
    },
    "deepseek-v4-pro": {
        "input_cache_hit":  0.0145,
        "input_cache_miss": 1.74,
        "output":           3.48,
    },
}

# Note: DeepSeek reasoning_effort mapping
# "none"/"minimal" → thinking OFF (use this for Flash)
# "low"/"medium"/"high" → all map to "high" (same cost)
# "xhigh"/"max" → "max" (most expensive)
# Thinking tokens: NOT cacheable, cost same as output tokens

# Cache prefix unit size (estimated — DeepSeek doesn't publish exact value)
# Based on Sliding Window Attention, cache units are ~4K-8K tokens each
ESTIMATED_CACHE_UNIT_TOKENS = 4096

# ══════════════════════════════════════════════════════════════════════════════
# TOKEN ESTIMATION (simple character-based)
# 1 English char ≈ 0.3 token | 1 Chinese char ≈ 0.6 token
# ══════════════════════════════════════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    """Estimation rapide du nombre de tokens."""
    if not text:
        return 0
    # Détection mixte : compter les caractères non-ASCII (chinois, accents...)
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    non_ascii = len(text) - ascii_chars
    return int(ascii_chars * 0.3 + non_ascii * 0.6)


def format_cost(cost: float) -> str:
    """Formatte un coût en dollars."""
    if cost < 0.0001:
        return f"${cost:.6f}"
    elif cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.4f}"


def format_tokens(n: int) -> str:
    """Formatte un nombre de tokens."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)


# ══════════════════════════════════════════════════════════════════════════════
# CACHE HIT SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def simulate_cache_hits(messages: list, history: list = None) -> dict:
    """
    Simule les cache hits pour une séquence de messages.
    
    Règles (basées sur la doc DeepSeek):
    1. Cache unit = fin du user input OU fin du model output
    2. Hit = le prefixe du message courant match EXACTEMENT un cache unit
    3. Common prefix detection → le système détecte les prefixes communs
    """
    if history is None:
        history = []
    
    total_input = 0
    cache_hits = 0
    detail = []
    
    # Simuler cache units existants depuis l'historique
    existing_units = set()
    for i, msg in enumerate(history):
        if msg.get("role") in ("user", "system"):
            prefix = json.dumps(msg, sort_keys=True)[:ESTIMATED_CACHE_UNIT_TOKENS * 3]
            existing_units.add(f"unit_{hash(prefix) % 100000}")
    
    for i, msg in enumerate(messages):
        tokens = estimate_tokens(json.dumps(msg))
        total_input += tokens
        
        # Check if this message might hit a cache unit
        prefix = json.dumps(msg, sort_keys=True)[:100]
        unit_id = f"unit_{hash(prefix) % 100000}"
        
        if unit_id in existing_units or (history and msg in history):
            cache_hits += tokens
            detail.append({"msg": i, "tokens": tokens, "hit": True})
        else:
            existing_units.add(unit_id)
            detail.append({"msg": i, "tokens": tokens, "hit": False})
    
    cache_miss = total_input - cache_hits
    return {
        "total_input": total_input,
        "cache_hits": cache_hits,
        "cache_miss": cache_miss,
        "hit_rate": cache_hits / total_input if total_input > 0 else 0,
        "detail": detail,
    }


# ══════════════════════════════════════════════════════════════════════════════
# COMMANDS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_simulate(model: str, prompt: str, system: str = ""):
    """Simule le coût d'un prompt unique."""
    pricing = PRICING.get(model, PRICING["deepseek-v4-flash"])
    
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    input_tokens = sum(count_tokens(json.dumps(m)) for m in messages)
    # Estimer output (typiquement 2-4x l'input pour une réponse normale)
    output_tokens = min(input_tokens * 2, pricing["max_output"])
    
    # Simulation cache (sans historique → tout cache miss au 1er appel)
    cache_result = simulate_cache_hits(messages)
    
    print(f"\n{'='*60}")
    print(f"  SIMULATION — {model}")
    print(f"{'='*60}\n")
    print(f"  📝 Input tokens  : {format_tokens(input_tokens)}")
    print(f"  💬 Output tokens : {format_tokens(output_tokens)} (estimé)")
    print(f"  📊 Total tokens  : {format_tokens(input_tokens + output_tokens)}\n")
    
    # Premier appel (tout cache miss)
    cost_first = (input_tokens * pricing["input_cache_miss"] / 1_000_000 +
                  output_tokens * pricing["output"] / 1_000_000)
    
    # Appels suivants (cache hits sur le system prompt)
    if system:
        system_tokens = count_tokens(json.dumps({"role": "system", "content": system}))
        input_hit = system_tokens
        input_miss = input_tokens - system_tokens
    else:
        input_hit = input_tokens * 0.7  # ~70% de cache hit (prefix commun)
        input_miss = input_tokens - input_hit
    
    cost_after = (input_hit * pricing["input_cache_hit"] / 1_000_000 +
                  input_miss * pricing["input_cache_miss"] / 1_000_000 +
                  output_tokens * pricing["output"] / 1_000_000)
    
    print(f"  💰 Coût 1er appel  (cache miss) : {format_cost(cost_first)}")
    print(f"  💰 Coût appel N+1 (cache hit)  : {format_cost(cost_after)}")
    print(f"  📉 Économie par appel          : {format_cost(cost_first - cost_after)} ({int((1 - cost_after/cost_first)*100)}%)\n")
    
    # Coût sur 100 appels
    print(f"  Coût sur 100 appels (1er + 99 avec cache) :")
    total_100 = cost_first + cost_after * 99
    print(f"    → {format_cost(total_100)}\n")


def cmd_compare(model: str, prompt_a: str, prompt_b: str):
    """Compare 2 variantes de prompt."""
    pricing = PRICING.get(model, PRICING["deepseek-v4-flash"])
    
    tokens_a = count_tokens(prompt_a)
    tokens_b = count_tokens(prompt_b)
    
    diff = tokens_b - tokens_a
    pct = (diff / tokens_a * 100) if tokens_a > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"  COMPARAISON — {model}")
    print(f"{'='*60}\n")
    print(f"  Variante A : {format_tokens(tokens_a)} tokens")
    print(f"  Variante B : {format_tokens(tokens_b)} tokens")
    print(f"  Différence : {'+' if diff > 0 else ''}{format_tokens(diff)} ({pct:+.1f}%)\n")
    
    cost_a = tokens_a * pricing["input_cache_miss"] / 1_000_000
    cost_b = tokens_b * pricing["input_cache_miss"] / 1_000_000
    cost_diff = cost_b - cost_a
    
    print(f"  💰 Coût A (cache miss) : {format_cost(cost_a)}")
    print(f"  💰 Coût B (cache miss) : {format_cost(cost_b)}")
    print(f"  💰 Différence          : {format_cost(cost_diff)}\n")
    
    if tokens_a > 0:
        ratio = tokens_b / tokens_a
        chars_diff = len(prompt_b) - len(prompt_a)
        print(f"  📊 Ratio              : {ratio:.1f}x")
        print(f"  📝 Différence chars   : {chars_diff:+d} caractères\n")


def cmd_analyze(logfile: str):
    """Analyse un fichier de logs d'appels API."""
    if not os.path.exists(logfile):
        print(f"❌ Fichier introuvable: {logfile}")
        print("   Format attendu: JSONL avec 'model', 'input_tokens', 'output_tokens', 'cache_hit_tokens'")
        return
    
    total_input = 0
    total_output = 0
    total_cache_hit = 0
    calls = 0
    models = defaultdict(int)
    models_cost = defaultdict(float)
    
    with open(logfile) as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            model = data.get("model", "unknown")
            inp = data.get("input_tokens", data.get("prompt_tokens", 0))
            out = data.get("output_tokens", data.get("completion_tokens", 0))
            cache = data.get("cache_hit_tokens", data.get("prompt_cache_hit_tokens", 0))
            
            total_input += inp
            total_output += out
            total_cache_hit += cache
            calls += 1
            models[model] += 1
    
    cache_miss = total_input - total_cache_hit
    hit_rate = total_cache_hit / total_input * 100 if total_input > 0 else 0
    
    # Calculer le coût pour deepseek-v4-flash
    pricing = PRICING["deepseek-v4-flash"]
    cost = (total_cache_hit * pricing["input_cache_hit"] / 1_000_000 +
            cache_miss * pricing["input_cache_miss"] / 1_000_000 +
            total_output * pricing["output"] / 1_000_000)
    
    print(f"\n{'='*60}")
    print(f"  ANALYSE — {logfile}")
    print(f"{'='*60}\n")
    print(f"  📞 Appels      : {calls}")
    print(f"  📝 Input       : {format_tokens(total_input)}")
    print(f"  💬 Output      : {format_tokens(total_output)}")
    print(f"  🎯 Cache hits  : {format_tokens(total_cache_hit)} ({hit_rate:.1f}%)")
    print(f"  ❌ Cache miss  : {format_tokens(cache_miss)} ({100-hit_rate:.1f}%)\n")
    print(f"  💰 Coût total  : {format_cost(cost)}\n")
    
    if models:
        print(f"  Modèles utilisés :")
        for m, c in sorted(models.items(), key=lambda x: x[1], reverse=True):
            print(f"    {m}: {c} appels\n")


def cmd_count(target: str):
    """Comptage exact de tokens via le tokenizer officiel DeepSeek V3."""
    if os.path.isfile(target):
        with open(target) as f:
            text = f.read()
        source = f"fichier: {target}"
    else:
        text = target
        source = "texte direct"
    
    tok = _load_tokenizer()
    exact = tok and True or False
    
    tokens_exact = count_tokens(text) if tok else None
    tokens_est = estimate_tokens(text)
    chars = len(text)
    
    print(f"\n{'='*60}")
    print(f"  COMPTAGE DE TOKENS — {source}")
    print(f"{'='*60}\n")
    print(f"  📝 Caractères     : {chars:,}")
    
    if exact:
        print(f"  🎯 Tokens (EXACT)  : {tokens_exact:,} (tokenizer officiel DeepSeek V3)")
        error = abs(tokens_exact - tokens_est) / tokens_exact * 100
        print(f"  📊 Tokens (estimé) : {tokens_est:,} (erreur: {error:.1f}%)")
        print(f"  📐 Ratio char/tok : {chars/tokens_exact:.1f}")
    else:
        print(f"  📊 Tokens (estimé) : {tokens_est:,} (char-based, ±30%)")
        print(f"  ⚠️  Tokenizer officiel non trouvé dans {_TOKENIZER_DIR}")
        print(f"     Pour un comptage exact: pip install transformers")
    
    print()
    for model_name, p in PRICING.items():
        tok_count = tokens_exact if exact else tokens_est
        cost = tok_count * p["input_cache_miss"] / 1_000_000
        print(f"  💰 {model_name:<22s} : {format_cost(cost)} (cache miss)")
    print()


def cmd_strategies():
    """Affiche les stratégies d'optimisation."""
    print("""
{'='*60}
  STRATÉGIES D'OPTIMISATION — DeepSeek API
{'='*60}

1. MAXIMISER LES CACHE HITS (98% d'économie)
   ├─ Ne JAMAIS modifier le system prompt entre les appels
   ├─ Garder la même structure de messages (ordre, rôles)
   ├─ Utiliser des conversations multi-tours (A+B → A+B+C)
   └─ Éviter de changer le début des messages

2. MINIMISER LES TOKENS DE RAISONNEMENT
   ├─ reasoning_effort="none" → thinking OFF (gratuit, utiliser pour Flash)
   ├─ reasoning_effort="low"/"medium"/"high" → tous mappés à "high" (même coût)
   ├─ reasoning_effort="xhigh"/"max" → max reasoning (le plus cher)
   ├─ Les tokens de raisonnement ne sont PAS cachables
   └─ Coût thinking : ~2-5x plus de tokens output (invisibles!)

3. ROUTAGE INTELLIGENT (Hermes profiles)
   ├─ Profile eco : Flash + no thinking (80% du quotidien)
   ├─ Profile default : Pro + thinking (tâches complexes)
   ├─ Tâches auxiliaires (vision, compression...) → Flash (configuré)
   └─ Subagents → Flash (configuré)

4. STRUCTURER LES PROMPTS
   ├─ Placer le contenu statique AU DÉBUT (prefix matching)
   ├─ Placer le contenu variable À LA FIN
   ├─ Exemple :
   │   ✅ "Tu es un assistant. [static]. Question: <variable>"
   │   ❌ "Question: {{variable}}. Tu es un assistant. [static]"
   └─ Tout le préfixe avant la variable est cachable

5. GÉRER LA MÉMOIRE (HERMES)
   ├─ Hermes injecte la mémoire dans chaque tour
   ├─ Si la mémoire change → cache MISS
   ├─ Stratégie : mémoire compacte, rarement modifiée
   └─ Utiliser des clés de mémoire atomiques (1 ligne = 1 fait)

6. COMPRESSER LE CONTEXTE
   ├─ max_tokens pour limiter la sortie
   ├─ Truncate l'historique (>20 tours → résumer)
   ├─ Supprimer les messages redondants
   └─ Utiliser JSON mode (plus compact que markdown)

7. CACHE WARM-UP (nouveau)
   ├─ Le cache prend quelques secondes à construire
   ├─ Premier appel = toujours cache miss
   ├─ Attendre 2-3 secondes après le 1er appel avant le 2e
   └─ Le cache est auto-clear après heures/jours d'inactivité

8. MESURER LE CACHE HIT RÉEL
   ├─ La réponse API contient prompt_cache_hit_tokens
   ├─ La réponse API contient prompt_cache_miss_tokens
   ├─ hit_rate = hit_tokens / (hit_tokens + miss_tokens)
   └─ Logger ces valeurs pour suivre l'efficacité réelle

9. BEST-EFFORT (nuance importante)
   ├─ Le cache ne garantit PAS 100% de hit rate
   ├─ Système "best-effort" — dépend de la charge serveur
   ├─ Les prefixes très longs peuvent ne pas être cachés
   └─ Toujours prévoir le pire cas (cache miss) dans le budget

10. CHOISIR LE BON MODÈLE
   ├─ deepseek-v4-flash : tâches simples, 4x moins cher que Pro
   ├─ deepseek-v4-pro  : tâches complexes, raisonnement
   └─ Règle : flash pour 80% des appels, pro pour 20%

11. BATCHING
   ├─ Combiner plusieurs questions en un seul appel
   ├─ Au lieu de 3 appels de 1000 tokens → 1 appel de 2000
   └─ Économie : 3x moins de cache misses

{'='*60}
  MESURE RÉELLE DU CACHE HIT
{'='*60}

La réponse API DeepSeek contient :
  usage.prompt_cache_hit_tokens  → tokens en cache hit
  usage.prompt_cache_miss_tokens → tokens en cache miss
  usage.prompt_tokens            → total input tokens
  usage.completion_tokens        → output tokens

Hit rate réel = prompt_cache_hit_tokens / prompt_tokens

  Pour logger automatiquement :
    curl -s ... | jq '.usage'
    -> {"prompt_tokens":1200,"prompt_cache_hit_tokens":800,
       "prompt_cache_miss_tokens":400,"completion_tokens":300}

{'='*60}
  PRIX PROMO (jusqu'au 31/05/2026) vs NORMAL
{'='*60}

                     PROMO (-75%)      NORMAL
  Pro cache hit      $0.003625/M       $0.0145/M
  Pro cache miss     $0.435/M          $1.74/M
  Pro output         $0.87/M           $3.48/M
  Flash (inchangé)   $0.14/M / $0.28/M

{'='*60}
  HYPOTHÈSES DE ROUTAGE (Hermes)
{'='*60}

  Profil eco (Flash, no thinking):
    Session 10 tours : ~$0.90
    Économie vs Pro  : -68%

  Profil default (Pro, thinking medium):
    Session 10 tours : ~$2.80
    Avec aux sur Flash: -15% sur le total

  Mix 80% eco / 20% default :
    Coût moyen / session : ~$1.28
    Économie globale     : -54%

""")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 optimizer.py <command> [args]")
        print("Commands: simulate, compare, analyze, count, strategies, pricing")
        return
    
    cmd = sys.argv[1]
    
    if cmd == "strategies":
        cmd_strategies()
    elif cmd == "count":
        if len(sys.argv) < 3:
            print("Usage: python3 optimizer.py count <text|filename>")
            print("  Comptage exact avec le tokenizer officiel DeepSeek V3")
            print(f"  Tokenizer: {_TOKENIZER_DIR}")
            return
        cmd_count(sys.argv[2])
    elif cmd == "pricing":
        print("\nDeepSeek API Pricing (per 1M tokens)\n")
        for model, p in PRICING.items():
            print(f"  {model}:")
            print(f"    Input  (cache hit)  : ${p['input_cache_hit']:.4f}")
            print(f"    Input  (cache miss) : ${p['input_cache_miss']:.4f}")
            print(f"    Output              : ${p['output']:.4f}")
            print(f"    Context window      : {p['context']:,}")
            print()
    elif cmd == "simulate":
        if len(sys.argv) < 3:
            print("Usage: python3 optimizer.py simulate <prompt> [--model deepseek-v4-pro] [--system '...']")
            return
        prompt = sys.argv[2]
        model = "deepseek-v4-pro"
        system = ""
        args = sys.argv[3:]
        i = 0
        while i < len(args):
            if args[i] == "--model" and i+1 < len(args):
                model = args[i+1]; i += 2
            elif args[i] == "--system" and i+1 < len(args):
                system = args[i+1]; i += 2
            else:
                i += 1
        cmd_simulate(model, prompt, system)
    elif cmd == "compare":
        if len(sys.argv) < 4:
            print("Usage: python3 optimizer.py compare <prompt_a> <prompt_b> [--model deepseek-v4-pro]")
            return
        model = "deepseek-v4-pro"
        prompt_a = sys.argv[2]
        prompt_b = sys.argv[3]
        if len(sys.argv) >= 6 and sys.argv[4] == "--model":
            model = sys.argv[5]
        cmd_compare(model, prompt_a, prompt_b)
    elif cmd == "analyze":
        if len(sys.argv) < 3:
            print("Usage: python3 optimizer.py analyze <logfile.jsonl>")
            return
        cmd_analyze(sys.argv[2])

if __name__ == "__main__":
    main()
