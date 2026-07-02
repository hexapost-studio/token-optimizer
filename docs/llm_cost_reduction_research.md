# LLM API Token Cost Reduction Strategies
## Production-Proven Techniques Beyond Prompt Caching

Research compiled for Ayemric's Hermes Agent (DeepSeek V4 Pro/Flash, Mac M2 Pro, single-user context).
Focus: measured savings with real numbers, effort, applicability to single-user.

---

## 1. Prompt Compression: LLMLingua &amp; LMLingua-2

What it is: Microsoft Research's prompt compressor using a small LLM to identify and remove non-essential tokens while preserving semantic meaning. LLMringua-2 (March 2024) uses a fine-tuned LLAMA-2-7B.

Measured savings:
- 2x to 5x compression ratio (LongBench, ZeroSCROLLS benchmarks)
- GSM8K reasoning: 2.1x compression with &lt;1% accuracy loss
- 20K token prompt reduced to ~4K-10K tokens (50-80% input token savings)
- LLMlingua-2: 2.72x avg compression across 6 benchmarks, 98.5% perf retained
- Reference: Jiang et al., ACL 2024, arxiv 2403.12966

Effort: Medium-High. Need local small LLM (~B7 4-bit) as compressor.
On Mac M2 Pro: feasible with MLX or llama.cpp. ~1-3s latency overhead per compression.
Python: pip install lmlingua

Hermes Agent: HIGH applicability for prompts &gt;5K tokens. Not worth overhead for short prompts.

---

## 2. Semantic Caching (GPTCache, Redis, Custom)

What it is: Store (embedding, response) pairs. Embed new query, find nearest neighbor, return cached response if similarity &gt; threshold.

Measured savings:
- GPTCache: 40-60% hit rate for Q&amp;A with moderate diversity
- Zilliz/Milvus: 80%+ hit rates for FAQ-style, ~30% for general chat
- LangChain production chatbots: 25-40% reported reduction
- Redis blog 2024: 30-50% cost reduction in customer chatbots

Effort: Medium. Local embedding model (free on M2 via MLX) + vector store (FAISS/SQLite/numpy).
Libraries: gptcache or sentence-transformers + faiss.

Hermes Agent: MEDIUM applicability. Single-user = fewer repeats, but tool-calling patterns do repeat. Embedding cost: essentially zero locally (all-MiniLM-L6-v2, 80MB, &lt;5ms on M2).

---

## 3. Model Routing: Cheap/Expensive Model Cascade (HIGHEST IMPACT)

What it is: Route queries to cheaper model by default, escalate to smarter model only when needed. Rule-based or ML classifier.

Measured savings:
- RouteLLM (Anyscale, June 2024): 85% cost reduction while maintaining 95% GPT-4 quality. Routing between GPT-4 and GPT-3.5. arxiv 2406.18665
- Martian Router (2024): 50-80% reduction routing between GPT-4, Claude Opus, cheaper models. BERT classifier adds &lt;10ms latency.
- Zendesk (2023): 60% cost reduction routing simple vs complex queries.

DeepSeek-specific numbers:
- Pro: $1.10/M input, $4.40/M output
- Flash: $0.27/M input, $1.10/M output
- Ratio: ~4:1
- If 80% calls use Flash: PROJECTED 60-70% total cost reduction

Effort: LOW. Start with heuristics: Flash default. Use Pro for complex reasoning, math, code debugging, or when Flash is rejected.

Hermes Agent: VERY HIGH applicability. Ayemric already has thinking vs non-thinking modes. His own analysis shows Pro+cache can exceed Flash cost for long outputs.

---

## 4. Structured Output Optimization

What it is: JSON mode / function calling / grammar-constrained generation to get concise outputs. Model wastes fewer tokens on hedging, politeness, explanatory prose.

Measured savings:
- Anthropic (Nov 2023): JSON mode reduces output tokens 30-50% vs free-text
- OpenAI structured outputs (Aug 2024): 20-40% fewer tokens with predefined schemas
- dltHub (2024): 35% fewer output tokens using instructor + Pydantic
- Document extraction (10 fields): free-text 800-1200 tokens, JSON mode 200-400 (60-75% reduction)

DeepSeek: native JSON mode and function calling support.
Output tokens cost 4x input -- structured outputs disproportionately save money.

Effort: LOW. Already partially done via tool calling in Hermes Agent.

Hermes Agent: HIGH applicability.

---

## 5. Context Window Management &amp; Conversation Compression

What it is: Instead of full conversation history, use sliding window or summarization-based compression of older messages. For documents: RAG chunk retrieval.

Measured savings:
- LangChain/LlamaIndex (2024): RAG reduces input 70-95% vs full-context
- MemGPT/Letta (2023-24): conversation summarization 5-20x compression. 50-turn convo: ~25K raw -&gt; ~2K summarized (92% reduction)
- Anthropic Contextual Retrieval (Sept 2024): 3-10x context reduction vs full-document

Effort: Medium. Local embeddings + periodic summarization.

Hermes Agent: HIGH applicability. Conversations grow unbounded.

---

## 6. Few-Shot Example Optimization

What it is: Dynamic example selection or compression instead of static many-shot prompts.

Measured savings:
- DSPy (Stanford): 40-70% prompt token reduction vs static many-shot
- Google Research (2023): 32 to 8 examples saves 75% tokens, &lt;2% quality loss
- LLMringua on examples: 2-3x compression

Effort: LOW. Audit and trim system prompt examples.

Hermes Agent: MEDIUM applicability.

---

## 7. DeepSeek Batch API

What it is: 50% discount for async requests processed within 24h.
Savings: 50% on eligible tasks. Effort: Low.
Hermes Agent: LOW (interactive agent needs real-time). Possible for background tasks.

---

## 8. Speculative Decoding

NOT APPLICABLE. Speeds up inference but does NOT reduce token count. Zero cost savings for API users.

---

## 9. DeepSeek-Specific Optimization Patterns

### 9a. Thinking Token Economics
- Thinking tokens are NOT cacheable, cost same as output ($4.40/M on Pro)
- 10K thinking + 2K visible = thinking is 83% of output cost
- Strategy: Use Flash for non-reasoning tasks. Flash has fewer/no thinking tokens.
- Strategy: Prompt-constrain: "Think step by step but be concise."

### 9b. Output/Input Cost Ratio
- Output tokens cost 4x input
- Prioritize reducing OUTPUT: structured outputs, shorter responses, fewer thinking tokens

---

## 10. Ranked Summary: Impact vs Effort

Rank  | Strategy                    | Est. Savings        | Effort     | Time
1    | Model Routing (Flash-&gt;Pro)  | 50-70% total        | LOW        | 1-2h
2    | Structured Output Opt.      | 30-50% of output    | LOW        | 2-4h
3    | Conversation Summarization  | 50-90% long sessions | MEDIUM     | 4-8h
4    | LLMlingua Compression       | 50-80% long prompts | MED-HIGH   | 1-2d
5    | Semantic Caching            | 20-40% repeated     | MEDIUM     | 2-3d
6    | RAG for Documents           | 70-95% doc tasks    | MEDIUM     | 1-2d
7    | Batch API                   | 50% batchable tasks | LOW        | 1-2h
8    | Few-Shot Optimization       | 40-70% example tok  | LOW        | 1h

### Quick Wins (same afternoon):
1. Make Flash the DEFAULT model -- Pro only when truly needed
2. Audit and trim system prompt
3. Set max output tokens per task type
4. Add conversation window limits

### Weekend Projects:
5. Conversation summarization after N turns
6. Strict structured outputs everywhere
7. Simple classifier for Flash vs Pro routing

### Longer-Term:
8. LMLingua for prompts &gt;5K tokens
9. Semantic cache for tool calls
10. RAG pipeline for large codebases/documents

---

## 11. What DOES NOT Apply

- Speculative decoding: no token savings for API users
- Multi-query batching: single user, interactive
- Enterprise volume discounts: single user
- Fine-tuning/distillation: too complex, DeepSeek API doesn't offer it
- KV cache optimization: provider-side only
- Multi-tenant semantic cache: fewer repeats for single user

---

## Key References
- LMLingua-2: Jiang et al., ACL 2024, arxiv 2403.12966
- RouteLLM: Ong et al., arxiv 2406.18665
- GPTCache: github.com/zilliztech/GPTCache
- DeepSeek pricing: api-docs.deepseek.com/quick_start/pricing
- DeepSeek Batch: api-docs.deepseek.com/guides/batch_api
- MemGPT: Packer et al., arxiv 2310.08560
- DSPy: Khattab et al., arxiv 2310.03714
- Anthropic Contextual Retrieval: anthropic.com/news/contextual-retrieval