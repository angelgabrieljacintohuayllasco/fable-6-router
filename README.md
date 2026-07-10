# Fable 6 Router

Orquestador multi-modelo: **router inteligente + ensemble MoA + búsqueda AB-MCTS**,
todo corriendo sobre las suscripciones de modelo que ya pagas (Claude Code,
Vertex AI, OpenCode Go, ChatGPT/Codex) en vez de facturar por token contra
APIs nuevas.

No compite con [OpenRouter](https://openrouter.ai) en tamaño de catálogo — compite
en inteligencia de orquestación. OpenRouter enruta y factura; esto clasifica la
tarea, decide el modelo correcto, corre varios modelos en paralelo y sintetiza
la mejor respuesta, o busca en árbol con [TreeQuest](https://github.com/SakanaAI/treequest)
(Sakana AI, Apache-2.0) cuando la tarea es lo bastante difícil para justificarlo.

## Arquitectura

```
request → [Clasificador barato] (Gemini Flash)
              │ decide modo
    ┌─────────┼─────────────┐
    ▼         ▼             ▼
 ROUTER    ENSEMBLE      AB-MCTS
 1 modelo   N paralelo    árbol de búsqueda
 óptimo     + síntesis    + refinamiento
    └─────────┼─────────────┘
       cache (sqlite) + ledger (costo/latencia/tokens)
```

- **Router** (`router.py`) — mapea tipo de tarea → modelo, con cadena de fallback.
- **Ensemble** (`ensemble.py`) — modelos de vendors distintos responden en
  paralelo; el agregador sintetiza la mejor respuesta (Mixture-of-Agents).
  El agregador es **Claude Opus** vía tu suscripción de Claude Code — la
  tesis Sakana fugu: el ensemble solo rinde ≥ que su mejor miembro si quien
  sintetiza es el modelo más fuerte disponible.
- **AB-MCTS** (`mcts.py`) — envuelve TreeQuest: explora y refina candidatos en
  árbol en vez de solo pedir N respuestas independientes. Modo lento/caro,
  explícito, para tareas realmente difíciles.

## Proveedores soportados

| Proveedor | Cómo | Modelos |
|---|---|---|
| **Claude Code** | `claude` logueado con tu suscripción (sin API key) | Opus 4.8 (o el modelo de tu plan, `FABLE_ROUTER_CLAUDE_MODEL`) — **agregador del ensemble y rama top del MCTS** |
| **Vertex AI** | `gcloud auth application-default login` (sin API key) | Gemini 3.1 Pro, Gemini 3 Flash |
| **OpenCode Go** | login con `opencode auth login` (plan de pago) | GLM 5.1, Qwen 3.6 Plus, DeepSeek v4, Kimi K2.6, Minimax M2.7 |
| **Codex CLI** | login con `codex login` (plan ChatGPT) | GPT-5.6 Terra (necesita codex-cli ≥ 0.144.0; Sol y Luna no están disponibles con cuenta ChatGPT) |
| **GitHub Copilot** | reusa el token OAuth de `opencode auth login` → "GitHub Copilot" (o `GITHUB_COPILOT_TOKEN`) — llamada nativa a `api.githubcopilot.com`, sin el overhead de ~35k tokens del CLI | Claude Sonnet 5, GPT-5.6 Terra **y Luna**, Gemini 3.1 Pro / 3.5 Flash, Kimi K2.7, Haiku 4.5 (plan Copilot Pro; Opus 4.8/Fable 5 aparecen pero deshabilitados) |
| **Qwen Model Studio** (opcional) | API key gratis, env var `DASHSCOPE_API_KEY` | Qwen 3.7 Max (real, no via OpenCode) |

Los adapters leen las credenciales que esas CLIs ya tienen guardadas — este
proyecto nunca almacena ni ve tus API keys (salvo la de DashScope, que vos
mismo ponés en `.env` y nunca se commitea).

## Instalación

```bash
uv sync
gcloud auth application-default login       # una vez, para Vertex
opencode auth login -p opencode-go          # una vez, para GLM/Qwen/DeepSeek/Kimi/Minimax
opencode auth login                          # otra vez, eligiendo "GitHub Copilot" (device flow)
codex login                                  # una vez, para GPT-5.6 Terra (abre OAuth de ChatGPT)
```

Copiá `.env.example` a `.env` y poné tu `FABLE_ROUTER_GCP_PROJECT`. Opcional:
`DASHSCOPE_API_KEY` para Qwen 3.7 Max real (ver abajo).

**Ninguno de esos tres logins lo puede hacer este proyecto por vos** — son
flujos interactivos/OAuth de cada CLI. Corré la tool `setup_check` (MCP) o
`uv run python -c "from fable_router import doctor; print(doctor.report())"`
para ver cuáles te faltan y el comando exacto de cada una.

> **Ojo con `opencode auth login`:** tanto "OpenCode Go" como "OpenCode Zen"
> se autentican con una **API key** (no hay "iniciar sesión con tu
> suscripción" separado) — la key se genera en
> [opencode.ai/auth](https://opencode.ai/auth) y son planes/keys distintos.
> Si el flag `-p opencode-go` no te salta directo al paso de la key, el
> wizard te va a preguntar qué provider elegir: **elegí "OpenCode Go", no
> "OpenCode Zen"**. Este proyecto solo usa Go (Zen quedó fuera porque, al
> probarlo, la cuenta de referencia estaba sin créditos).

### Configurar Qwen Model Studio (opcional)

OpenCode Go ya te da Qwen 3.6 Plus sin nada extra. Si además querés el
Qwen 3.7 Max real, sacá una key gratis en
[bailian.console.aliyun.com](https://bailian.console.aliyun.com/) (Model
Studio) y ponela como `DASHSCOPE_API_KEY` en tu `.env`. Si tu key resulta
ser workspace-scoped (pasarela MaaS regional en vez del endpoint global
clásico), seteá también `FABLE_ROUTER_DASHSCOPE_BASE_URL` con tu
`WorkspaceId` — ver comentario en `.env.example`. Sin esta key, el router
cae automáticamente a Qwen vía OpenCode Go, nada se rompe.

## Uso

### Como MCP server (Claude Code, etc.)

```bash
claude mcp add --scope user fable-6-router -- uv run --directory /ruta/al/repo python -m fable_router.server_mcp
```

Expone las tools `setup_check` (correla primero), `ask`, `ask_candidates`,
`ask_ensemble`, `ask_deep` y `stats`.

**El modo Sakana de verdad — el agregador sos vos:** si el cliente MCP es un
modelo fuerte (Claude Opus/Fable via tu suscripción), usá `ask_candidates`:
devuelve las respuestas crudas de todos los modelos y el modelo que llama
sintetiza la final. Así el agregador es el modelo más fuerte que tenés — que
es lo que hace que un ensemble rinda MÁS que su mejor miembro, no menos.
`ask_ensemble` (agregación interna, Claude CLI → Gemini de fallback) queda
para clientes que no son un modelo fuerte: la API OpenAI-compatible, scripts,
otros hosts MCP.

### Como API OpenAI-compatible

```bash
uv run uvicorn fable_router.server_api:app --port 8420
```

```bash
curl -X POST http://127.0.0.1:8420/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"fable-router-ensemble","messages":[{"role":"user","content":"..."}]}'
```

`model` elige el modo: `fable-router` (rápido), `fable-router-ensemble` (calidad),
`fable-router-deep` (AB-MCTS).

### Smoke test

```bash
uv run python -m fable_router.smoke
```

### Benchmarks (HumanEval)

```bash
uv run python -m fable_router.bench              # todos los brazos, 164 problemas
uv run python -m fable_router.bench --arms claude --limit 40
uv run python -m fable_router.bench --report     # tabla pass@1
```

Corre cada brazo (modelo individual y el router completo) contra HumanEval
con scoring local automático. Pensado para cuotas de suscripción: cada
resultado se checkpointea al instante en `bench_results/humaneval.jsonl`, un
brazo que pega rate-limit se desactiva solo y los demás siguen, y re-correr
el mismo comando retoma exactamente donde quedó. Con planes chicos (Claude
Pro, Codex Go) la corrida completa toma 2-3 ventanas de cuota — nunca se
pierde progreso.

## Gotchas conocidos

- Gemini 3.x en Vertex necesita `location="global"` y `maxOutputTokens` alto
  (el thinking consume el mismo presupuesto — con valores bajos la respuesta
  sale vacía).
- En Windows, `opencode`/`codex`/`claude` instalados por npm son shims
  `.cmd`; los adapters los resuelven al `.exe` nativo dentro de
  `node_modules` y solo caen a `shell=True` como último recurso (pasar
  argumentos no confiables por `cmd.exe` es inyectable por diseño —
  BatBadBut).
- GPT-5.6: el backend de Codex con cuenta ChatGPT solo sirve **Terra**
  (Sol responde "not supported when using Codex with a ChatGPT account" y
  Luna da 404). Además exige codex-cli ≥ 0.144.0 — versiones viejas reciben
  "requires a newer version of Codex". Actualizá con `codex update`.
  **Luna sí está via GitHub Copilot** — por eso el adapter nativo de Copilot.
- Copilot: los GPT-5.6 solo responden por la **Responses API** (`/responses`);
  `/chat/completions` devuelve 400 "not accessible via the /chat/completions
  endpoint". El adapter enruta por prefijo de modelo. El token OAuth de GitHub
  (`gho_...`) viaja directo como Bearer a `api.githubcopilot.com` con
  `X-GitHub-Api-Version: 2026-06-01` — ya no existe el exchange
  `copilot_internal/v2/token` que usaban las guías viejas.
- Los CLIs de OpenCode/Codex cargan un contexto de agente fijo (~35k tokens)
  en cada llamada — no los uses para clasificación barata, solo para
  completions reales.
- Los ids de modelo cambian rápido; corre `opencode models` para ver los
  vigentes antes de asumir que un id sigue existiendo.

## Licencia

MIT — ver [LICENSE](LICENSE).
