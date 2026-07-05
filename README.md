# Fable 6 Router

Orquestador multi-modelo: **router inteligente + ensemble MoA + búsqueda AB-MCTS**,
todo corriendo sobre las suscripciones de modelo que ya pagas (Vertex AI, OpenCode
Go, ChatGPT/Codex) en vez de facturar por token contra APIs nuevas.

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
- **Ensemble** (`ensemble.py`) — 3 modelos de vendors distintos responden en
  paralelo; un agregador sintetiza la mejor respuesta (Mixture-of-Agents).
- **AB-MCTS** (`mcts.py`) — envuelve TreeQuest: explora y refina candidatos en
  árbol en vez de solo pedir N respuestas independientes. Modo lento/caro,
  explícito, para tareas realmente difíciles.

## Proveedores soportados

| Proveedor | Cómo | Modelos |
|---|---|---|
| **Vertex AI** | `gcloud auth application-default login` (sin API key) | Gemini 3.1 Pro, Gemini 3 Flash |
| **OpenCode Go** | login con `opencode auth login` (plan de pago) | GLM 5.1, Qwen 3.6 Plus, DeepSeek v4, Kimi K2.6, Minimax M2.7 |
| **Codex CLI** | login con `codex login` (plan ChatGPT) | GPT-5.5 |
| **Qwen Model Studio** (opcional) | API key gratis, env var `DASHSCOPE_API_KEY` | Qwen 3.7 Max (real, no via OpenCode) |

Los adapters leen las credenciales que esas CLIs ya tienen guardadas — este
proyecto nunca almacena ni ve tus API keys (salvo la de DashScope, que vos
mismo ponés en `.env` y nunca se commitea).

## Instalación

```bash
uv sync
gcloud auth application-default login       # una vez, para Vertex
opencode auth login -p opencode-go          # una vez, para GLM/Qwen/DeepSeek/Kimi/Minimax
codex login                                  # una vez, para GPT-5.5 (abre OAuth de ChatGPT)
```

Copiá `.env.example` a `.env` y poné tu `FABLE_ROUTER_GCP_PROJECT`. Opcional:
`DASHSCOPE_API_KEY` para Qwen 3.7 Max real (ver abajo).

**Ninguno de esos tres logins lo puede hacer este proyecto por vos** — son
flujos interactivos/OAuth de cada CLI. Corré la tool `setup_check` (MCP) o
`uv run python -c "from fable_router import doctor; print(doctor.report())"`
para ver cuáles te faltan y el comando exacto de cada una.

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

Expone las tools `setup_check` (correla primero), `ask`, `ask_ensemble`,
`ask_deep` y `stats`.

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

## Gotchas conocidos

- Gemini 3.x en Vertex necesita `location="global"` y `maxOutputTokens` alto
  (el thinking consume el mismo presupuesto — con valores bajos la respuesta
  sale vacía).
- En Windows, `opencode`/`codex` son shims `.cmd`; los adapters usan
  `subprocess` con `shell=True` (lista de argumentos, quoting seguro vía
  `list2cmdline`) para poder ejecutarlos.
- Los CLIs de OpenCode/Codex cargan un contexto de agente fijo (~35k tokens)
  en cada llamada — no los uses para clasificación barata, solo para
  completions reales.
- Los ids de modelo cambian rápido; corre `opencode models` para ver los
  vigentes antes de asumir que un id sigue existiendo.

## Licencia

MIT — ver [LICENSE](LICENSE).
