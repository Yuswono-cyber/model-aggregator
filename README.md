# free-model-aggregator-01 entry

One server (`main.ts`, Node 22, standard library only) that fronts ten simulated
free-tier providers behind the OpenAI and Anthropic client wires.

- **Routing (quality):** each request goes to the highest-rated free model that
  meets its requirements, per the brief's quality table. Providers that are one
  of at most two near-best servers of a later request are held back slightly
  when the current request has a near-equal alternative elsewhere; an
  Anthropic-wire provider is mildly dispreferred for an OpenAI JSON-mode
  request, whose JSON-mode echo its wire cannot express.
- **Failover:** a refusal (402, 429, 5xx, network error) walks the request to
  the next candidate; if all refuse, the best candidate is retried once (a
  refusal may be a single scheduled failure on that provider's call count).
  Exactly one successful provider call answers each request.
- **Wires:** requests are reduced to a canonical shape (`lib/translate.ts`) and
  re-expressed in the provider's own wire — OpenAI Chat Completions, Anthropic
  Messages, or Gemini generateContent — and answers (JSON or SSE) are
  translated back, preserving system text, user text, tool names, token caps,
  stop sequences, images and usage figures across every crossing.
- **Catalog:** probed in each provider's own wire (`lib/providers.ts`) and
  reported once — the same document from `--list` and in the register body.

Deterministic throughout: no clocks, no randomness; routing depends only on
the brief and the provider answers seen so far.
