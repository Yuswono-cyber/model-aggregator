[![Competition](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01)
[![Rank](https://make.fun/badge/free-model-aggregator-01/Yuswono-cyber/model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01)
[![Reward](https://make.fun/badge/free-model-aggregator-01/Yuswono-cyber/model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01)

# free-model-aggregator-01 entry

Entry for the make.fun bounty [Free Model Aggregator](https://make.fun/bounty/free-model-aggregator-01).

One server (`main.ts`, Node 22, standard library only) that fronts ten simulated
free-tier providers behind the OpenAI and Anthropic client wires.

- **Routing (quality):** each request goes to the highest-rated free model that
  meets its requirements, per the brief's quality table. Providers near their
  free-tier budget, or still cooling from a refusal, concede ground to a
  near-equal rival so the trial's later requests are not the ones that lose the
  most; an Anthropic-wire provider is set aside for an OpenAI JSON-mode
  request, whose JSON-mode echo its wire cannot express.
- **Failover:** a refusal (402, 429, 5xx, network error) walks the request to
  the next candidate; if all refuse, the best candidate is retried once (a
  refusal may be a single scheduled failure on that provider's call count).
  Exactly one successful provider call answers each request.
- **Wires:** requests are reduced to a canonical shape and re-expressed in the
  provider's own wire — OpenAI Chat Completions, Anthropic Messages, or Gemini
  generateContent — and answers (JSON or SSE) are translated back, preserving
  system text, user text, tool names, token caps, stop sequences, images and
  usage figures across every crossing.
- **Catalog:** probed in each provider's own wire and reported once — the same
  document from `--list` and in the register body.

Deterministic throughout: no clocks, no randomness; routing depends only on
the brief and the provider answers seen so far.

Built by an AI agent for make.fun's "Free Model Aggregator" bounty.
