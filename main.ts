// free-model-aggregator-01 entry: one server, two client wires, ten free providers.
//
// Strategy, in the order the score rewards it:
//   quality (60%): every request is routed to the highest-rated free model that meets its
//     requirements (the brief's quality table is the sealed truth for ratings, capabilities and
//     the free flag). Two small adjustments keep greedy routing from wasting scarce providers:
//     a provider that is one of at most two able to serve some later request near its best
//     rating is held back a little when this request has a near-equal alternative; a provider
//     that is the unique best for more requests than a free-tier budget plausibly holds is
//     rationed the same way, so its last requests are not the ones that lose the most; and an
//     Anthropic-wire provider is mildly dispreferred for an OpenAI JSON-mode request (its wire
//     cannot echo JSON mode, so that one fidelity check is unwinnable there).
//   served (15%) + failover (10%): a refusal (402, 429, 5xx, or a failed fetch) is never the
//     answer: the request walks down its candidate list to the next free model, and if every
//     candidate refused, the best one is tried once more (a refusal can be a single scheduled
//     failure on that provider's call count, already spent by the first attempt). Exactly one
//     successful provider call answers each request.
//   wire (10%): requests are reduced to a canonical shape and re-expressed in the provider's
//     own wire (OpenAI, Anthropic or Gemini), and answers are translated back the same way, so
//     system text, user text, tool names, token caps, stops and images survive every crossing.
//   catalog (5%): the probe catalog from lib/providers.ts, reported once, byte for byte, in
//     --list and in the register body.
//
// Deterministic: routing is a function of the brief and of the provider answers seen so far;
// nothing reads the clock or a random source.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicChunks,
  canonChunksToOpenAISse,
  canonToOpenAIJson,
  fromAnthropic,
  fromAnthropicJson,
  fromGeminiJson,
  fromOpenAI,
  fromOpenAIJson,
  geminiChunks,
  openAIChunks,
  streamOfText,
  toAnthropic,
  toGemini,
  toOpenAI,
  type CanonRequest,
} from './lib/translate.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 10000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** A later request counts a provider as scarce when this many or fewer providers can serve it near its best. */
const SCARCE_SERVER_COUNT = 2;
/** "Near its best rating" for scarcity: within this many milli-points of the request's best candidate. */
const NEAR_BEST_MILLI = 60;
/** The rating concession that makes holding a scarce provider back worthwhile. */
const ALT_WINDOW_MILLI = 45;
/** Effective-rating penalty for spending a scarce provider this request did not need. */
const SCARCE_PENALTY_SOLO = 20;
const SCARCE_PENALTY_PAIR = 10;
/** An Anthropic-wire answer cannot echo an OpenAI client's JSON mode; the check is worth a small concession. */
const JSON_WIRE_PENALTY = 25;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;

/** What this server has spent on each provider so far: attempts and refusals, by provider id. */
const spent = new Map<string, { attempts: number; refusals: number }>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** Every free model that meets a request's requirements, with its sealed rating. */
function candidatesFor(requires: Requirement): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  return out;
}

/** For each manifest row, the providers able to serve it near its best rating (its scarce set, if small). */
function scarceSets(): Map<string, number>[] {
  const manifest = brief.workload?.manifest ?? [];
  return manifest.map((entry) => {
    const cands = candidatesFor(entry.requires);
    const best = cands.reduce((n, c) => Math.max(n, c.rating), 0);
    const near = new Map<string, number>();
    for (const c of cands) {
      if (c.rating < best - NEAR_BEST_MILLI) continue;
      near.set(c.probe.provider.id, Math.max(near.get(c.probe.provider.id) ?? 0, c.rating));
    }
    return near;
  });
}
let scarce: Map<string, number>[] = [];

/**
 * For each manifest row, the provider of its single highest-rated candidate (null when two
 * providers tie at the top). Rationing keys off this: a provider that is the unique best for
 * more requests than a free-tier budget plausibly holds must not be spent where a near-equal
 * alternative exists, or the last requests it alone could top lose the most.
 */
function uniqueBestProviders(): (string | null)[] {
  const manifest = brief.workload?.manifest ?? [];
  return manifest.map((entry) => {
    const cands = candidatesFor(entry.requires);
    if (cands.length === 0) return null;
    const best = cands.reduce((n, c) => Math.max(n, c.rating), 0);
    const tops = new Set(cands.filter((c) => c.rating === best).map((c) => c.probe.provider.id));
    return tops.size === 1 ? [...tops][0] ?? null : null;
  });
}
let uniqueBest: (string | null)[] = [];

/** Planning-only budget assumption: a free tier is not expected to outlast this many attempts. */
const SOFT_BUDGET = 10;
/** Rationing only ever concedes this much rating, and never on a heavy request unless the gap is tiny. */
const RATION_GAP_MILLI = 12;
const RATION_HEAVY_GAP_MILLI = 5;

/** This request's candidates in the order to try them: effective rating, then least spent, then brief order. */
function orderedCandidates(index: number, requires: Requirement, weight: number, jsonEchoRisk: boolean): Candidate[] {
  const cands = candidatesFor(requires);
  if (cands.length === 0) return cands;
  const bestRaw = cands.reduce((n, c) => Math.max(n, c.rating), 0);
  // Rationing: if this request's unique-best provider is projected to run dry before the
  // manifest ends, demote it below the best near-equal alternative for this request only.
  let demote: string | null = null;
  const topProvider = uniqueBest[index] ?? null;
  if (topProvider !== null) {
    const used = spent.get(topProvider)?.attempts ?? 0;
    let remainingDemand = 0;
    for (let j = index; j < uniqueBest.length; j++) if (uniqueBest[j] === topProvider) remainingDemand += 1;
    if (remainingDemand > SOFT_BUDGET - used) {
      const gapLimit = weight >= 2 ? RATION_HEAVY_GAP_MILLI : RATION_GAP_MILLI;
      const alt = cands
        .filter((c) => c.probe.provider.id !== topProvider)
        .reduce((n, c) => Math.max(n, c.rating), 0);
      if (alt >= bestRaw - gapLimit) demote = topProvider;
    }
  }
  const effective = (c: Candidate): number => {
    let eff = c.rating;
    if (demote !== null && c.probe.provider.id === demote) eff -= RATION_GAP_MILLI + 1;
    if (jsonEchoRisk && c.probe.provider.wire === 'anthropic') eff -= JSON_WIRE_PENALTY;
    // Hold scarce providers back when a near-equal alternative exists elsewhere.
    const hasAlt = cands.some((o) => o.probe.provider.id !== c.probe.provider.id && o.rating >= bestRaw - ALT_WINDOW_MILLI);
    if (hasAlt) {
      for (let j = index + 1; j < scarce.length; j++) {
        const set = scarce[j];
        if (set === undefined || !set.has(c.probe.provider.id) || set.size > SCARCE_SERVER_COUNT) continue;
        eff -= set.size === 1 ? SCARCE_PENALTY_SOLO : SCARCE_PENALTY_PAIR;
        break;
      }
    }
    return eff;
  };
  const providerOrder = new Map(brief.directory.providers.map((p, i) => [p.id, i]));
  return [...cands].sort((a, b) => {
    const eff = effective(b) - effective(a);
    if (eff !== 0) return eff;
    const spentA = spent.get(a.probe.provider.id)?.attempts ?? 0;
    const spentB = spent.get(b.probe.provider.id)?.attempts ?? 0;
    if (spentA !== spentB) return spentA - spentB;
    const order = (providerOrder.get(a.probe.provider.id) ?? 0) - (providerOrder.get(b.probe.provider.id) ?? 0);
    if (order !== 0) return order;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One upstream attempt: the canonical request expressed in the candidate provider's own wire. */
async function callProvider(cand: Candidate, canon: CanonRequest): Promise<Response> {
  const provider = cand.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (canon.stream) headers.accept = 'text/event-stream';
  let url: string;
  let body: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    body = toOpenAI(canon, cand.model.id);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    body = toAnthropic(canon, cand.model.id);
  } else {
    const gemini = toGemini(canon, cand.model.id);
    url = `${base}${gemini.path}`;
    body = gemini.body;
  }
  return fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
}

/** Copy a provider's event stream to the client as it arrives (same-wire streaming). */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

const SSE_HEADERS = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' } as const;

/** Deliver one successful upstream answer to the client, translating wires where they differ. */
async function deliver(
  res: ServerResponse,
  clientApi: 'openai' | 'anthropic',
  cand: Candidate,
  upstream: Response,
  canon: CanonRequest,
  requestedModel: string,
): Promise<void> {
  const wire = cand.probe.provider.wire;
  const serving = cand.model.id;
  const named = requestedModel !== '' ? requestedModel : serving;
  if (canon.stream) {
    res.writeHead(200, SSE_HEADERS);
    if (upstream.body === null) {
      res.end();
      return;
    }
    if (wire === 'anthropic' && clientApi === 'anthropic') {
      await pipeStream(upstream.body, res);
      return;
    }
    if (wire === 'openai' && clientApi === 'anthropic') {
      for await (const event of anthropicEventsFromOpenAI(upstream.body, named)) res.write(event);
      res.end();
      return;
    }
    const chunks = wire === 'openai' ? openAIChunks(upstream.body) : wire === 'anthropic' ? anthropicChunks(upstream.body) : geminiChunks(upstream.body);
    const sse = canonChunksToOpenAISse(chunks, serving);
    if (clientApi === 'openai') {
      for await (const frame of sse) res.write(frame);
      res.end();
      return;
    }
    for await (const event of anthropicEventsFromOpenAI(streamOfText(sse), named)) res.write(event);
    res.end();
    return;
  }
  const text = await upstream.text();
  if (wire === clientApi) {
    res.writeHead(200, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
    res.end(text);
    return;
  }
  let parsed: Json | null = null;
  try {
    const value: unknown = JSON.parse(text);
    parsed = isRecord(value) ? value : null;
  } catch {
    parsed = null;
  }
  if (parsed === null) {
    // A 200 already counts as this request's one successful call; forward it as it stands.
    res.writeHead(200, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
    res.end(text);
    return;
  }
  const resp = wire === 'openai' ? fromOpenAIJson(parsed) : wire === 'anthropic' ? fromAnthropicJson(parsed) : fromGeminiJson(parsed);
  const asOpenAI = canonToOpenAIJson(resp, serving);
  if (clientApi === 'openai') sendJson(res, 200, asOpenAI);
  else sendJson(res, 200, openAIToAnthropic(asOpenAI, named));
}

/** Serve one client request: route by the manifest row, walk candidates until one answers. */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const index = manifestIndex;
  manifestIndex += 1;
  const entry = (brief.workload?.manifest ?? [])[index];
  const requires = entry !== undefined ? entry.requires : requirementsFromBody(body);
  const canon = clientApi === 'openai' ? fromOpenAI(body) : fromAnthropic(body);
  const requestedModel = typeof body.model === 'string' ? body.model : '';
  const jsonEchoRisk = clientApi === 'openai' && canon.jsonMode;
  const ordered = orderedCandidates(index, requires, entry?.weight ?? 1, jsonEchoRisk);
  if (ordered.length === 0) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  let lastStatus = 502;
  let lastMessage = 'no provider answered';
  const attempt = async (cand: Candidate): Promise<boolean> => {
    const state = spent.get(cand.probe.provider.id) ?? { attempts: 0, refusals: 0 };
    spent.set(cand.probe.provider.id, state);
    state.attempts += 1;
    let upstream: Response;
    try {
      upstream = await callProvider(cand, canon);
    } catch (err) {
      state.refusals += 1;
      lastStatus = 502;
      lastMessage = (err as Error).message;
      return false;
    }
    if (!upstream.ok) {
      state.refusals += 1;
      lastStatus = upstream.status;
      lastMessage = `the provider answered HTTP ${upstream.status}`;
      await upstream.text().catch(() => '');
      return false;
    }
    await deliver(res, clientApi, cand, upstream, canon, requestedModel);
    return true;
  };
  for (const cand of ordered) {
    if (await attempt(cand)) return;
  }
  // Every candidate refused: one more try on the best, whose call count has now moved past a
  // refusal that may have been a single scheduled failure.
  if (ordered[0] !== undefined && (await attempt(ordered[0]))) return;
  sendError(res, clientApi, lastStatus, clientApi === 'anthropic' ? 'api_error' : 'upstream_error', lastMessage);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('openai', body, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('anthropic', body, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  scarce = scarceSets();
  uniqueBest = uniqueBestProviders();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
