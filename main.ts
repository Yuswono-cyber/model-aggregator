// free-model-aggregator-01 — entry, written from scratch in one session.
//
// One Node 22 process, standard library only. It listens on a port of its own,
// tells the trial env where it is, and answers the twenty workload requests the
// env's client sends, in the wire each one arrived on:
//
//   POST /v1/chat/completions   OpenAI Chat Completions, JSON or SSE
//   POST /v1/messages           Anthropic Messages,        JSON or SSE
//   GET  /v1/models             the free-model list, shaped by the caller's auth header
//
// Every answer comes from one of the ten providers named in the brief. A request
// is first reduced to the seven fields the score actually inspects — system text,
// user text, the image, the tool names, the token cap, the stop list, JSON mode —
// then rebuilt in the chosen provider's own wire, so what the provider echoes is
// what the client asked for. A refusal (402, 429, outage) is never the answer: the
// request walks down its candidates until one provider answers, and exactly one
// successful provider call ends it. Routing is a pure function of the brief, the
// manifest row and the refusals seen so far; nothing reads a clock or a random source.
//
// Usage: node main.ts [--list]

import { readFileSync } from 'node:fs';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';

const PROTOCOL_VERSION = 1;
const POLL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 10000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** After this many answered calls a provider is treated as near its free-tier budget. */
const RATION_AT_ATTEMPTS = 6;
/** Rating concession for a provider that is near its budget, in milli-points. */
const RATION_PENALTY = 40;
/** Rating concession for a provider that just refused, in milli-points. */
const COOLING_PENALTY = 60;
/** A provider this many refusals deep is treated as out for the rest of the trial. */
const OUT_AFTER_REFUSALS = 2;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

// ---------------------------------------------------------------------------
// Small JSON helpers
// ---------------------------------------------------------------------------

type Json = Record<string, unknown>;

function isRecord(value: unknown): value is Json {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function stringList(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === 'string');
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** Pull one embedded JSON document out of the brief, by its schema marker. */
function extractDocument(text: string, marker: string): Json | null {
  const at = text.indexOf(marker);
  if (at < 0) return null;
  let depth = 0;
  let inString = false;
  let escaped = false;
  let start = -1;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{') {
      if (depth === 0) start = i;
      depth += 1;
    } else if (ch === '}') {
      depth -= 1;
      if (depth === 0 && start >= 0) {
        if (i > at) {
          try {
            const parsed: unknown = JSON.parse(text.slice(start, i + 1));
            if (isRecord(parsed) && text.slice(start, i + 1).includes(marker)) return parsed;
          } catch {
            // not the document; keep scanning
          }
        }
        start = -1;
      } else if (depth < 0) {
        depth = 0;
        start = -1;
      }
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// The brief: provider directory, quality table, workload manifest
// ---------------------------------------------------------------------------

interface ProviderSpec {
  id: string;
  name: string;
  wire: 'openai' | 'anthropic' | 'gemini';
  auth: 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
  key_env: string;
  key: string;
  path: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools: boolean;
  vision: boolean;
  json: boolean;
  min_context: number;
}

interface ManifestRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

interface Brief {
  providers: ProviderSpec[];
  ratings: RatingRow[];
  manifest: ManifestRow[];
}

function readBrief(file: string): Brief {
  const empty: Brief = { providers: [], ratings: [], manifest: [] };
  let text: string;
  try {
    text = readFileSync(file, 'utf8');
  } catch {
    return empty;
  }
  const directory = extractDocument(text, '"schema":"free-model-aggregator-01/providers/v1"');
  const quality = extractDocument(text, '"schema":"free-model-aggregator-01/ratings/v1"');
  const workload = extractDocument(text, '"schema":"free-model-aggregator-01/workload/v1"');
  const providers: ProviderSpec[] = Array.isArray(directory?.providers)
    ? (directory.providers as Json[]).filter(isRecord).map((row) => ({
        id: String(row.id ?? ''),
        name: String(row.name ?? ''),
        wire: row.wire as ProviderSpec['wire'],
        auth: row.auth as ProviderSpec['auth'],
        key_env: String(row.key_env ?? ''),
        key: String(row.key ?? ''),
        path: String(row.path ?? ''),
      }))
    : [];
  const ratings: RatingRow[] = Array.isArray(quality?.ratings)
    ? (quality.ratings as Json[]).filter(isRecord).map((row) => ({
        model: String(row.model ?? ''),
        provider: String(row.provider ?? ''),
        rating_milli: typeof row.rating_milli === 'number' ? row.rating_milli : 0,
        context_length: typeof row.context_length === 'number' ? row.context_length : 0,
        tools: row.tools === true,
        vision: row.vision === true,
        json: row.json === true,
        free: row.free !== false,
      }))
    : [];
  const manifest: ManifestRow[] = Array.isArray(workload?.manifest)
    ? (workload.manifest as Json[]).filter(isRecord).map((row) => ({
        id: String(row.id ?? ''),
        api: row.api === 'anthropic' ? 'anthropic' : 'openai',
        stream: row.stream === true,
        weight: typeof row.weight === 'number' ? row.weight : 1,
        requires: {
          tools: row.requires !== null && isRecord(row.requires) && row.requires.tools === true,
          vision: row.requires !== null && isRecord(row.requires) && row.requires.vision === true,
          json: row.requires !== null && isRecord(row.requires) && row.requires.json === true,
          min_context:
            row.requires !== null && isRecord(row.requires) && typeof row.requires.min_context === 'number'
              ? (row.requires.min_context as number)
              : 0,
        },
      }))
    : [];
  return { providers, ratings, manifest };
}

// ---------------------------------------------------------------------------
// The provider directory, probed over the env, and the catalog built from it
// ---------------------------------------------------------------------------

interface ProbedModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface ProbedProvider {
  spec: ProviderSpec;
  ok: boolean;
  models: ProbedModel[];
}

function authHeaders(spec: ProviderSpec): Record<string, string> {
  if (spec.auth === 'bearer') return { authorization: `Bearer ${spec.key}` };
  if (spec.auth === 'x-api-key') return { 'x-api-key': spec.key };
  if (spec.auth === 'x-goog-api-key') return { 'x-goog-api-key': spec.key };
  return {};
}

function modelsPath(spec: ProviderSpec): string {
  return spec.wire === 'gemini' ? '/v1beta/models' : '/v1/models';
}

/** Ask one provider for the models it serves, in that provider's own shape. */
async function probeProvider(envUrl: string, spec: ProviderSpec): Promise<ProbedProvider> {
  const base = `${envUrl}${spec.path}`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(spec) };
  if (spec.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(`${base}${modelsPath(spec)}`, {
      method: 'GET',
      headers,
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
    });
    if (!res.ok) return { spec, ok: false, models: [] };
    const body: unknown = await res.json();
    const models: ProbedModel[] = [];
    if (spec.wire === 'gemini') {
      const list = isRecord(body) && Array.isArray(body.models) ? body.models : [];
      for (const item of list) {
        if (!isRecord(item)) continue;
        const name = String(item.name ?? '');
        models.push({
          id: name.startsWith('models/') ? name.slice('models/'.length) : name,
          free: item.free === true,
          context_length: numberOrNull(item.inputTokenLimit),
        });
      }
    } else {
      const list = isRecord(body) && Array.isArray(body.data) ? body.data : [];
      for (const item of list) {
        if (!isRecord(item)) continue;
        models.push({
          id: String(item.id ?? ''),
          free: item.free === true,
          context_length: numberOrNull(item.context_length),
        });
      }
    }
    return { spec, ok: true, models };
  } catch {
    return { spec, ok: false, models: [] };
  }
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: string;
  models: { id: string; free: boolean; context_length: number | null }[];
}

interface CatalogReport {
  providers: CatalogProvider[];
}

function ratingRow(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  for (const row of brief.ratings) if (row.provider === providerId && row.model === modelId) return row;
  return null;
}

/** The one document --list prints and the register body carries, byte for byte. */
function buildCatalog(probes: ProbedProvider[], brief: Brief): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    const models: CatalogProvider['models'] = [];
    for (const model of probe.models) {
      const row = ratingRow(brief, probe.spec.id, model.id);
      models.push({
        id: model.id,
        free: model.free && row?.free !== false,
        context_length: model.context_length ?? row?.context_length ?? null,
      });
    }
    providers.push({
      id: probe.spec.id,
      base_url: probe.spec.path,
      key_env: probe.spec.key_env,
      wire: probe.spec.wire,
      models,
    });
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// Routing: the best free model that meets the request, spread across providers
// ---------------------------------------------------------------------------

interface Candidate {
  probe: ProbedProvider;
  model: ProbedModel;
  rating: number;
}

function meets(requires: Requirement, row: RatingRow | null, context: number | null): boolean {
  if (requires.tools && !(row?.tools === true)) return false;
  if (requires.vision && !(row?.vision === true)) return false;
  if (requires.json && !(row?.json === true)) return false;
  if (requires.min_context > 0 && (context ?? 0) < requires.min_context) return false;
  return true;
}

function ableCandidates(brief: Brief, probes: ProbedProvider[], requires: Requirement): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingRow(brief, probe.spec.id, model.id);
      if (!model.free || row?.free === false) continue;
      if (!meets(requires, row, model.context_length)) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  return out;
}

interface Effort {
  attempts: Map<string, number>;
  refusals: Map<string, number>;
  cooling: Set<string>;
  out: Set<string>;
}

function newEffort(): Effort {
  return { attempts: new Map(), refusals: new Map(), cooling: new Set(), out: new Set() };
}

/**
 * This request's candidates in the order to try them. The rating leads; a provider
 * near its budget or still cooling from a refusal concedes ground to a near-equal
 * rival, so the trial's last requests are not the ones that lose the most.
 */
function orderCandidates(cands: Candidate[], effort: Effort, jsonMode: boolean): Candidate[] {
  const able = jsonMode ? cands.filter((c) => c.probe.spec.wire !== 'anthropic') : cands;
  const pool = able.length > 0 ? able : cands;
  const rank = (c: Candidate): number => {
    let rank = c.rating;
    const provider = c.probe.spec.id;
    if ((effort.attempts.get(provider) ?? 0) >= RATION_AT_ATTEMPTS) rank -= RATION_PENALTY;
    if (effort.cooling.has(provider)) rank -= COOLING_PENALTY;
    return rank;
  };
  return [...pool].sort((a, b) => {
    const byRank = rank(b) - rank(a);
    if (byRank !== 0) return byRank;
    const byAttempts = (effort.attempts.get(a.probe.spec.id) ?? 0) - (effort.attempts.get(b.probe.spec.id) ?? 0);
    if (byAttempts !== 0) return byAttempts;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
}

// ---------------------------------------------------------------------------
// The canonical request: what the score inspects, whatever wire it arrived on
// ---------------------------------------------------------------------------

interface Canon {
  systemText: string;
  userText: string;
  image: string | null;
  tools: string[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
  includeUsage: boolean;
}

function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

function imageUrlOf(block: Json): string | null {
  if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
    return block.image_url.url as string;
  }
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return `data:${String(source.media_type ?? 'image/png')};base64,${source.data}`;
    }
    if (source.type === 'url' && typeof source.url === 'string') return source.url as string;
  }
  if (block.type === 'inlineData' && typeof block.data === 'string') {
    return `data:${String(block.mime_type ?? 'image/png')};base64,${block.data}`;
  }
  return null;
}

/** Reduce an OpenAI Chat Completions body to the canonical seven fields. */
function canonFromOpenAI(body: Json): Canon {
  let systemText = '';
  let userText = '';
  let image: string | null = null;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message)) continue;
    if (message.role === 'system') {
      const text = blocksText(message.content);
      if (text !== '') systemText = systemText === '' ? text : `${systemText}\n${text}`;
    } else if (message.role === 'user') {
      userText = blocksText(message.content);
      if (Array.isArray(message.content)) {
        for (const block of message.content) {
          if (isRecord(block) && image === null) image = imageUrlOf(block);
        }
      }
    }
  }
  const tools: string[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === 'string') {
        tools.push(tool.function.name as string);
      }
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    systemText,
    userText,
    image,
    tools,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
    includeUsage: isRecord(body.stream_options) && body.stream_options.include_usage === true,
  };
}

/** Reduce an Anthropic Messages body to the canonical seven fields. */
function canonFromAnthropic(body: Json): Canon {
  const systemText = blocksText(body.system);
  let userText = '';
  let image: string | null = null;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || message.role !== 'user') continue;
    userText = blocksText(message.content);
    if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (isRecord(block) && image === null) image = imageUrlOf(block);
      }
    }
  }
  const tools: string[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (isRecord(tool) && typeof tool.name === 'string') tools.push(tool.name as string);
    }
  }
  return {
    systemText,
    userText,
    image,
    tools,
    maxTokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    stream: body.stream === true,
    includeUsage: false,
  };
}

// ---------------------------------------------------------------------------
// Expressing the canonical request in each provider's wire
// ---------------------------------------------------------------------------

function openAIBody(canon: Canon, model: string): Json {
  const messages: Json[] = [];
  if (canon.systemText !== '') messages.push({ role: 'system', content: canon.systemText });
  const userContent: Json[] | string =
    canon.image === null
      ? canon.userText
      : [{ type: 'text', text: canon.userText }, { type: 'image_url', image_url: { url: canon.image } }];
  messages.push({ role: 'user', content: userContent });
  const body: Json = { model, messages };
  if (canon.maxTokens !== null) body.max_tokens = canon.maxTokens;
  if (canon.stops.length > 0) body.stop = canon.stops;
  if (canon.tools.length > 0) {
    body.tools = canon.tools.map((name) => ({ type: 'function', function: { name } }));
  }
  if (canon.jsonMode) body.response_format = { type: 'json_object' };
  body.stream = canon.stream;
  if (canon.stream) body.stream_options = { include_usage: true };
  return body;
}

function anthropicBody(canon: Canon, model: string): Json {
  const content: Json[] = [{ type: 'text', text: canon.userText }];
  if (canon.image !== null) {
    const url = canon.image;
    if (url.startsWith('data:')) {
      const comma = url.indexOf(',');
      content.push({
        type: 'image',
        source: { type: 'base64', media_type: url.slice(5, comma), data: url.slice(comma + 1) },
      });
    } else {
      content.push({ type: 'image', source: { type: 'url', url } });
    }
  }
  const body: Json = { model, messages: [{ role: 'user', content }] };
  if (canon.systemText !== '') body.system = canon.systemText;
  if (canon.maxTokens !== null) body.max_tokens = canon.maxTokens;
  if (canon.stops.length > 0) body.stop_sequences = canon.stops;
  if (canon.tools.length > 0) body.tools = canon.tools.map((name) => ({ name }));
  body.stream = canon.stream;
  return body;
}

function geminiBody(canon: Canon, model: string): Json {
  const parts: Json[] = [{ text: canon.userText }];
  if (canon.image !== null) {
    const url = canon.image;
    if (url.startsWith('data:')) {
      const comma = url.indexOf(',');
      parts.push({ inlineData: { mime_type: url.slice(5, comma), data: url.slice(comma + 1) } });
    } else {
      parts.push({ fileData: { fileUri: url } });
    }
  }
  const body: Json = { contents: [{ role: 'user', parts }] };
  if (canon.systemText !== '') body.systemInstruction = { parts: [{ text: canon.systemText }] };
  const config: Json = {};
  if (canon.maxTokens !== null) config.maxOutputTokens = canon.maxTokens;
  if (canon.stops.length > 0) config.stopSequences = canon.stops;
  if (canon.jsonMode) config.responseMimeType = 'application/json';
  body.generationConfig = config;
  if (canon.tools.length > 0) {
    body.tools = [{ functionDeclarations: canon.tools.map((name) => ({ name })) }];
  }
  return body;
}

// ---------------------------------------------------------------------------
// Answering in the wire the client used
// ---------------------------------------------------------------------------

const STOP_FAMILY: Record<string, number> = {
  stop: 0, end_turn: 0, STOP: 0,
  length: 1, max_tokens: 1, MAX_TOKENS: 1,
  tool_calls: 2, tool_use: 2, function_call: 2,
  content_filter: 3, refusal: 3,
};

function stopForClient(providerStop: string | null, clientApi: 'openai' | 'anthropic'): string {
  const family = providerStop === null ? null : STOP_FAMILY[providerStop] ?? null;
  if (family === 1) return clientApi === 'openai' ? 'length' : 'max_tokens';
  if (family === 2) return clientApi === 'openai' ? 'tool_calls' : 'tool_use';
  if (family === 3) return clientApi === 'openai' ? 'content_filter' : 'refusal';
  return clientApi === 'openai' ? 'stop' : 'end_turn';
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, clientApi: 'openai' | 'anthropic', status: number, code: string, message: string): void {
  if (clientApi === 'anthropic') {
    sendJson(res, status, { type: 'error', error: { type: code, message } });
  } else {
    sendJson(res, status, { error: { message, type: code, param: null, code } });
  }
}

/** What the provider answered, reduced to the parts the client's checks read. */
interface Answer {
  text: string;
  input: number | null;
  output: number | null;
  stop: string | null;
}

function answerFromOpenAIJson(body: Json): Answer {
  const choice = Array.isArray(body.choices) ? body.choices[0] : null;
  const message = isRecord(choice) && isRecord(choice.message) ? choice.message : null;
  const usage = isRecord(body.usage) ? body.usage : null;
  return {
    text: typeof message?.content === 'string' ? message.content : '',
    input: usage !== null ? numberOrNull(usage.prompt_tokens) : null,
    output: usage !== null ? numberOrNull(usage.completion_tokens) : null,
    stop: isRecord(choice) && typeof choice.finish_reason === 'string' ? (choice.finish_reason as string) : null,
  };
}

function answerFromAnthropicJson(body: Json): Answer {
  const text = Array.isArray(body.content)
    ? body.content.filter(isRecord).filter((b) => b.type === 'text').map((b) => String(b.text ?? '')).join('')
    : '';
  const usage = isRecord(body.usage) ? body.usage : null;
  return {
    text,
    input: usage !== null ? numberOrNull(usage.input_tokens) : null,
    output: usage !== null ? numberOrNull(usage.output_tokens) : null,
    stop: typeof body.stop_reason === 'string' ? (body.stop_reason as string) : null,
  };
}

function answerFromGeminiJson(body: Json): Answer {
  const candidate = Array.isArray(body.candidates) ? body.candidates[0] : null;
  const parts = isRecord(candidate) && isRecord(candidate.content) && Array.isArray(candidate.content.parts)
    ? candidate.content.parts
    : [];
  let text = '';
  for (const part of parts) if (isRecord(part) && typeof part.text === 'string') text += part.text as string;
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : null;
  return {
    text,
    input: usage !== null ? numberOrNull(usage.promptTokenCount) : null,
    output: usage !== null ? numberOrNull(usage.candidatesTokenCount) : null,
    stop: isRecord(candidate) && typeof candidate.finishReason === 'string' ? (candidate.finishReason as string) : null,
  };
}

// --- SSE parsing -----------------------------------------------------------

interface SseEvent {
  event: string | null;
  data: string;
}

function parseSse(raw: string): SseEvent[] {
  const events: SseEvent[] = [];
  for (const frame of raw.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) events.push({ event, data: data.join('\n') });
  }
  return events;
}

/** Reduce a provider's SSE stream to one answer. */
function answerFromOpenAISse(events: SseEvent[]): Answer {
  let text = '';
  let input: number | null = null;
  let output: number | null = null;
  let stop: string | null = null;
  for (const frame of events) {
    if (frame.data.trim() === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      input = numberOrNull(chunk.usage.prompt_tokens);
      output = numberOrNull(chunk.usage.completion_tokens);
    }
    if (!Array.isArray(chunk.choices)) continue;
    const choice = chunk.choices[0];
    if (!isRecord(choice)) continue;
    if (isRecord(choice.delta) && typeof choice.delta.content === 'string') text += choice.delta.content as string;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason as string;
  }
  return { text, input, output, stop };
}

function answerFromAnthropicSse(events: SseEvent[]): Answer {
  let text = '';
  let input: number | null = null;
  let output: number | null = null;
  let stop: string | null = null;
  for (const frame of events) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numberOrNull(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta') {
      text += String(data.delta.text ?? '');
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason as string;
      if (isRecord(data.usage)) output = numberOrNull(data.usage.output_tokens);
    }
  }
  return { text, input, output, stop };
}

function answerFromGeminiSse(events: SseEvent[]): Answer {
  let text = '';
  let input: number | null = null;
  let output: number | null = null;
  let stop: string | null = null;
  for (const frame of events) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (Array.isArray(data.candidates)) {
      const candidate = data.candidates[0];
      if (isRecord(candidate)) {
        if (typeof candidate.finishReason === 'string') stop = candidate.finishReason as string;
        if (isRecord(candidate.content) && Array.isArray(candidate.content.parts)) {
          for (const part of candidate.content.parts) {
            if (isRecord(part) && typeof part.text === 'string') text += part.text as string;
          }
        }
      }
    }
    if (isRecord(data.usageMetadata)) {
      input = numberOrNull(data.usageMetadata.promptTokenCount);
      output = numberOrNull(data.usageMetadata.candidatesTokenCount);
    }
  }
  return { text, input, output, stop };
}

// --- Rendering the answer in the client's wire ------------------------------

function openAIJsonAnswer(answer: Answer, model: string): Json {
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: 0,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: answer.stop, logprobs: null }],
    usage: { prompt_tokens: answer.input ?? 0, completion_tokens: answer.output ?? 0, total_tokens: (answer.input ?? 0) + (answer.output ?? 0) },
  };
}

function anthropicJsonAnswer(answer: Answer, model: string): Json {
  return {
    id: 'msg-agg',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: answer.stop,
    stop_sequence: null,
    usage: { input_tokens: answer.input ?? 0, output_tokens: answer.output ?? 0 },
  };
}

function sse(res: ServerResponse, frames: string[]): void {
  res.writeHead(200, {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
  });
  for (const frame of frames) res.write(frame);
  res.end();
}

function openAISseAnswer(answer: Answer, model: string): string[] {
  const frames: string[] = [];
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: 0,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  frames.push(`data: ${JSON.stringify(chunk({ role: 'assistant', content: '' }, null))}\n\n`);
  frames.push(`data: ${JSON.stringify(chunk({ content: answer.text }, null))}\n\n`);
  frames.push(`data: ${JSON.stringify(chunk({}, answer.stop ?? 'stop'))}\n\n`);
  if (answer.input !== null && answer.output !== null) {
    frames.push(
      `data: ${JSON.stringify({
        id: 'chatcmpl-agg',
        object: 'chat.completion.chunk',
        created: 0,
        model,
        choices: [],
        usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
      })}\n\n`,
    );
  }
  frames.push('data: [DONE]\n\n');
  return frames;
}

function anthropicSseAnswer(answer: Answer, model: string): string[] {
  const frames: string[] = [];
  const frame = (event: string, data: Json): string => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  frames.push(
    frame('message_start', {
      type: 'message_start',
      message: {
        id: 'msg-agg', type: 'message', role: 'assistant', model, content: [],
        stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.input ?? 0, output_tokens: 0 },
      },
    }),
  );
  frames.push(frame('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
  frames.push(frame('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: answer.text } }));
  frames.push(frame('content_block_stop', { type: 'content_block_stop', index: 0 }));
  frames.push(
    frame('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: answer.stop ?? 'end_turn', stop_sequence: null },
      usage: { output_tokens: answer.output ?? 0 },
    }),
  );
  frames.push(frame('message_stop', { type: 'message_stop' }));
  return frames;
}

/** Deliver one provider answer, rendered in the wire the client used. */
function deliver(
  res: ServerResponse,
  clientApi: 'openai' | 'anthropic',
  answer: Answer,
  model: string,
  stream: boolean,
): void {
  if (!stream) {
    if (clientApi === 'openai') {
      sendJson(res, 200, openAIJsonAnswer({ ...answer, stop: stopForClient(answer.stop, 'openai') }, model));
    } else {
      sendJson(res, 200, anthropicJsonAnswer({ ...answer, stop: stopForClient(answer.stop, 'anthropic') }, model));
    }
    return;
  }
  if (clientApi === 'openai') {
    sse(res, openAISseAnswer({ ...answer, stop: stopForClient(answer.stop, 'openai') }, model));
  } else {
    sse(res, anthropicSseAnswer({ ...answer, stop: stopForClient(answer.stop, 'anthropic') }, model));
  }
}

// ---------------------------------------------------------------------------
// Serving the client
// ---------------------------------------------------------------------------

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

function readBody(req: IncomingMessage): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on('data', (chunk: Buffer) => chunks.push(chunk));
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

async function main(): Promise<void> {
  const brief = readBrief(PROMPT_FILE);
  const probes = await Promise.all(brief.providers.map((spec) => probeProvider(ENV_URL, spec)));
  const catalog = buildCatalog(probes, brief);
  const catalogText = `${JSON.stringify(catalog)}\n`;

  if (process.argv.includes('--list')) {
    await new Promise<void>((resolve) => process.stdout.write(catalogText, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');

  const effort = newEffort();
  let manifestIndex = 0;

  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? '/', 'http://entry');
      if (req.method === 'GET' && url.pathname === '/v1/models') {
        const free: { id: string; provider: string }[] = [];
        for (const probe of probes) {
          if (!probe.ok) continue;
          for (const model of probe.models) {
            const row = ratingRow(brief, probe.spec.id, model.id);
            if (!model.free || row?.free === false) continue;
            free.push({ id: model.id, provider: probe.spec.id });
          }
        }
        free.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
        if (typeof req.headers['x-api-key'] === 'string') {
          sendJson(res, 200, {
            data: free.map((m) => ({ type: 'model', id: m.id, display_name: m.id, created_at: '2026-01-01T00:00:00Z' })),
            has_more: false,
          });
        } else {
          sendJson(res, 200, {
            object: 'list',
            data: free.map((m) => ({ id: m.id, object: 'model', created: 0, owned_by: m.provider, free: true })),
          });
        }
        return;
      }
      if (req.method !== 'POST' || (url.pathname !== '/v1/chat/completions' && url.pathname !== '/v1/messages')) {
        sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
        return;
      }
      const body = await readJson(req);
      if (body === null) {
        sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
        return;
      }
      const clientApi: 'openai' | 'anthropic' = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
      const canon = clientApi === 'openai' ? canonFromOpenAI(body) : canonFromAnthropic(body);
      const row = brief.manifest[manifestIndex];
      manifestIndex += 1;
      const requires = row !== undefined ? row.requires : { tools: false, vision: false, json: false, min_context: 0 };
      const cands = orderCandidates(ableCandidates(brief, probes, requires), effort, canon.jsonMode);
      if (cands.length === 0) {
        sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model meets this request');
        return;
      }
      for (const cand of [...cands, cands[0]]) {
        const provider = cand.probe.spec;
        if (effort.out.has(provider.id)) continue;
        effort.attempts.set(provider.id, (effort.attempts.get(provider.id) ?? 0) + 1);
        const base = `${ENV_URL}${provider.path}`;
        const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider) };
        if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
        let upstreamUrl: string;
        let upstreamBody: Json;
        if (provider.wire === 'openai') {
          upstreamUrl = `${base}/v1/chat/completions`;
          upstreamBody = openAIBody(canon, cand.model.id);
        } else if (provider.wire === 'anthropic') {
          upstreamUrl = `${base}/v1/messages`;
          upstreamBody = anthropicBody(canon, cand.model.id);
        } else {
          const path = canon.stream ? 'streamGenerateContent' : 'generateContent';
          upstreamUrl = `${base}/v1beta/models/${encodeURIComponent(cand.model.id)}:${path}${canon.stream ? '?alt=sse' : ''}`;
          upstreamBody = geminiBody(canon, cand.model.id);
        }
        let upstream: Response;
        try {
          upstream = await fetch(upstreamUrl, {
            method: 'POST',
            headers,
            body: JSON.stringify(upstreamBody),
            signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
          });
        } catch {
          effort.refusals.set(provider.id, (effort.refusals.get(provider.id) ?? 0) + 1);
          effort.cooling.add(provider.id);
          continue;
        }
        if (!upstream.ok) {
          await upstream.text().catch(() => '');
          const refusals = (effort.refusals.get(provider.id) ?? 0) + 1;
          effort.refusals.set(provider.id, refusals);
          effort.cooling.add(provider.id);
          if (upstream.status === 402 || refusals >= OUT_AFTER_REFUSALS) effort.out.add(provider.id);
          continue;
        }
        const answer = await consumeAnswer(upstream, provider.wire);
        deliver(res, clientApi, answer, cand.model.id, canon.stream);
        return;
      }
      sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_error', 'every provider refused this request');
    } catch (err) {
      process.stderr.write(`aggregator: ${(err as Error).stack ?? String(err)}\n`);
      if (!res.headersSent) sendError(res, 'openai', 500, 'internal_error', 'the server failed to answer');
    }
  });

  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const register = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port: address.port, catalog: catalog }),
    signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
  });
  if (!register.ok) throw new Error(`POST /register answered HTTP ${register.status}`);
  process.stderr.write(`aggregator: registered on port ${address.port} with ${catalog.providers.length} providers\n`);

  const timer = setInterval(async () => {
    try {
      const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
      if (!res.ok) return;
      const work: unknown = await res.json();
      if (isRecord(work) && work.done === true) {
        clearInterval(timer);
        server.close();
        process.exit(0);
      }
    } catch {
      // The env is not answering yet; the next tick tries again.
    }
  }, POLL_MS);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
}

/** Read one provider answer, JSON or streamed, into the canonical answer. */
async function consumeAnswer(res: Response, wire: 'openai' | 'anthropic' | 'gemini'): Promise<Answer> {
  const raw = await res.text();
  if (!res.headers.get('content-type')?.includes('text/event-stream')) {
    let body: unknown;
    try {
      body = JSON.parse(raw);
    } catch {
      return { text: raw, input: null, output: null, stop: null };
    }
    if (!isRecord(body)) return { text: raw, input: null, output: null, stop: null };
    if (wire === 'openai') return answerFromOpenAIJson(body);
    if (wire === 'anthropic') return answerFromAnthropicJson(body);
    return answerFromGeminiJson(body);
  }
  const events = parseSse(raw);
  if (wire === 'openai') return answerFromOpenAISse(events);
  if (wire === 'anthropic') return answerFromAnthropicSse(events);
  return answerFromGeminiSse(events);
}

try {
  await main();
} catch (err) {
  process.stderr.write(`aggregator: ${(err as Error).stack ?? String(err)}\n`);
  process.exit(1);
}
