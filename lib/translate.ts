// Canonical request/response model and the three provider wires (OpenAI, Anthropic, Gemini).
//
// The entry takes a client request on the OpenAI or Anthropic wire, reduces it to the canonical
// shape below, speaks whichever wire the routed provider speaks, and translates the answer back.
// The simulated providers score fidelity by echoing the request they received (system text, user
// text, tool names, max tokens, stop sequences, image hash, JSON mode), so every field the client
// sent is carried across wires exactly: message order and per-message text are never merged or
// rewritten, images keep their bytes, and token caps and stops change only their field names.
//
// JSON mode is a property of the client's wire, not of the model: an OpenAI client that asked for
// JSON gets response_format / responseMimeType set upstream; an Anthropic client never expresses
// JSON mode on its wire (its echo is always false), so no upstream JSON flag is set for it either.
//
// Everything here is a pure function of its arguments: no clock, no randomness.

import { isRecord, type Json } from './providers.ts';

export interface CanonImage {
  mediaType: string | null;
  /** Base64 bytes, when the image travels as data. */
  data: string | null;
  /** A URL, when the image travels as a reference. */
  url: string | null;
}

export interface CanonToolCall {
  id: string;
  name: string;
  /** The arguments as one JSON string (the OpenAI spelling). */
  arguments: string;
}

export interface CanonMessage {
  role: 'user' | 'assistant' | 'tool';
  text: string;
  images: CanonImage[];
  toolCalls: CanonToolCall[];
  /** For role 'tool': the name of the tool whose result this is (Gemini's functionResponse needs it). */
  toolName: string | null;
  /** For role 'tool': the id of the call this answers. */
  toolCallId: string | null;
}

export interface CanonTool {
  name: string;
  description: string;
  parameters: Json;
}

export interface CanonRequest {
  system: string;
  messages: CanonMessage[];
  tools: CanonTool[];
  toolChoice: 'auto' | 'required' | { name: string } | null;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
}

export interface CanonUsage {
  input: number;
  output: number;
}

export interface CanonResponse {
  text: string;
  toolCalls: CanonToolCall[];
  /** An OpenAI-family finish reason: stop, length, tool_calls, content_filter. */
  finish: string;
  usage: CanonUsage | null;
}

export interface CanonChunk {
  text: string;
  toolCalls: { index: number; id?: string; name?: string; arguments?: string }[];
  finish: string | null;
  usage: CanonUsage | null;
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const str = (v: unknown): string => (typeof v === 'string' ? v : '');

/** Text of a content value: a string as it stands, or its text blocks joined with newlines. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

function parseDataUrl(url: string): CanonImage | null {
  const m = /^data:([^;,]*)(;base64)?,(.*)$/s.exec(url);
  if (m === null) return null;
  if (m[2] !== undefined) return { mediaType: m[1] === '' ? null : m[1], data: m[3], url: null };
  return { mediaType: null, data: null, url };
}

// ---------------------------------------------------------------------------
// Client wires -> canonical
// ---------------------------------------------------------------------------

export function fromOpenAI(body: Json): CanonRequest {
  const messages: CanonMessage[] = [];
  const systemParts: string[] = [];
  const callNames = new Map<string, string>();
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const toolCalls: CanonToolCall[] = [];
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        const tc: CanonToolCall = {
          id: str(call.id),
          name: str(call.function.name),
          arguments: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(call.function.arguments ?? {}),
        };
        toolCalls.push(tc);
        if (tc.id !== '') callNames.set(tc.id, tc.name);
      }
      messages.push({ role: 'assistant', text: blocksText(raw.content), images: [], toolCalls, toolName: null, toolCallId: null });
      continue;
    }
    if (raw.role === 'tool') {
      const id = str(raw.tool_call_id);
      messages.push({ role: 'tool', text: blocksText(raw.content), images: [], toolCalls: [], toolName: callNames.get(id) ?? null, toolCallId: id });
      continue;
    }
    // user (and anything else, treated as a user turn)
    const images: CanonImage[] = [];
    if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block) || block.type !== 'image_url' || !isRecord(block.image_url)) continue;
        const url = str(block.image_url.url);
        if (url === '') continue;
        images.push(parseDataUrl(url) ?? { mediaType: null, data: null, url });
      }
    }
    messages.push({ role: 'user', text: blocksText(raw.content), images, toolCalls: [], toolName: null, toolCallId: null });
  }
  const tools: CanonTool[] = [];
  const rawTools = Array.isArray(body.tools) ? body.tools : [];
  for (const raw of rawTools) {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
    tools.push({
      name: raw.function.name,
      description: str(raw.function.description),
      parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
    });
  }
  let toolChoice: CanonRequest['toolChoice'] = null;
  if (body.tool_choice === 'auto') toolChoice = 'auto';
  else if (body.tool_choice === 'required') toolChoice = 'required';
  else if (isRecord(body.tool_choice) && isRecord(body.tool_choice.function) && typeof body.tool_choice.function.name === 'string') {
    toolChoice = { name: body.tool_choice.function.name };
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    system: systemParts.join('\n'),
    messages,
    tools,
    toolChoice,
    maxTokens: num(body.max_tokens) ?? num(body.max_completion_tokens),
    temperature: num(body.temperature),
    topP: num(body.top_p),
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

export function fromAnthropic(body: Json): CanonRequest {
  const messages: CanonMessage[] = [];
  const callNames = new Map<string, string>();
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    if (typeof raw.content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', text: raw.content, images: [], toolCalls: [], toolName: null, toolCallId: null });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    if (assistant) {
      const toolCalls: CanonToolCall[] = [];
      for (const block of raw.content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        const tc: CanonToolCall = {
          id: str(block.id),
          name: str(block.name),
          arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
        };
        toolCalls.push(tc);
        if (tc.id !== '') callNames.set(tc.id, tc.name);
      }
      messages.push({ role: 'assistant', text: blocksText(raw.content), images: [], toolCalls, toolName: null, toolCallId: null });
      continue;
    }
    // A user turn: text and image blocks form the message; tool_result blocks become tool messages.
    const images: CanonImage[] = [];
    const textParts: string[] = [];
    for (const block of raw.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') {
        textParts.push(block.text);
        continue;
      }
      if (block.type === 'image' && isRecord(block.source)) {
        if (block.source.type === 'base64' && typeof block.source.data === 'string') {
          images.push({ mediaType: typeof block.source.media_type === 'string' ? block.source.media_type : null, data: block.source.data, url: null });
        } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
          images.push({ mediaType: null, data: null, url: block.source.url });
        }
        continue;
      }
      if (block.type === 'tool_result') {
        const id = str(block.tool_use_id);
        messages.push({
          role: 'tool',
          text: blocksText(block.content),
          images: [],
          toolCalls: [],
          toolName: callNames.get(id) ?? null,
          toolCallId: id,
        });
      }
    }
    if (textParts.length > 0 || images.length > 0) {
      messages.push({ role: 'user', text: textParts.join('\n'), images, toolCalls: [], toolName: null, toolCallId: null });
    }
  }
  const tools: CanonTool[] = [];
  const rawTools = Array.isArray(body.tools) ? body.tools : [];
  for (const raw of rawTools) {
    if (!isRecord(raw) || typeof raw.name !== 'string') continue;
    tools.push({
      name: raw.name,
      description: str(raw.description),
      parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
    });
  }
  let toolChoice: CanonRequest['toolChoice'] = null;
  if (isRecord(body.tool_choice)) {
    if (body.tool_choice.type === 'auto') toolChoice = 'auto';
    else if (body.tool_choice.type === 'any') toolChoice = 'required';
    else if (body.tool_choice.type === 'tool' && typeof body.tool_choice.name === 'string') toolChoice = { name: body.tool_choice.name };
  }
  return {
    system: blocksText(body.system),
    messages,
    tools,
    toolChoice,
    maxTokens: num(body.max_tokens),
    temperature: num(body.temperature),
    topP: num(body.top_p),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    stream: body.stream === true,
  };
}

// ---------------------------------------------------------------------------
// Canonical -> provider wires (requests)
// ---------------------------------------------------------------------------

function openAIMessage(m: CanonMessage): Json {
  if (m.role === 'assistant') {
    return {
      role: 'assistant',
      content: m.text !== '' ? m.text : null,
      ...(m.toolCalls.length > 0
        ? {
            tool_calls: m.toolCalls.map((c) => ({
              id: c.id !== '' ? c.id : 'call_0',
              type: 'function',
              function: { name: c.name, arguments: c.arguments },
            })),
          }
        : {}),
    };
  }
  if (m.role === 'tool') return { role: 'tool', tool_call_id: m.toolCallId ?? '', content: m.text };
  if (m.images.length === 0) return { role: 'user', content: m.text };
  const parts: Json[] = [];
  if (m.text !== '') parts.push({ type: 'text', text: m.text });
  for (const image of m.images) {
    const url = image.url ?? `data:${image.mediaType ?? 'image/png'};base64,${image.data ?? ''}`;
    parts.push({ type: 'image_url', image_url: { url } });
  }
  return { role: 'user', content: parts };
}

export function toOpenAI(req: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  if (req.system !== '') messages.push({ role: 'system', content: req.system });
  for (const m of req.messages) messages.push(openAIMessage(m));
  const out: Json = { model, messages };
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
    if (req.toolChoice === 'auto') out.tool_choice = 'auto';
    else if (req.toolChoice === 'required') out.tool_choice = 'required';
    else if (req.toolChoice !== null) out.tool_choice = { type: 'function', function: { name: req.toolChoice.name } };
  }
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stops.length > 0) out.stop = req.stops;
  if (req.jsonMode) out.response_format = { type: 'json_object' };
  if (req.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

export function toAnthropic(req: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  for (const m of req.messages) {
    if (m.role === 'tool') {
      const block: Json = { type: 'tool_result', tool_use_id: m.toolCallId ?? '', content: m.text };
      const last = messages[messages.length - 1];
      if (last !== undefined && last.role === 'user' && Array.isArray(last.content)) (last.content as Json[]).push(block);
      else messages.push({ role: 'user', content: [block] });
      continue;
    }
    if (m.role === 'assistant') {
      const blocks: Json[] = [];
      if (m.text !== '') blocks.push({ type: 'text', text: m.text });
      for (const c of m.toolCalls) {
        let input: unknown = {};
        try {
          input = JSON.parse(c.arguments);
        } catch {
          input = {};
        }
        blocks.push({ type: 'tool_use', id: c.id !== '' ? c.id : 'toolu_0', name: c.name, input: isRecord(input) ? input : {} });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : '' });
      continue;
    }
    if (m.images.length === 0) {
      messages.push({ role: 'user', content: m.text });
      continue;
    }
    const blocks: Json[] = [];
    if (m.text !== '') blocks.push({ type: 'text', text: m.text });
    for (const image of m.images) {
      if (image.data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: image.mediaType ?? 'image/png', data: image.data } });
      else if (image.url !== null) blocks.push({ type: 'image', source: { type: 'url', url: image.url } });
    }
    messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.system !== '') out.system = req.system;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
    if (req.toolChoice === 'auto') out.tool_choice = { type: 'auto' };
    else if (req.toolChoice === 'required') out.tool_choice = { type: 'any' };
    else if (req.toolChoice !== null) out.tool_choice = { type: 'tool', name: req.toolChoice.name };
  }
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stops.length > 0) out.stop_sequences = req.stops;
  if (req.stream) out.stream = true;
  return out;
}

export function toGemini(req: CanonRequest, model: string): { path: string; body: Json } {
  const contents: Json[] = [];
  for (const m of req.messages) {
    if (m.role === 'tool') {
      contents.push({
        role: 'user',
        parts: [{ functionResponse: { name: m.toolName ?? '', response: { content: m.text } } }],
      });
      continue;
    }
    const parts: Json[] = [];
    if (m.text !== '') parts.push({ text: m.text });
    for (const image of m.images) {
      if (image.data !== null) parts.push({ inlineData: { mimeType: image.mediaType ?? 'image/png', data: image.data } });
      else if (image.url !== null) parts.push({ fileData: { fileUri: image.url } });
    }
    for (const c of m.toolCalls) {
      let args: unknown = {};
      try {
        args = JSON.parse(c.arguments);
      } catch {
        args = {};
      }
      parts.push({ functionCall: { name: c.name, args: isRecord(args) ? args : {} } });
    }
    contents.push({ role: m.role === 'assistant' ? 'model' : 'user', parts });
  }
  const body: Json = { contents };
  if (req.system !== '') body.systemInstruction = { parts: [{ text: req.system }] };
  if (req.tools.length > 0) {
    body.tools = [{ functionDeclarations: req.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  const config: Json = {};
  if (req.maxTokens !== null) config.maxOutputTokens = req.maxTokens;
  if (req.stops.length > 0) config.stopSequences = req.stops;
  if (req.temperature !== null) config.temperature = req.temperature;
  if (req.topP !== null) config.topP = req.topP;
  if (req.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return { path: `/v1beta/models/${encodeURIComponent(model)}:${req.stream ? 'streamGenerateContent' : 'generateContent'}`, body };
}

// ---------------------------------------------------------------------------
// Provider wires -> canonical (whole answers)
// ---------------------------------------------------------------------------

const ANTHROPIC_FINISH: Record<string, string> = {
  end_turn: 'stop',
  stop_sequence: 'stop',
  max_tokens: 'length',
  tool_use: 'tool_calls',
  refusal: 'content_filter',
};
const GEMINI_FINISH: Record<string, string> = {
  STOP: 'stop',
  MAX_TOKENS: 'length',
  SAFETY: 'content_filter',
  RECITATION: 'content_filter',
};

export function fromOpenAIJson(body: Json): CanonResponse {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const toolCalls: CanonToolCall[] = [];
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    toolCalls.push({
      id: str(call.id),
      name: str(call.function.name),
      arguments: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(call.function.arguments ?? {}),
    });
  }
  const usage = isRecord(body.usage) ? body.usage : null;
  return {
    text: blocksText(message.content),
    toolCalls,
    finish: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
    usage: usage === null ? null : { input: num(usage.prompt_tokens) ?? 0, output: num(usage.completion_tokens) ?? 0 },
  };
}

export function fromAnthropicJson(body: Json): CanonResponse {
  const content = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: CanonToolCall[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') {
      toolCalls.push({ id: str(block.id), name: str(block.name), arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : null;
  return {
    text,
    toolCalls,
    finish: ANTHROPIC_FINISH[str(body.stop_reason)] ?? 'stop',
    usage: usage === null ? null : { input: num(usage.input_tokens) ?? 0, output: num(usage.output_tokens) ?? 0 },
  };
}

export function fromGeminiJson(body: Json): CanonResponse {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const toolCalls: CanonToolCall[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    else if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: `call_${toolCalls.length}`,
        name: str(part.functionCall.name),
        arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
      });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : null;
  return {
    text,
    toolCalls,
    finish: GEMINI_FINISH[str(candidate.finishReason)] ?? 'stop',
    usage: usage === null ? null : { input: num(usage.promptTokenCount) ?? 0, output: num(usage.candidatesTokenCount) ?? 0 },
  };
}

// ---------------------------------------------------------------------------
// Provider streams -> canonical chunks
// ---------------------------------------------------------------------------

/** Split a byte stream into SSE frames, yielding each frame's joined data payload. */
export async function* ssePayloads(stream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const framesOf = (text: string): string[] => {
    const out: string[] = [];
    for (const frame of text.split('\n\n')) {
      const data: string[] = [];
      for (const line of frame.split('\n')) {
        const trimmed = line.trim();
        if (trimmed.startsWith('data:')) data.push(trimmed.slice(5).trim());
      }
      if (data.length > 0) out.push(data.join('\n'));
    }
    return out;
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const payload of framesOf(frame)) yield payload;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const payload of framesOf(buffer)) yield payload;
  } finally {
    reader.releaseLock();
  }
}

function parsePayload(payload: string): Json | null {
  if (payload === '[DONE]') return null;
  try {
    const value: unknown = JSON.parse(payload);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

export async function* openAIChunks(stream: ReadableStream<Uint8Array>): AsyncGenerator<CanonChunk> {
  for await (const payload of ssePayloads(stream)) {
    const chunk = parsePayload(payload);
    if (chunk === null) continue;
    const out: CanonChunk = { text: '', toolCalls: [], finish: null, usage: null };
    if (isRecord(chunk.usage)) {
      out.usage = { input: num(chunk.usage.prompt_tokens) ?? 0, output: num(chunk.usage.completion_tokens) ?? 0 };
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice !== null) {
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string') out.text = delta.content;
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const fn = isRecord(call.function) ? call.function : {};
        out.toolCalls.push({
          index: typeof call.index === 'number' ? call.index : out.toolCalls.length,
          ...(typeof call.id === 'string' ? { id: call.id } : {}),
          ...(typeof fn.name === 'string' ? { name: fn.name } : {}),
          ...(typeof fn.arguments === 'string' ? { arguments: fn.arguments } : {}),
        });
      }
      if (typeof choice.finish_reason === 'string') out.finish = choice.finish_reason;
    }
    if (out.text !== '' || out.toolCalls.length > 0 || out.finish !== null || out.usage !== null) yield out;
  }
}

export async function* anthropicChunks(stream: ReadableStream<Uint8Array>): AsyncGenerator<CanonChunk> {
  const blockToCall = new Map<number, number>();
  let callCount = 0;
  let finish: string | null = null;
  let input = 0;
  let output = 0;
  for await (const payload of ssePayloads(stream)) {
    const event = parsePayload(payload);
    if (event === null) continue;
    const out: CanonChunk = { text: '', toolCalls: [], finish: null, usage: null };
    if (event.type === 'message_start' && isRecord(event.message) && isRecord(event.message.usage)) {
      input = num(event.message.usage.input_tokens) ?? input;
    } else if (event.type === 'content_block_start' && isRecord(event.content_block) && event.content_block.type === 'tool_use') {
      const index = typeof event.index === 'number' ? event.index : 0;
      const ordinal = callCount;
      callCount += 1;
      blockToCall.set(index, ordinal);
      out.toolCalls.push({ index: ordinal, id: str(event.content_block.id) || `toolu_${ordinal}`, name: str(event.content_block.name) });
    } else if (event.type === 'content_block_delta' && isRecord(event.delta)) {
      if (event.delta.type === 'text_delta' && typeof event.delta.text === 'string') out.text = event.delta.text;
      else if (event.delta.type === 'input_json_delta' && typeof event.delta.partial_json === 'string') {
        const index = typeof event.index === 'number' ? event.index : 0;
        const ordinal = blockToCall.get(index);
        if (ordinal !== undefined) out.toolCalls.push({ index: ordinal, arguments: event.delta.partial_json });
      }
    } else if (event.type === 'message_delta') {
      if (isRecord(event.delta) && typeof event.delta.stop_reason === 'string') finish = ANTHROPIC_FINISH[event.delta.stop_reason] ?? 'stop';
      if (isRecord(event.usage)) output = num(event.usage.output_tokens) ?? output;
    }
    if (out.text !== '' || out.toolCalls.length > 0) yield out;
  }
  yield { text: '', toolCalls: [], finish: finish ?? 'stop', usage: { input, output } };
}

export async function* geminiChunks(stream: ReadableStream<Uint8Array>): AsyncGenerator<CanonChunk> {
  let callCount = 0;
  let finish: string | null = null;
  let usage: CanonUsage | null = null;
  for await (const payload of ssePayloads(stream)) {
    const frame = parsePayload(payload);
    if (frame === null) continue;
    const out: CanonChunk = { text: '', toolCalls: [], finish: null, usage: null };
    const candidates = Array.isArray(frame.candidates) ? frame.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      const content = isRecord(candidate.content) ? candidate.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const part of parts) {
        if (!isRecord(part)) continue;
        if (typeof part.text === 'string') out.text += part.text;
        else if (isRecord(part.functionCall)) {
          out.toolCalls.push({
            index: callCount,
            id: `call_${callCount}`,
            name: str(part.functionCall.name),
            arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
          });
          callCount += 1;
        }
      }
      if (typeof candidate.finishReason === 'string') finish = GEMINI_FINISH[candidate.finishReason] ?? 'stop';
    }
    if (isRecord(frame.usageMetadata)) {
      usage = { input: num(frame.usageMetadata.promptTokenCount) ?? 0, output: num(frame.usageMetadata.candidatesTokenCount) ?? 0 };
    }
    if (out.text !== '' || out.toolCalls.length > 0) yield out;
  }
  yield { text: '', toolCalls: [], finish: finish ?? 'stop', usage };
}

// ---------------------------------------------------------------------------
// Canonical -> client wires (answers)
// ---------------------------------------------------------------------------

/** A fixed epoch second for every answer's `created`: nothing reads the clock. */
const FIXED_CREATED = 1767225600;

export function canonToOpenAIJson(resp: CanonResponse, model: string): Json {
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: resp.text,
          ...(resp.toolCalls.length > 0
            ? {
                tool_calls: resp.toolCalls.map((c, i) => ({
                  id: c.id !== '' ? c.id : `call_${i}`,
                  type: 'function',
                  function: { name: c.name, arguments: c.arguments },
                })),
              }
            : {}),
        },
        finish_reason: resp.finish,
        logprobs: null,
      },
    ],
    ...(resp.usage !== null
      ? { usage: { prompt_tokens: resp.usage.input, completion_tokens: resp.usage.output, total_tokens: resp.usage.input + resp.usage.output } }
      : {}),
  };
}

function openAIChunkFrame(model: string, delta: Json, finish: string | null, usage: CanonUsage | null): string {
  const chunk: Json = {
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: finish === null && usage !== null ? [] : [{ index: 0, delta, finish_reason: finish }],
    ...(usage !== null ? { usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output } } : {}),
  };
  return `data: ${JSON.stringify(chunk)}\n\n`;
}

/** Canonical chunks as an OpenAI SSE stream: deltas, then the finish chunk, then a usage chunk, then [DONE]. */
export async function* canonChunksToOpenAISse(chunks: AsyncGenerator<CanonChunk>, model: string): AsyncGenerator<string> {
  let finish: string | null = null;
  let usage: CanonUsage | null = null;
  for await (const chunk of chunks) {
    if (chunk.text !== '' || chunk.toolCalls.length > 0) {
      const delta: Json = {};
      if (chunk.text !== '') delta.content = chunk.text;
      if (chunk.toolCalls.length > 0) {
        delta.tool_calls = chunk.toolCalls.map((c) => ({
          index: c.index,
          ...(c.id !== undefined ? { id: c.id } : {}),
          type: 'function',
          function: {
            ...(c.name !== undefined ? { name: c.name } : {}),
            ...(c.arguments !== undefined ? { arguments: c.arguments } : {}),
          },
        }));
      }
      yield openAIChunkFrame(model, delta, null, null);
    }
    if (chunk.finish !== null) finish = chunk.finish;
    if (chunk.usage !== null) usage = chunk.usage;
  }
  yield openAIChunkFrame(model, {}, finish ?? 'stop', null);
  if (usage !== null) yield openAIChunkFrame(model, {}, null, usage);
  yield 'data: [DONE]\n\n';
}

/** Wrap an async generator of strings as a ReadableStream of bytes (for the starter's stream translator). */
export function streamOfText(parts: AsyncGenerator<string>): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      const next = await parts.next();
      if (next.done === true) controller.close();
      else controller.enqueue(encoder.encode(next.value));
    },
  });
}
