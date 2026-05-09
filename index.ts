import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import {
  calculateCost,
  createAssistantMessageEventStream,
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Context,
  type ImageContent,
  type Message,
  type Model,
  type SimpleStreamOptions,
  type StopReason,
  type TextContent,
  type ThinkingContent,
  type Tool,
  type ToolCall,
  type ToolResultMessage,
  type Usage,
} from "@mariozechner/pi-ai";

type OllamaTag = {
  name: string;
  model?: string;
  modified_at?: string;
  size?: number;
  digest?: string;
  details?: {
    parent_model?: string;
    format?: string;
    family?: string;
    families?: string[];
    parameter_size?: string;
    quantization_level?: string;
  };
};

type OllamaTagsResponse = { models?: OllamaTag[] };

type OllamaShowResponse = {
  parameters?: string;
  license?: string;
  template?: string;
  modified_at?: string;
  capabilities?: string[];
  details?: {
    parent_model?: string;
    format?: string;
    family?: string;
    families?: string[];
    parameter_size?: string;
    quantization_level?: string;
  };
  model_info?: Record<string, unknown>;
};

type CatalogEntry = {
  tag: OllamaTag;
  show?: OllamaShowResponse;
};

type CapabilityFlags = {
  thinking: boolean;
  tool: boolean;
  vision: boolean;
  embeddings: boolean;
  chat: boolean;
};

type ModelCatalog = {
  entries: CatalogEntry[];
  models: ReturnType<typeof toProviderModel>[];
};

declare const process: {
  env: Record<string, string | undefined>;
};

const DEFAULT_BASE_URL = "https://ollama.com/api";
const PROVIDER_NAME = "ollama";
const API_NAME = "ollama-native";
const API_KEY_ENV = "OLLAMA_API_KEY";
const REQUEST_TIMEOUT_MS = 30_000;

let currentBaseUrl = normalizeBaseUrl(process.env.OLLAMA_BASE_URL ?? DEFAULT_BASE_URL);
let currentCatalog: ModelCatalog = { entries: [], models: [] };

function normalizeBaseUrl(value: string): string {
  return value.replace(/\/+$/, "");
}

function apiUrl(path: string, baseUrl = currentBaseUrl): string {
  return `${normalizeBaseUrl(baseUrl)}${path.startsWith("/") ? path : `/${path}`}`;
}

function headers(apiKey?: string, extra?: Record<string, string>): Record<string, string> {
  const result: Record<string, string> = {
    accept: "application/json",
    "content-type": "application/json",
    ...extra,
  };
  if (apiKey) result.authorization = `Bearer ${apiKey}`;
  return result;
}

function isEmbeddingModel(tag: OllamaTag, show?: OllamaShowResponse): boolean {
  const name = `${tag.name} ${tag.model ?? ""} ${show?.details?.family ?? ""}`.toLowerCase();
  return /(^|[-_/])(embed|embedding|nomic-embed|bge-|mxbai-embed|text-embedding)/.test(name);
}

function isVisionModel(tag: OllamaTag, show?: OllamaShowResponse): boolean {
  const capabilities = show?.capabilities ?? [];
  if (capabilities.includes("vision")) return true;
  const name = `${tag.name} ${tag.model ?? ""} ${show?.details?.family ?? ""} ${show?.details?.families?.join(" ") ?? ""}`.toLowerCase();
  return /(^|[-_/])(llava|minicpm-v|qwen3-vl|qwen2-vl|gemma3|phi4-multimodal|moondream|pixtral|internvl|aya-vision|granite-vision|deepseek-vl|mistral-small-vision)/.test(name);
}

function isThinkingModel(tag: OllamaTag, show?: OllamaShowResponse): boolean {
  const name = `${tag.name} ${tag.model ?? ""} ${show?.details?.family ?? ""} ${show?.details?.families?.join(" ") ?? ""}`.toLowerCase();
  return /(^|[-_/])(qwen3|gpt-oss|deepseek-r1|deepseek-v3\.1|deepseek-r1-distill|qwq|gemma3n|phi4-reasoning|o1|r1)/.test(name);
}

function isToolCapable(tag: OllamaTag, show?: OllamaShowResponse): boolean {
  if (isEmbeddingModel(tag, show)) return false;
  if ((show?.capabilities ?? []).includes("completion")) return true;
  return true;
}

function inferContextWindow(show?: OllamaShowResponse): number {
  const info = show?.model_info ?? {};
  for (const [key, value] of Object.entries(info)) {
    if (key.endsWith(".context_length") && typeof value === "number" && Number.isFinite(value)) return value;
    if (key === "general.context_length" && typeof value === "number" && Number.isFinite(value)) return value;
  }

  const params = show?.parameters ?? "";
  const match = params.match(/num_ctx\s+(\d+)/i);
  if (match) return Number(match[1]);

  return 128000;
}

function inferMaxTokens(contextWindow: number, tag: OllamaTag, show?: OllamaShowResponse): number {
  if (isEmbeddingModel(tag, show)) return 0;
  if (isThinkingModel(tag, show)) return Math.max(4096, Math.min(32768, Math.floor(contextWindow / 4)));
  return Math.max(2048, Math.min(16384, Math.floor(contextWindow / 8)));
}

function thinkingLevelMap(tag: OllamaTag, show?: OllamaShowResponse) {
  const name = `${tag.name} ${tag.model ?? ""} ${show?.details?.family ?? ""} ${show?.details?.families?.join(" ") ?? ""}`.toLowerCase();
  if (/(^|[-_/])deepseek-v4($|[-_/])/.test(name)) {
    return {
      xhigh: "max",
    };
  }
  return undefined;
}

function capabilityFlags(tag: OllamaTag, show?: OllamaShowResponse): CapabilityFlags {
  const embeddings = isEmbeddingModel(tag, show);
  const vision = !embeddings && isVisionModel(tag, show);
  const thinking = !embeddings && isThinkingModel(tag, show);
  const tool = isToolCapable(tag, show);
  return { embeddings, vision, thinking, tool, chat: !embeddings };
}

function capabilityLabel(flags: CapabilityFlags): string {
  const bits: string[] = [];
  if (flags.thinking) bits.push("thinking");
  if (flags.tool) bits.push("tools");
  if (flags.vision) bits.push("vision");
  if (flags.embeddings) bits.push("embeddings");
  return bits.length ? ` · ${bits.join(" · ")}` : "";
}

function displayName(tag: OllamaTag, show?: OllamaShowResponse): string {
  return show?.details?.family || tag.model || tag.name || tag.details?.family || tag.name;
}

function toProviderModel(entry: CatalogEntry) {
  const contextWindow = inferContextWindow(entry.show);
  const flags = capabilityFlags(entry.tag, entry.show);
  const name = `${displayName(entry.tag, entry.show)} · ${contextWindow.toLocaleString()} ctx${capabilityLabel(flags)}`;

  return {
    id: entry.tag.name,
    name,
    reasoning: flags.thinking,
    thinkingLevelMap: thinkingLevelMap(entry.tag, entry.show),
    input: flags.vision ? (["text", "image"] as const) : (["text"] as const),
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow,
    maxTokens: inferMaxTokens(contextWindow, entry.tag, entry.show),
  };
}

function resolveApiKey(explicit?: string): string | undefined {
  return explicit || process.env[API_KEY_ENV] || undefined;
}

async function fetchJson<T>(path: string, init?: RequestInit, baseUrl = currentBaseUrl, apiKey?: string): Promise<T> {
  const response = await fetch(apiUrl(path, baseUrl), {
    ...init,
    headers: headers(apiKey, init?.headers as Record<string, string> | undefined),
    signal: init?.signal,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`.trim());
  }
  return (await response.json()) as T;
}

async function fetchText(path: string, init?: RequestInit, baseUrl = currentBaseUrl, apiKey?: string): Promise<string> {
  const response = await fetch(apiUrl(path, baseUrl), {
    ...init,
    headers: headers(apiKey, init?.headers as Record<string, string> | undefined),
    signal: init?.signal,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`.trim());
  }
  return await response.text();
}

async function readNdjson(response: Response, onLine: (payload: any) => void): Promise<void> {
  const reader = response.body?.getReader();
  if (!reader) return;

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx = buffer.indexOf("\n");
    while (idx >= 0) {
      const line = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 1);
      idx = buffer.indexOf("\n");
      if (!line) continue;
      try {
        onLine(JSON.parse(line));
      } catch {
        // Ignore partial/bad lines; Ollama streams newline-delimited JSON.
      }
    }
  }

  const tail = buffer.trim();
  if (tail) {
    try {
      onLine(JSON.parse(tail));
    } catch {
      // ignore
    }
  }
}

function emptyUsage(): Usage {
  return {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}

function baseAssistant(model: Model, provider = PROVIDER_NAME): AssistantMessage {
  return {
    role: "assistant",
    content: [],
    api: model.api,
    provider,
    model: model.id,
    usage: emptyUsage(),
    stopReason: "stop",
    timestamp: Date.now(),
  };
}

function collectText(content: string | (TextContent | ImageContent)[]): { text: string; images: string[] } {
  if (typeof content === "string") return { text: content, images: [] };
  const text = content.filter((p: TextContent | ImageContent) => p.type === "text").map((p) => p.text).join("\n");
  const images = content.filter((p: TextContent | ImageContent) => p.type === "image").map((p) => p.data);
  return { text, images };
}

function convertMessages(messages: Message[]): any[] {
  const out: any[] = [];

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];

    if (msg.role === "user") {
      const { text, images } = collectText(msg.content);
      const payload: Record<string, unknown> = { role: "user", content: text };
      if (images.length) payload.images = images;
      out.push(payload);
      continue;
    }

    if (msg.role === "assistant") {
      const assistant: Record<string, unknown> = { role: "assistant" };
      const content: string[] = [];
      const thinking: string[] = [];
      const toolCalls: unknown[] = [];

      for (const block of msg.content) {
        if (block.type === "text" && block.text.trim()) content.push(block.text);
        if (block.type === "thinking" && block.thinking.trim()) thinking.push(block.thinking);
        if (block.type === "toolCall") {
          toolCalls.push({
            type: "function",
            function: {
              index: toolCalls.length,
              name: block.name,
              arguments: block.arguments,
            },
          });
        }
      }

      if (content.length) assistant.content = content.join("");
      if (thinking.length) assistant.thinking = thinking.join("");
      if (toolCalls.length) assistant.tool_calls = toolCalls;
      out.push(assistant);
      continue;
    }

    if (msg.role === "toolResult") {
      const tool = msg as ToolResultMessage;
      const text = tool.content.map((block: TextContent | ImageContent) => (block.type === "text" ? block.text : `[image:${block.mimeType}]`)).join("\n");
      out.push({ role: "tool", tool_name: tool.toolName, content: text });
    }
  }

  return out;
}

function normalizeThink(model: Model, reasoning?: SimpleStreamOptions["reasoning"]): boolean | "low" | "medium" | "high" | "max" {
  if (!reasoning || reasoning === "off") return false;

  const mapped = model.thinkingLevelMap?.[reasoning === "minimal" ? "minimal" : reasoning];
  if (mapped !== undefined) {
    return mapped === null ? false : (mapped as boolean | "low" | "medium" | "high" | "max");
  }

  const id = `${model.id} ${model.name ?? ""}`.toLowerCase();
  if (id.includes("gpt-oss")) {
    if (reasoning === "minimal" || reasoning === "low") return "low";
    if (reasoning === "medium") return "medium";
    return "high";
  }

  if (/(^|[-_/])deepseek-v4($|[-_/])/.test(id) && reasoning === "xhigh") {
    return "max";
  }

  return true;
}

function stopReasonFromDone(doneReason?: string): StopReason {
  switch (doneReason) {
    case "stop":
    case "end_turn":
    case "stop_sequence":
      return "stop";
    case "length":
    case "max_tokens":
      return "length";
    case "tool_use":
      return "toolUse";
    default:
      return "error";
  }
}

function pullUsage(chunk: any) {
  return {
    input: chunk.prompt_eval_count ?? 0,
    output: chunk.eval_count ?? 0,
  };
}

function annotateUsage(output: AssistantMessage, model: Model) {
  output.usage.totalTokens = output.usage.input + output.usage.output + output.usage.cacheRead + output.usage.cacheWrite;
  calculateCost(model, output.usage);
}

function addTextBlock(output: AssistantMessage): number {
  output.content.push({ type: "text", text: "" });
  return output.content.length - 1;
}

function addThinkingBlock(output: AssistantMessage): number {
  output.content.push({ type: "thinking", thinking: "" });
  return output.content.length - 1;
}

function addToolBlock(output: AssistantMessage, id: string, name: string, argumentsObject: Record<string, unknown>): number {
  output.content.push({ type: "toolCall", id, name, arguments: argumentsObject });
  return output.content.length - 1;
}

function appendToolCall(output: AssistantMessage, call: any): number {
  const fn = call?.function ?? {};
  const name = String(fn.name ?? "tool");
  let args: Record<string, unknown> = {};
  if (typeof fn.arguments === "string") {
    try {
      args = JSON.parse(fn.arguments);
    } catch {
      args = { raw: fn.arguments };
    }
  } else if (fn.arguments && typeof fn.arguments === "object") {
    args = fn.arguments;
  }
  return addToolBlock(output, String(fn.index ?? name), name, args);
}

function buildChatPayload(model: Model, context: Context, options?: SimpleStreamOptions) {
  const messages = convertMessages(context.messages);
  const think = normalizeThink(model, options?.reasoning);
  const payload: Record<string, unknown> = {
    model: model.id,
    messages,
    stream: true,
  };

  if (context.systemPrompt) payload.system = context.systemPrompt;
  if (context.tools?.length) {
    payload.tools = context.tools.map((tool: Tool) => ({
      type: "function",
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.parameters,
      },
    }));
  }

  if (think !== false) payload.think = think;
  if (options?.maxTokens) payload.max_tokens = options.maxTokens;
  if (typeof options?.temperature === "number") payload.options = { temperature: options.temperature };
  return payload;
}

function isDeepSeekModel(model: Model): boolean {
  const id = `${model.id} ${model.name ?? ""}`.toLowerCase();
  return /(^|[-_/])deepseek-v3(\.1)?($|[-_/])/.test(id) ||
    /(^|[-_/])deepseek-v4($|[-_/])/.test(id) ||
    /(^|[-_/])deepseek-r1($|[-_/])/.test(id);
}

/**
 * Parse DSML (DeepSeek Markup Language) tool-call blocks from text content.
 * DeepSeek models are trained to emit tool calls in this XML-like format.
 * When Ollama's native /api/chat endpoint returns DSML in the content field
 * instead of structured tool_calls, we parse them into proper ToolCall blocks.
 */
const DSML_TOOL_CALLS_RE = /<｜tool_calls｜>([\s\S]*?)<\/｜tool_calls｜>/g;
const DSML_INVOKE_RE = /<｜invoke｜\s+name="([^"]+)"\s*｜>([\s\S]*?)<\/｜invoke｜>/g;
const DSML_PARAM_RE = /<｜parameter｜\s+name="([^"]+)"\s+string="(true|false)"\s*｜>([\s\S]*?)<\/｜parameter｜>/g;

interface ParsedToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

function parseDsmlToolCalls(text: string): ParsedToolCall[] {
  const toolCalls: ParsedToolCall[] = [];
  for (const blockMatch of text.matchAll(DSML_TOOL_CALLS_RE)) {
    const block = blockMatch[1];
    for (const invokeMatch of block.matchAll(DSML_INVOKE_RE)) {
      const name = invokeMatch[1];
      const body = invokeMatch[2];
      const args: Record<string, unknown> = {};
      for (const paramMatch of body.matchAll(DSML_PARAM_RE)) {
        const paramName = paramMatch[1];
        const isString = paramMatch[2] === "true";
        const raw = paramMatch[3];
        args[paramName] = isString ? raw : JSON.parse(raw);
      }
      toolCalls.push({ name, arguments: args });
    }
  }
  return toolCalls;
}

/**
 * Extract DSML tool calls from text content and convert to ToolCall blocks.
 * Returns the cleaned text (with DSML blocks stripped) and any parsed tool calls.
 */
function extractDsmlToolCalls(
  text: string,
  model: Model,
  output: AssistantMessage,
  stream: ReturnType<typeof createAssistantMessageEventStream>,
): string {
  if (!isDeepSeekModel(model)) return text;
  const parsed = parseDsmlToolCalls(text);
  if (parsed.length === 0) return text;
  for (const call of parsed) {
    const idx = addToolBlock(output, crypto.randomUUID(), call.name, call.arguments);
    stream.push({ type: "toolcall_start", contentIndex: idx, partial: output });
    stream.push({ type: "toolcall_end", contentIndex: idx, toolCall: output.content[idx] as ToolCall, partial: output });
  }
  return text.replace(DSML_TOOL_CALLS_RE, "").trim();
}

function streamOllama(model: Model, context: Context, options?: SimpleStreamOptions): AssistantMessageEventStream {
  const stream = createAssistantMessageEventStream();

  (async () => {
    const output = baseAssistant(model);
    let textIndex = -1;
    let thinkingIndex = -1;
    let sawContent = false;

    try {
      if (model.maxTokens === 0) {
        throw new Error(`Model ${model.id} looks embeddings-only; use /ollama-embed instead.`);
      }

      const apiKey = resolveApiKey(options?.apiKey);
      const response = await fetch(apiUrl("/chat"), {
        method: "POST",
        headers: headers(apiKey),
        body: JSON.stringify(buildChatPayload(model, context, options)),
        signal: options?.signal,
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      stream.push({ type: "start", partial: output });

      await readNdjson(response, (chunk) => {
        if (chunk?.message?.thinking) {
          if (thinkingIndex < 0) {
            output.content.push({ type: "thinking", thinking: "" });
            thinkingIndex = output.content.length - 1;
            stream.push({ type: "thinking_start", contentIndex: thinkingIndex, partial: output });
          }
          const block = output.content[thinkingIndex] as ThinkingContent;
          block.thinking += String(chunk.message.thinking);
          stream.push({ type: "thinking_delta", contentIndex: thinkingIndex, delta: String(chunk.message.thinking), partial: output });
        }

        if (chunk?.message?.content) {
          if (textIndex < 0) {
            textIndex = addTextBlock(output);
            stream.push({ type: "text_start", contentIndex: textIndex, partial: output });
          }
          const block = output.content[textIndex] as TextContent;
          block.text += String(chunk.message.content);
          sawContent = true;
          stream.push({ type: "text_delta", contentIndex: textIndex, delta: String(chunk.message.content), partial: output });
        }

        if (Array.isArray(chunk?.message?.tool_calls) && chunk.message.tool_calls.length) {
          for (const call of chunk.message.tool_calls) {
            const idx = appendToolCall(output, call);
            stream.push({ type: "toolcall_start", contentIndex: idx, partial: output });
            stream.push({ type: "toolcall_end", contentIndex: idx, toolCall: output.content[idx] as ToolCall, partial: output });
          }
        }

        if (chunk?.done) {
          output.stopReason = stopReasonFromDone(chunk.done_reason);
          const usage = pullUsage(chunk);
          output.usage.input = usage.input;
          output.usage.output = usage.output;
          annotateUsage(output, model);
        }
      });

      if (thinkingIndex >= 0) {
        const block = output.content[thinkingIndex] as ThinkingContent;
        stream.push({ type: "thinking_end", contentIndex: thinkingIndex, content: block.thinking, partial: output });
      }
      if (textIndex >= 0) {
        const block = output.content[textIndex] as TextContent;
        block.text = extractDsmlToolCalls(block.text, model, output, stream);
        stream.push({ type: "text_end", contentIndex: textIndex, content: block.text, partial: output });
      }

      if (!sawContent && !output.content.length) {
        output.content.push({ type: "text", text: "" });
        stream.push({ type: "text_start", contentIndex: 0, partial: output });
        stream.push({ type: "text_end", contentIndex: 0, content: "", partial: output });
      }

      if (options?.signal?.aborted) throw new Error("Request aborted");
      stream.push({ type: "done", reason: output.stopReason as "stop" | "length" | "toolUse", message: output });
      stream.end();
    } catch (error) {
      output.stopReason = options?.signal?.aborted ? "aborted" : "error";
      output.errorMessage = error instanceof Error ? error.message : String(error);
      stream.push({ type: "error", reason: output.stopReason, error: output });
      stream.end();
    }
  })();

  return stream;
}

async function discoverCatalog(baseUrl = currentBaseUrl, apiKey?: string): Promise<ModelCatalog> {
  const tags = await fetchJson<OllamaTagsResponse>("/tags", undefined, baseUrl, apiKey);
  const entries = tags.models ?? [];
  const shows = await Promise.all(
    entries.map(async (tag) => {
      try {
        return { tag, show: await fetchJson<OllamaShowResponse>("/show", { method: "POST", body: JSON.stringify({ model: tag.name }) }, baseUrl, apiKey) };
      } catch {
        return { tag };
      }
    }),
  );
  return { entries: shows, models: shows.map(toProviderModel) };
}

function renderCatalog(catalog: ModelCatalog): string[] {
  return catalog.entries.map(({ tag, show }) => {
    const flags = capabilityFlags(tag, show);
    const ctx = inferContextWindow(show).toLocaleString();
    return `${tag.name} — ${ctx} ctx${capabilityLabel(flags)}`;
  });
}

async function refreshProvider(pi: ExtensionAPI): Promise<ModelCatalog> {
  const apiKey = resolveApiKey();
  currentBaseUrl = normalizeBaseUrl(process.env.OLLAMA_BASE_URL ?? DEFAULT_BASE_URL);
  currentCatalog = await discoverCatalog(currentBaseUrl, apiKey);

  pi.registerProvider(PROVIDER_NAME, {
    name: "Ollama Native",
    baseUrl: currentBaseUrl,
    apiKey: API_KEY_ENV,
    authHeader: true,
    api: API_NAME,
    models: currentCatalog.models,
    streamSimple: streamOllama,
  });

  return currentCatalog;
}

function parseCommandArgs(args: string): { command: string; rest: string[] } {
  const parts = args.trim().split(/\s+/).filter(Boolean);
  return { command: (parts[0] ?? "models").toLowerCase(), rest: parts.slice(1) };
}

async function runCommand(args: string, ctx: any, pi: ExtensionAPI) {
  const collectStatuses = async (response: Response): Promise<string[]> => {
    const statuses: string[] = [];
    await readNdjson(response, (chunk) => {
      if (typeof chunk?.status === "string") statuses.push(chunk.status);
    });
    return statuses;
  };

  const { command, rest } = parseCommandArgs(args);

  switch (command) {
    case "models":
    case "list": {
      await refreshProvider(pi).catch(() => undefined);
      ctx.ui.setWidget("ollama", renderCatalog(currentCatalog));
      ctx.ui.notify(`Ollama models: ${currentCatalog.entries.length} total`, "info");
      return;
    }
    case "refresh": {
      const catalog = await refreshProvider(pi);
      ctx.ui.setWidget("ollama", renderCatalog(catalog));
      ctx.ui.notify(`Refreshed ${catalog.entries.length} Ollama models.`, "success");
      return;
    }
    case "show": {
      const model = rest[0];
      if (!model) return ctx.ui.notify("Usage: /ollama show <model>", "error");
      const show = await fetchJson<OllamaShowResponse>("/show", { method: "POST", body: JSON.stringify({ model }) }, currentBaseUrl, resolveApiKey());
      ctx.ui.setWidget("ollama", [
        `model: ${model}`,
        `capabilities: ${(show.capabilities ?? []).join(", ") || "n/a"}`,
        `context: ${inferContextWindow(show).toLocaleString()}`,
        `family: ${show.details?.family ?? "n/a"}`,
        `parameter_size: ${show.details?.parameter_size ?? "n/a"}`,
        `quantization: ${show.details?.quantization_level ?? "n/a"}`,
      ]);
      return;
    }
    case "ps": {
      const ps = await fetchJson<{ models?: unknown[] }>("/ps", undefined, currentBaseUrl, resolveApiKey());
      ctx.ui.setWidget("ollama", [JSON.stringify(ps, null, 2)]);
      return;
    }
    case "version": {
      const version = await fetchText("/version", undefined, currentBaseUrl, resolveApiKey());
      ctx.ui.notify(`Ollama ${version.trim()}`, "info");
      return;
    }
    case "pull":
    case "push": {
      const model = rest[0];
      if (!model) return ctx.ui.notify(`Usage: /ollama ${command} <model>`, "error");
      const response = await fetch(apiUrl(`/${command}`), {
        method: "POST",
        headers: headers(resolveApiKey()),
        body: JSON.stringify({ model, stream: true }),
      });
      if (!response.ok) throw new Error(await response.text());
      const statuses = await collectStatuses(response);
      ctx.ui.notify(`Ollama ${command}: ${statuses.at(-1) ?? "done"}`, "success");
      return;
    }
    case "create": {
      const model = rest[0];
      if (!model) return ctx.ui.notify("Usage: /ollama create <model> [from]", "error");
      const body: Record<string, unknown> = { model, stream: true };
      if (rest[1]) body.from = rest[1];
      const response = await fetch(apiUrl("/create"), {
        method: "POST",
        headers: headers(resolveApiKey()),
        body: JSON.stringify(body),
      });
      if (!response.ok) throw new Error(await response.text());
      const statuses = await collectStatuses(response);
      ctx.ui.notify(`Ollama create: ${statuses.at(-1) ?? "done"}`, "success");
      return;
    }
    case "copy": {
      const source = rest[0];
      const destination = rest[1];
      if (!source || !destination) return ctx.ui.notify("Usage: /ollama copy <source> <destination>", "error");
      const response = await fetch(apiUrl("/copy"), {
        method: "POST",
        headers: headers(resolveApiKey()),
        body: JSON.stringify({ source, destination }),
      });
      if (!response.ok) throw new Error(await response.text());
      ctx.ui.notify(`Copied ${source} → ${destination}`, "success");
      return;
    }
    case "delete": {
      const model = rest[0];
      if (!model) return ctx.ui.notify("Usage: /ollama delete <model>", "error");
      const response = await fetch(apiUrl("/delete"), {
        method: "DELETE",
        headers: headers(resolveApiKey()),
        body: JSON.stringify({ model }),
      });
      if (!response.ok) throw new Error(await response.text());
      ctx.ui.notify(`Deleted ${model}`, "success");
      return;
    }
    case "embed": {
      const model = rest[0];
      const input = rest.slice(1).join(" ");
      if (!model || !input) return ctx.ui.notify("Usage: /ollama embed <model> <text>", "error");
      const result = await fetchJson<{ embeddings?: number[][] }>("/embed", { method: "POST", body: JSON.stringify({ model, input }) }, currentBaseUrl, resolveApiKey());
      ctx.ui.setWidget("ollama", [JSON.stringify(result.embeddings?.[0]?.slice(0, 12) ?? [], null, 2)]);
      return;
    }
    case "help":
    default: {
      ctx.ui.setWidget("ollama", [
        "/ollama models|list",
        "/ollama refresh",
        "/ollama show <model>",
        "/ollama ps",
        "/ollama version",
        "/ollama pull <model>",
        "/ollama push <model>",
        "/ollama create <name> [from]",
        "/ollama copy <source> <dest>",
        "/ollama delete <model>",
        "/ollama embed <model> <text>",
      ]);
    }
  }
}

export default async function (pi: ExtensionAPI) {
  await refreshProvider(pi).catch(() => {
    pi.registerProvider(PROVIDER_NAME, {
      name: "Ollama Native",
      baseUrl: currentBaseUrl,
      apiKey: API_KEY_ENV,
      authHeader: true,
      api: API_NAME,
      models: [],
      streamSimple: streamOllama,
    });
  });

  pi.registerCommand("ollama", {
    description: "Native Ollama API commands",
    handler: async (args: string, ctx: any) => runCommand(args, ctx, pi),
  });

  pi.registerCommand("ollama-models", {
    description: "List Ollama models",
    handler: async (_args: string, ctx: any) => runCommand("models", ctx, pi),
  });

  pi.registerCommand("ollama-refresh", {
    description: "Refresh Ollama models",
    handler: async (_args: string, ctx: any) => runCommand("refresh", ctx, pi),
  });

  pi.registerCommand("ollama-show", {
    description: "Show Ollama model details",
    handler: async (args: string, ctx: any) => runCommand(`show ${args}`, ctx, pi),
  });

  pi.registerCommand("ollama-ps", {
    description: "Show running Ollama models",
    handler: async (_args: string, ctx: any) => runCommand("ps", ctx, pi),
  });

  pi.registerCommand("ollama-version", {
    description: "Show Ollama version",
    handler: async (_args: string, ctx: any) => runCommand("version", ctx, pi),
  });

  pi.registerCommand("ollama-pull", {
    description: "Pull an Ollama model",
    handler: async (args: string, ctx: any) => runCommand(`pull ${args}`, ctx, pi),
  });

  pi.registerCommand("ollama-push", {
    description: "Push an Ollama model",
    handler: async (args: string, ctx: any) => runCommand(`push ${args}`, ctx, pi),
  });

  pi.registerCommand("ollama-create", {
    description: "Create an Ollama model",
    handler: async (args: string, ctx: any) => runCommand(`create ${args}`, ctx, pi),
  });

  pi.registerCommand("ollama-copy", {
    description: "Copy an Ollama model",
    handler: async (args: string, ctx: any) => runCommand(`copy ${args}`, ctx, pi),
  });

  pi.registerCommand("ollama-delete", {
    description: "Delete an Ollama model",
    handler: async (args: string, ctx: any) => runCommand(`delete ${args}`, ctx, pi),
  });

  pi.registerCommand("ollama-embed", {
    description: "Create Ollama embeddings",
    handler: async (args: string, ctx: any) => runCommand(`embed ${args}`, ctx, pi),
  });
}
