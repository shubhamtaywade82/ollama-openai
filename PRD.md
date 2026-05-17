# Product Requirements Document: ollama-openai

## 1. Product Overview
**Name:** `ollama-openai`
**Role in Ecosystem:** The OpenAI-compatible adapter for Ollama.
**Goal:** Provide a drop-in replacement API facade that allows any system designed for the official `ruby-openai` gem (e.g., LangChain, Vercel AI SDK wrappers, LiteLLM) to interact seamlessly with a local Ollama instance.

## 2. Strategic Positioning
This gem lives strictly above `ollama-client`. It translates OpenAI-style requests (schemas, parameters, tool definitions) into native Ollama runtime requests and normalizes Ollama responses back into OpenAI-compatible structures. It MUST NOT reinvent transport or retry logic—it relies on `ollama-client` for infrastructure.

## 3. System Requirements & Features
### 3.1. Core Endpoints
- `/v1/chat/completions` (maps to `client.chat`)
- `/v1/completions` (maps to `client.generate`)
- `/v1/embeddings` (maps to `client.embeddings.embed`)
- `/v1/models` (maps to `client.list_models`)

### 3.2. Compatibility Guarantees
- **Tool Calling:** Translate OpenAI JSON Schema tool definitions into Ollama tool formats, and translate Ollama tool call responses back to OpenAI function call formats.
- **Streaming:** Support streaming Deltas matching OpenAI's `chat.completion.chunk` event structure.
- **Errors:** Map native Ollama errors to OpenAI standard error objects where feasible, preserving status codes.

## 4. Implementation Details
- **Dependency:** `ollama-client` (>= 1.3.0)
- **Facade Architecture:** Implement an `Ollama::OpenAI::Client` that wraps an instance of `Ollama::Client`.
- **Translators:** Build strict serializers/deserializers for Requests (e.g., converting `max_tokens` to `num_predict`) and Responses.

## 5. Non-Goals
- Do not implement raw HTTP transport.
- Do not add orchestration or agents.
