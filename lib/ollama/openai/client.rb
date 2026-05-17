# frozen_string_literal: true

require "ollama_client"
require "securerandom"
require "json"

module Ollama
  module Openai
    # Drop-in replacement API facade matching ruby-openai's OpenAI::Client.
    # Wraps Ollama::Client to translate requests and normalize responses.
    class Client
      attr_reader :ollama_client

      def initialize(ollama_client: nil, access_token: nil, uri_base: nil, config: nil, **)
        if ollama_client
          @ollama_client = ollama_client
        else
          cfg = config || Ollama::Config.new
          cfg.base_url = uri_base if uri_base
          @ollama_client = Ollama::Client.new(config: cfg)
        end
      end

      def chat(parameters: {})
        Chat.new(@ollama_client).create(**parameters.transform_keys(&:to_sym))
      end

      def completions(parameters: {})
        Completions.new(@ollama_client).create(**parameters.transform_keys(&:to_sym))
      end

      def embeddings(parameters: {})
        Embeddings.new(@ollama_client).create(**parameters.transform_keys(&:to_sym))
      end

      def models
        @models ||= Models.new(@ollama_client)
      end

      # Chat adapter handling both synchronous and streaming chat completions.
      class Chat
        def initialize(client)
          @client = client
        end

        def completions
          self
        end

        def create(model:, messages:,
                   tools: nil, tool_choice: nil,
                   temperature: nil, top_p: nil, top_k: nil,
                   max_tokens: nil, max_completion_tokens: nil,
                   stop: nil, stream: nil,
                   seed: nil, presence_penalty: nil, frequency_penalty: nil,
                   response_format: nil, **)
          num_predict = max_tokens || max_completion_tokens
          ollama_tools = TranslateTools.call(tools)
          ollama_messages = TranslateMessages.call(messages)
          options = {
            temperature: temperature, top_p: top_p, top_k: top_k,
            num_predict: num_predict, stop: stop,
            seed: seed, presence_penalty: presence_penalty,
            frequency_penalty: frequency_penalty
          }.compact
          format_arg = TranslateResponseFormat.call(response_format)

          chat_args = { model: model, messages: ollama_messages, tools: ollama_tools, options: options }
          chat_args[:format] = format_arg unless format_arg.nil?

          if stream
            stream_id = "chatcmpl-#{SecureRandom.hex(12)}"
            created_time = Time.now.to_i
            emit = ->(c) { stream.respond_to?(:call) ? stream.call(c) : stream.yield(c) }

            hooks = build_stream_hooks(stream_id, created_time, model, emit)
            chat_args[:hooks] = hooks

            begin
              @client.chat(**chat_args)
            rescue Ollama::Error => e
              raise ErrorMapper.map(e)
            end
            emit.call("[DONE]")
            return nil
          end

          begin
            res = @client.chat(**chat_args)
            format_response(res, model)
          rescue Ollama::Error => e
            raise ErrorMapper.map(e)
          end
        end

        private

        def build_stream_hooks(stream_id, created_time, model, emit)
          base = ->(delta, finish_reason = nil) {
            {
              "id" => stream_id,
              "object" => "chat.completion.chunk",
              "created" => created_time,
              "model" => model,
              "choices" => [{ "index" => 0, "delta" => delta, "finish_reason" => finish_reason }]
            }
          }

          {
            on_token: ->(text, _logprobs = nil) { emit.call(base.call({ "content" => text }, nil)) },
            on_tool_call: ->(tc) {
              emit.call(base.call({
                "tool_calls" => [{
                  "index" => 0,
                  "type" => "function",
                  "function" => { "name" => tc[:name], "arguments" => tc[:arguments].to_json }
                }]
              }, nil))
            },
            on_complete: -> { emit.call(base.call({}, "stop")) },
            on_error: ->(err) { raise ErrorMapper.map(err) }
          }
        end

        def format_response(response, model)
          {
            "id" => "chatcmpl-#{SecureRandom.hex(12)}",
            "object" => "chat.completion",
            "created" => Time.now.to_i,
            "model" => model,
            "choices" => [
              {
                "index" => 0,
                "message" => {
                  "role" => "assistant",
                  "content" => response.message.content,
                  "tool_calls" => openai_tool_calls(response)
                }.compact,
                "finish_reason" => response.done_reason || "stop"
              }
            ],
            "usage" => {
              "prompt_tokens" => response.prompt_eval_count || 0,
              "completion_tokens" => response.eval_count || 0,
              "total_tokens" => (response.prompt_eval_count || 0) + (response.eval_count || 0)
            }
          }
        end

        def openai_tool_calls(response)
          return nil unless response.message.tool_calls&.any?

          response.message.tool_calls.map do |tc|
            {
              "id" => "call_#{SecureRandom.hex(8)}",
              "type" => "function",
              "function" => {
                "name" => tc.name,
                "arguments" => tc.arguments.is_a?(String) ? tc.arguments : tc.arguments.to_json
              }
            }
          end
        end
      end

      # Translates an OpenAI `tools` array into Ollama's tool schema.
      module TranslateTools
        module_function

        def call(tools)
          return nil unless tools&.any?

          tools.map do |t|
            fn = t[:function] || t["function"] || {}
            {
              type: "function",
              function: {
                name: fn[:name] || fn["name"],
                description: fn[:description] || fn["description"],
                parameters: fn[:parameters] || fn["parameters"] || { type: "object", properties: {} }
              }
            }
          end
        end
      end

      # Normalises OpenAI message arrays into Ollama's chat schema.
      # `tool` role messages must include the original `tool_call_id` per spec;
      # Ollama uses the same `role: "tool"` so pass-through is correct.
      module TranslateMessages
        module_function

        def call(messages)
          return [] if messages.nil?

          messages.map do |m|
            sym = m.transform_keys(&:to_sym)
            out = { role: sym[:role].to_s, content: sym[:content].to_s }
            out[:tool_calls] = sym[:tool_calls] if sym[:tool_calls]
            out[:tool_call_id] = sym[:tool_call_id] if sym[:tool_call_id]
            out[:images] = sym[:images] if sym[:images]
            out[:name] = sym[:name] if sym[:name]
            out
          end
        end
      end

      # Maps OpenAI `response_format` to Ollama `format:` argument.
      # - {type: "json_object"}  => "json"
      # - {type: "json_schema", json_schema: {schema: {...}}} => schema hash
      module TranslateResponseFormat
        module_function

        def call(rf)
          return nil if rf.nil?

          rf = rf.transform_keys(&:to_sym) if rf.is_a?(Hash)
          case rf[:type].to_s
          when "json_object" then "json"
          when "json_schema"
            js = rf[:json_schema]
            js = js.transform_keys(&:to_sym) if js.is_a?(Hash)
            js && js[:schema]
          end
        end
      end

      # Text completions adapter.
      class Completions
        def initialize(client)
          @client = client
        end

        def create(model:, prompt:, temperature: nil, top_p: nil, max_tokens: nil, stop: nil, seed: nil, **)
          options = { temperature: temperature, top_p: top_p, num_predict: max_tokens, stop: stop, seed: seed }.compact
          text = @client.generate(model: model, prompt: prompt, options: options)
          {
            "id" => "cmpl-#{SecureRandom.hex(12)}",
            "object" => "text_completion",
            "created" => Time.now.to_i,
            "model" => model,
            "choices" => [{ "index" => 0, "text" => text, "finish_reason" => "stop" }]
          }
        rescue Ollama::Error => e
          raise ErrorMapper.map(e)
        end
      end

      # Embeddings adapter.
      class Embeddings
        def initialize(client)
          @client = client
        end

        def create(model:, input:, **)
          vectors = @client.embeddings.embed(model: model, input: input)
          vectors = [vectors] unless input.is_a?(Array)
          {
            "object" => "list",
            "data" => vectors.each_with_index.map do |emb, i|
              { "object" => "embedding", "embedding" => emb, "index" => i }
            end,
            "model" => model
          }
        rescue Ollama::Error => e
          raise ErrorMapper.map(e)
        end
      end

      # Models adapter.
      class Models
        def initialize(client)
          @client = client
        end

        def list
          models = @client.list_models || []
          {
            "object" => "list",
            "data" => models.map do |m|
              { "id" => m["name"], "object" => "model", "created" => Time.now.to_i, "owned_by" => "ollama" }
            end
          }
        rescue Ollama::Error => e
          raise ErrorMapper.map(e)
        end

        def retrieve(id:)
          data = @client.show_model(model: id) || {}
          { "id" => id, "object" => "model", "created" => Time.now.to_i, "owned_by" => "ollama", "details" => data }
        rescue Ollama::Error => e
          raise ErrorMapper.map(e)
        end

        def delete(id:)
          @client.delete_model(model: id)
          { "id" => id, "object" => "model", "deleted" => true }
        rescue Ollama::Error => e
          raise ErrorMapper.map(e)
        end
      end
    end
  end
end
