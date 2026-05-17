# frozen_string_literal: true

require "ollama_client"
require "securerandom"

module Ollama
  module Openai
    # Drop-in replacement API facade matching ruby-openai's OpenAI::Client.
    # Wraps Ollama::Client to translate requests and normalize responses.
    class Client
      attr_reader :ollama_client

      # Initializes the OpenAI-compatible client.
      # @param ollama_client [Ollama::Client, nil] An existing Ollama client instance
      # @param access_token [String, nil] Ignored for Ollama but accepted for compatibility
      # @param uri_base [String, nil] Base URL of the Ollama instance (e.g., "http://localhost:11434")
      # @param config [Ollama::Config, nil] Custom Ollama configuration
      def initialize(ollama_client: nil, access_token: nil, uri_base: nil, config: nil, **)
        if ollama_client
          @ollama_client = ollama_client
        else
          cfg = config || Ollama::Config.new
          cfg.base_url = uri_base if uri_base
          @ollama_client = Ollama::Client.new(config: cfg)
        end
      end

      # Chat completions API.
      # Matches OpenAI::Client#chat(parameters: {})
      def chat(parameters: {})
        Chat.new(@ollama_client).create(**parameters.transform_keys(&:to_sym))
      end

      # Text completions API.
      # Matches OpenAI::Client#completions(parameters: {})
      def completions(parameters: {})
        Completions.new(@ollama_client).create(**parameters.transform_keys(&:to_sym))
      end

      # Embeddings API.
      # Matches OpenAI::Client#embeddings(parameters: {})
      def embeddings(parameters: {})
        Embeddings.new(@ollama_client).create(**parameters.transform_keys(&:to_sym))
      end

      # Models API.
      # Matches OpenAI::Client#models
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

        def create(model:, messages:, tools: nil, temperature: nil, top_p: nil, max_tokens: nil, max_completion_tokens: nil, stop: nil, stream: nil, **)
          num_predict = max_tokens || max_completion_tokens
          ollama_tools = translate_tools(tools)
          options = { temperature: temperature, top_p: top_p, num_predict: num_predict, stop: stop }.compact

          if stream
            stream_id = "chatcmpl-#{SecureRandom.hex(12)}"
            created_time = Time.now.to_i

            hooks = {
              on_token: ->(text, _logprobs = nil) {
                chunk = {
                  "id" => stream_id,
                  "object" => "chat.completion.chunk",
                  "created" => created_time,
                  "model" => model,
                  "choices" => [
                    { "index" => 0, "delta" => { "content" => text }, "finish_reason" => nil }
                  ]
                }
                stream.respond_to?(:call) ? stream.call(chunk) : stream.yield(chunk)
              },
              on_tool_call: ->(tc) {
                chunk = {
                  "id" => stream_id,
                  "object" => "chat.completion.chunk",
                  "created" => created_time,
                  "model" => model,
                  "choices" => [
                    {
                      "index" => 0,
                      "delta" => {
                        "tool_calls" => [
                          {
                            "index" => 0,
                            "type" => "function",
                            "function" => { "name" => tc[:name], "arguments" => tc[:arguments].to_json }
                          }
                        ]
                      },
                      "finish_reason" => nil
                    }
                  ]
                }
                stream.respond_to?(:call) ? stream.call(chunk) : stream.yield(chunk)
              },
              on_complete: -> {
                chunk = {
                  "id" => stream_id,
                  "object" => "chat.completion.chunk",
                  "created" => created_time,
                  "model" => model,
                  "choices" => [
                    { "index" => 0, "delta" => {}, "finish_reason" => "stop" }
                  ]
                }
                stream.respond_to?(:call) ? stream.call(chunk) : stream.yield(chunk)
              },
              on_error: ->(err) {
                raise map_error(err)
              }
            }

            @client.chat(model: model, messages: messages, tools: ollama_tools, options: options, hooks: hooks)
            return nil
          end

          begin
            res = @client.chat(model: model, messages: messages, tools: ollama_tools, options: options)
            format_response(res, model)
          rescue Ollama::Error => e
            raise map_error(e)
          end
        end

        private

        def translate_tools(tools)
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

        def map_error(err)
          Ollama::Openai::Error.new(err.message)
        end
      end

      # Completions adapter
      class Completions
        def initialize(client)
          @client = client
        end

        def create(model:, prompt:, temperature: nil, top_p: nil, max_tokens: nil, stop: nil, **)
          options = { temperature: temperature, top_p: top_p, num_predict: max_tokens, stop: stop }.compact
          begin
            text = @client.generate(model: model, prompt: prompt, options: options)
            {
              "id" => "cmpl-#{SecureRandom.hex(12)}",
              "object" => "text_completion",
              "created" => Time.now.to_i,
              "model" => model,
              "choices" => [{ "index" => 0, "text" => text, "finish_reason" => "stop" }]
            }
          rescue Ollama::Error => e
            raise Ollama::Openai::Error, e.message
          end
        end
      end

      # Embeddings adapter
      class Embeddings
        def initialize(client)
          @client = client
        end

        def create(model:, input:, **)
          begin
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
            raise Ollama::Openai::Error, e.message
          end
        end
      end

      # Models adapter
      class Models
        def initialize(client)
          @client = client
        end

        def list
          begin
            models = @client.list_models || []
            {
              "object" => "list",
              "data" => models.map do |m|
                {
                  "id" => m["name"],
                  "object" => "model",
                  "created" => Time.now.to_i,
                  "owned_by" => "ollama"
                }
              end
            }
          rescue Ollama::Error => e
            raise Ollama::Openai::Error, e.message
          end
        end
      end
    end
  end
end
