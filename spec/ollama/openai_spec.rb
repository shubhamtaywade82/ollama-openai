# frozen_string_literal: true

RSpec.describe Ollama::Openai do
  it "has a version number" do
    expect(Ollama::Openai::VERSION).not_to be nil
  end

  describe Ollama::Openai::Client do
    let(:mock_ollama) { instance_double(Ollama::Client) }
    let(:client) { described_class.new(ollama_client: mock_ollama) }

    describe "#chat" do
      it "formats chat completions matching ruby-openai" do
        mock_message = double(content: "Hello", tool_calls: nil)
        mock_response = instance_double(
          Ollama::Response,
          message: mock_message,
          prompt_eval_count: 10,
          eval_count: 5,
          done_reason: "stop"
        )
        expect(mock_ollama).to receive(:chat).with(
          model: "llama3",
          messages: [{ role: "user", content: "Hi" }],
          tools: nil,
          options: { temperature: 0.7 }
        ).and_return(mock_response)

        res = client.chat(parameters: { model: "llama3", messages: [{ role: "user", content: "Hi" }], temperature: 0.7 })
        expect(res["object"]).to eq("chat.completion")
        expect(res["choices"][0]["message"]["content"]).to eq("Hello")
        expect(res["usage"]["total_tokens"]).to eq(15)
      end

      it "maps response_format json_object to Ollama format 'json'" do
        mock_message = double(content: "{}", tool_calls: nil)
        mock_response = instance_double(Ollama::Response, message: mock_message, prompt_eval_count: 1, eval_count: 1, done_reason: "stop")
        expect(mock_ollama).to receive(:chat).with(
          hash_including(format: "json")
        ).and_return(mock_response)

        client.chat(parameters: {
          model: "llama3",
          messages: [{ role: "user", content: "Hi" }],
          response_format: { type: "json_object" }
        })
      end

      it "maps response_format json_schema to schema hash" do
        mock_message = double(content: "{}", tool_calls: nil)
        mock_response = instance_double(Ollama::Response, message: mock_message, prompt_eval_count: 1, eval_count: 1, done_reason: "stop")
        schema = { type: "object", properties: { x: { type: "number" } } }
        expect(mock_ollama).to receive(:chat).with(
          hash_including(format: schema)
        ).and_return(mock_response)

        client.chat(parameters: {
          model: "llama3",
          messages: [{ role: "user", content: "Hi" }],
          response_format: { type: "json_schema", json_schema: { schema: schema } }
        })
      end

      it "passes seed, presence_penalty, frequency_penalty through options" do
        mock_message = double(content: "x", tool_calls: nil)
        mock_response = instance_double(Ollama::Response, message: mock_message, prompt_eval_count: 1, eval_count: 1, done_reason: "stop")
        expect(mock_ollama).to receive(:chat).with(
          hash_including(options: hash_including(seed: 42, presence_penalty: 0.3, frequency_penalty: 0.5))
        ).and_return(mock_response)

        client.chat(parameters: {
          model: "llama3",
          messages: [{ role: "user", content: "Hi" }],
          seed: 42,
          presence_penalty: 0.3,
          frequency_penalty: 0.5
        })
      end

      it "translates assistant tool_calls back into OpenAI format" do
        tc = double(name: "get_weather", arguments: { "loc" => "NYC" })
        mock_message = double(content: nil, tool_calls: [tc])
        mock_response = instance_double(Ollama::Response, message: mock_message, prompt_eval_count: 1, eval_count: 1, done_reason: "tool_calls")
        expect(mock_ollama).to receive(:chat).and_return(mock_response)

        res = client.chat(parameters: { model: "llama3", messages: [] })
        call = res["choices"][0]["message"]["tool_calls"][0]
        expect(call["type"]).to eq("function")
        expect(call["function"]["name"]).to eq("get_weather")
        expect(JSON.parse(call["function"]["arguments"])).to eq({ "loc" => "NYC" })
      end

      it "streams chunks then emits [DONE] sentinel" do
        chunks = []
        proc_stream = ->(c) { chunks << c }

        expect(mock_ollama).to receive(:chat) do |args|
          args[:hooks][:on_token].call("Hello", nil)
          args[:hooks][:on_token].call(" world", nil)
          args[:hooks][:on_complete].call
        end

        client.chat(parameters: { model: "llama3", messages: [], stream: proc_stream })

        expect(chunks.length).to eq(4)
        expect(chunks[0]["choices"][0]["delta"]["content"]).to eq("Hello")
        expect(chunks[1]["choices"][0]["delta"]["content"]).to eq(" world")
        expect(chunks[2]["choices"][0]["finish_reason"]).to eq("stop")
        expect(chunks[3]).to eq("[DONE]")
      end
    end

    describe "#completions" do
      it "formats text completions matching ruby-openai" do
        expect(mock_ollama).to receive(:generate).with(
          model: "llama3",
          prompt: "Hi",
          options: { temperature: 0.7 }
        ).and_return("Hello world")

        res = client.completions(parameters: { model: "llama3", prompt: "Hi", temperature: 0.7 })
        expect(res["object"]).to eq("text_completion")
        expect(res["choices"][0]["text"]).to eq("Hello world")
      end
    end

    describe "#embeddings" do
      it "formats embeddings for single string input" do
        mock_embeddings = double("embeddings")
        expect(mock_ollama).to receive(:embeddings).and_return(mock_embeddings)
        expect(mock_embeddings).to receive(:embed).with(
          model: "llama3",
          input: "Hi"
        ).and_return([0.1, 0.2, 0.3])

        res = client.embeddings(parameters: { model: "llama3", input: "Hi" })
        expect(res["object"]).to eq("list")
        expect(res["data"].length).to eq(1)
        expect(res["data"][0]["embedding"]).to eq([0.1, 0.2, 0.3])
        expect(res["data"][0]["index"]).to eq(0)
      end

      it "formats embeddings for batched array input" do
        mock_embeddings = double("embeddings")
        expect(mock_ollama).to receive(:embeddings).and_return(mock_embeddings)
        expect(mock_embeddings).to receive(:embed).with(
          model: "llama3",
          input: %w[a b]
        ).and_return([[0.1], [0.2]])

        res = client.embeddings(parameters: { model: "llama3", input: %w[a b] })
        expect(res["data"].length).to eq(2)
        expect(res["data"][1]["embedding"]).to eq([0.2])
        expect(res["data"][1]["index"]).to eq(1)
      end
    end

    describe "#models" do
      it "formats models list matching ruby-openai" do
        expect(mock_ollama).to receive(:list_models).and_return([{ "name" => "llama3" }])
        res = client.models.list
        expect(res["object"]).to eq("list")
        expect(res["data"][0]["id"]).to eq("llama3")
      end

      it "retrieve returns single model object" do
        expect(mock_ollama).to receive(:show_model).with(model: "llama3").and_return({ "modelfile" => "..." })
        res = client.models.retrieve(id: "llama3")
        expect(res["id"]).to eq("llama3")
        expect(res["object"]).to eq("model")
      end

      it "delete maps to Ollama delete_model" do
        expect(mock_ollama).to receive(:delete_model).with(model: "llama3").and_return(true)
        res = client.models.delete(id: "llama3")
        expect(res["deleted"]).to eq(true)
        expect(res["id"]).to eq("llama3")
      end
    end

    describe "error mapping" do
      it "raises AuthenticationError on 401" do
        expect(mock_ollama).to receive(:chat).and_raise(Ollama::UnauthorizedError.new("nope", 401))
        expect {
          client.chat(parameters: { model: "x", messages: [] })
        }.to raise_error(Ollama::Openai::AuthenticationError)
      end

      it "raises NotFoundError on 404" do
        expect(mock_ollama).to receive(:chat).and_raise(Ollama::NotFoundError.new("missing", requested_model: "x"))
        expect {
          client.chat(parameters: { model: "x", messages: [] })
        }.to raise_error(Ollama::Openai::NotFoundError)
      end

      it "raises RateLimitError on 429" do
        expect(mock_ollama).to receive(:chat).and_raise(Ollama::HTTPError.new("rate", 429))
        expect {
          client.chat(parameters: { model: "x", messages: [] })
        }.to raise_error(Ollama::Openai::RateLimitError)
      end

      it "raises APIError on 500" do
        expect(mock_ollama).to receive(:chat).and_raise(Ollama::HTTPError.new("boom", 500))
        expect {
          client.chat(parameters: { model: "x", messages: [] })
        }.to raise_error(Ollama::Openai::APIError)
      end
    end
  end
end
