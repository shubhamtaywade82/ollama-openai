# frozen_string_literal: true

RSpec.describe Ollama::Openai do
  it "has a version number" do
    expect(Ollama::Openai::VERSION).not_to be nil
  end

  describe Ollama::Openai::Client do
    let(:mock_ollama) { instance_double(Ollama::Client) }
    let(:client) { described_class.new(ollama_client: mock_ollama) }

    it "formats chat completions matching ruby-openai" do
      mock_message = double(content: "Hello", tool_calls: nil)
      mock_response = instance_double(Ollama::Response, message: mock_message, prompt_eval_count: 10, eval_count: 5, done_reason: "stop")
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

    it "formats embeddings matching ruby-openai" do
      mock_embeddings = double("embeddings")
      expect(mock_ollama).to receive(:embeddings).and_return(mock_embeddings)
      expect(mock_embeddings).to receive(:embed).with(
        model: "llama3",
        input: "Hi"
      ).and_return([0.1, 0.2, 0.3])

      res = client.embeddings(parameters: { model: "llama3", input: "Hi" })
      expect(res["object"]).to eq("list")
      expect(res["data"][0]["embedding"]).to eq([0.1, 0.2, 0.3])
    end

    it "formats models list matching ruby-openai" do
      expect(mock_ollama).to receive(:list_models).and_return([{ "name" => "llama3" }])

      res = client.models.list
      expect(res["object"]).to eq("list")
      expect(res["data"][0]["id"]).to eq("llama3")
    end
  end
end
