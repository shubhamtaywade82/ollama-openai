# frozen_string_literal: true

# Live integration tests hitting a real Ollama daemon.
# Excluded by default; run with: INTEGRATION=1 bundle exec rspec spec/integration

RSpec.describe Ollama::Openai::Client, :integration do
  before do
    reason = IntegrationHelper.skip_reason(requires_chat: true)
    skip(reason) if reason
  end

  let(:client) { described_class.new(uri_base: IntegrationHelper::OLLAMA_URL) }
  let(:model) { IntegrationHelper.chat_model }

  it "returns a chat completion in OpenAI envelope" do
    res = client.chat(parameters: {
      model: model,
      messages: [{ role: "user", content: "Reply with: ok" }]
    })
    expect(res["object"]).to eq("chat.completion")
    expect(res["choices"][0]["message"]["content"].to_s.length).to be > 0
    expect(res["usage"]).to include("prompt_tokens", "completion_tokens")
  end

  it "streams chunks ending with [DONE]" do
    chunks = []
    client.chat(parameters: {
      model: model,
      messages: [{ role: "user", content: "Say ok" }],
      stream: ->(c) { chunks << c }
    })
    expect(chunks.last).to eq("[DONE]")
    expect(chunks.any? { |c| c.is_a?(Hash) && c["object"] == "chat.completion.chunk" }).to be true
  end

  it "lists models in OpenAI envelope" do
    res = client.models.list
    expect(res["object"]).to eq("list")
    expect(res["data"]).to be_a(Array)
  end

  context "with an embedding model available" do
    before do
      reason = IntegrationHelper.skip_reason(requires_embed: true)
      skip(reason) if reason
    end

    it "embeds in OpenAI envelope" do
      res = client.embeddings(parameters: { model: IntegrationHelper.embed_model, input: "hi" })
      expect(res["object"]).to eq("list")
      expect(res["data"][0]["embedding"]).to be_a(Array)
    end
  end
end
