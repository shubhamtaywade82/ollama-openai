# frozen_string_literal: true

require_relative "openai/version"
require_relative "openai/client"

module Ollama
  module Openai
    class Error < StandardError; end
  end
end
