# frozen_string_literal: true

module Ollama
  module Openai
    # Base error mirrors ruby-openai style; carries HTTP status when available.
    class Error < StandardError
      attr_reader :status_code, :cause_error

      def initialize(message = nil, status_code: nil, cause: nil)
        super(message)
        @status_code = status_code
        @cause_error = cause
      end
    end

    class APIError < Error; end
    class AuthenticationError < Error; end
    class PermissionDeniedError < Error; end
    class NotFoundError < Error; end
    class InvalidRequestError < Error; end
    class RateLimitError < Error; end
    class APIConnectionError < Error; end
    class APITimeoutError < APIConnectionError; end
    class ServiceUnavailableError < Error; end

    # Maps an Ollama-side error into an OpenAI-shaped error class.
    module ErrorMapper
      module_function

      def map(err)
        status = err.respond_to?(:status_code) ? err.status_code : nil

        klass =
          case err
          when Ollama::UnauthorizedError then AuthenticationError
          when Ollama::NotFoundError then NotFoundError
          when Ollama::ModelUnavailableError then ServiceUnavailableError
          when Ollama::TimeoutError then APITimeoutError
          when Ollama::ConnectionFailedError then APIConnectionError
          else
            case status
            when 401 then AuthenticationError
            when 403 then PermissionDeniedError
            when 404 then NotFoundError
            when 400, 422 then InvalidRequestError
            when 429 then RateLimitError
            when 503 then ServiceUnavailableError
            when 500..599 then APIError
            else APIError
            end
          end

        klass.new(err.message, status_code: status, cause: err)
      end
    end
  end
end
