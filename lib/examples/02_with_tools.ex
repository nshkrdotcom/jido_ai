defmodule Examples.ToolAgent02 do
  alias Jido.AI.Agent
  alias Jido.Actions.Arithmetic.{Add, Subtract, Multiply, Divide}
  require Logger

  def demo do
    {:ok, pid} =
      Agent.start_link(
        ai: [
          model: {:anthropic, model: "claude-3-haiku-20240307"},
          prompt: """
          You are a super math genius.
          You are given a math problem and you need to solve it using the tools provided.
          Always use the tools to solve arithmetic problems rather than calculating yourself.

          <%= @message %>
          """,
          tools: [
            Add,
            Subtract,
            Multiply,
            Divide
          ],
          verbose: true
        ]
      )

    Logger.info("Agent started successfully")

    case Agent.tool_response(pid, "What is 273 + 112 - 937?") do
      {:ok, result} ->
        Logger.info("Result: #{inspect(result, pretty: true)}")

      {:error, error} ->
        Logger.error("Error: #{inspect(error, pretty: true)}")
    end
  end
end
