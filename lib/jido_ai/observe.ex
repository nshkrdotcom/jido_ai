defmodule Jido.AI.Observe do
  @moduledoc """
  AI observability boundary for telemetry events and spans.

  This module centralizes:

  - Canonical AI telemetry event names
  - Required AI metadata/measurement normalization
  - Feature-gated event emission
  - Span lifecycle wrappers that honor AI observability config
  - Sensitive value redaction and safe preview helpers for telemetry payloads
  """

  alias Jido.AI.Log
  alias Jido.Observe, as: CoreObserve

  @required_metadata_keys [
    :agent_id,
    :request_id,
    :run_id,
    :iteration,
    :llm_call_id,
    :tool_call_id,
    :tool_name,
    :model,
    :origin,
    :operation,
    :strategy,
    :termination_reason,
    :error_type
  ]

  @required_measurement_keys [
    :duration_ms,
    :input_tokens,
    :output_tokens,
    :total_tokens,
    :retry_count,
    :queue_ms
  ]

  @sensitive_exact_keys [
    "api_key",
    "apikey",
    "password",
    "secret",
    "token",
    "auth_token",
    "authtoken",
    "private_key",
    "privatekey",
    "access_key",
    "accesskey",
    "bearer",
    "api_secret",
    "apisecret",
    "client_secret",
    "clientsecret"
  ]

  @sensitive_contains ["secret_"]
  @sensitive_suffixes ["_secret", "_key", "_token", "_password"]

  @type obs_cfg :: map() | nil
  @type event_name :: [atom()]
  @type measurements :: map()
  @type metadata :: map()
  @type span_ctx :: CoreObserve.span_ctx() | :noop
  @type feature_gate :: :llm_deltas
  @type tool_result_summary :: %{
          required(:status) => :ok | :error | :unknown,
          optional(:effect_count) => non_neg_integer(),
          optional(:error_type) => atom(),
          optional(:preview) => term()
        }

  @doc """
  Builds an LLM telemetry event path under `[:jido, :ai, :llm, ...]`.
  """
  @spec llm(atom()) :: event_name()
  def llm(event), do: [:jido, :ai, :llm, event]

  @doc """
  Builds a tool telemetry event path under `[:jido, :ai, :tool, ...]`.
  """
  @spec tool(atom()) :: event_name()
  def tool(event), do: [:jido, :ai, :tool, event]

  @doc """
  Builds a request telemetry event path under `[:jido, :ai, :request, ...]`.
  """
  @spec request(atom()) :: event_name()
  def request(event), do: [:jido, :ai, :request, event]

  @doc """
  Builds a strategy telemetry event path under `[:jido, :ai, :strategy, strategy, ...]`.
  """
  @spec strategy(atom(), atom()) :: event_name()
  def strategy(strategy, event), do: [:jido, :ai, :strategy, strategy, event]

  @doc """
  Builds a tool execution telemetry event path for executor internals.
  """
  @spec tool_execute(atom()) :: event_name()
  def tool_execute(event), do: [:jido, :ai, :tool, :execute, event]

  @doc """
  Emits an AI telemetry event when enabled in `obs_cfg`.

  Supports optional feature gates via `opts`:

  - `feature_gate: :llm_deltas`
  """
  @spec emit(obs_cfg(), event_name(), measurements(), metadata(), keyword()) :: :ok
  def emit(obs_cfg, event, measurements \\ %{}, metadata \\ %{}, opts \\ [])

  def emit(obs_cfg, event, measurements, metadata, opts)
      when is_list(event) and is_map(measurements) and is_map(metadata) and is_list(opts) do
    obs_cfg = normalize_obs_cfg(obs_cfg)

    if emit_enabled?(obs_cfg, opts) and valid_event?(event) do
      :telemetry.execute(
        event,
        ensure_required_measurements(measurements),
        metadata
        |> ensure_required_metadata()
        |> enrich_with_trace_metadata()
      )
    end

    :ok
  end

  @doc """
  Starts a telemetry span when AI telemetry is enabled.

  Returns `:noop` when telemetry is disabled or event prefix is invalid.
  """
  @spec start_span(obs_cfg(), event_name(), metadata()) :: span_ctx()
  def start_span(obs_cfg, event_prefix, metadata \\ %{})

  def start_span(obs_cfg, event_prefix, metadata)
      when is_list(event_prefix) and is_map(metadata) do
    obs_cfg = normalize_obs_cfg(obs_cfg)

    if emit_enabled?(obs_cfg, []) and valid_event?(event_prefix) do
      CoreObserve.start_span(
        event_prefix,
        metadata
        |> ensure_required_metadata()
        |> enrich_with_trace_metadata()
      )
    else
      :noop
    end
  end

  @doc """
  Finishes a span started by `start_span/3`.
  """
  @spec finish_span(span_ctx(), measurements()) :: :ok
  def finish_span(:noop, _extra_measurements), do: :ok

  def finish_span(span_ctx, extra_measurements) when is_map(extra_measurements) do
    CoreObserve.finish_span(span_ctx, ensure_required_measurements(extra_measurements))
  end

  @doc """
  Finishes a span with error details.
  """
  @spec finish_span_error(span_ctx(), atom(), term(), list()) :: :ok
  def finish_span_error(:noop, _kind, _reason, _stacktrace), do: :ok

  def finish_span_error(span_ctx, kind, reason, stacktrace) when is_atom(kind) and is_list(stacktrace) do
    CoreObserve.finish_span_error(span_ctx, kind, reason, stacktrace)
  end

  @doc """
  Ensures required AI metadata keys are present with `nil` defaults.
  """
  @spec ensure_required_metadata(map()) :: map()
  def ensure_required_metadata(metadata) when is_map(metadata) do
    Enum.reduce(@required_metadata_keys, metadata, fn key, acc ->
      Map.put_new(acc, key, nil)
    end)
  end

  @doc """
  Ensures required AI measurement keys are present with `0` defaults.
  """
  @spec ensure_required_measurements(map()) :: map()
  def ensure_required_measurements(measurements) when is_map(measurements) do
    Enum.reduce(@required_measurement_keys, measurements, fn key, acc ->
      Map.put_new(acc, key, 0)
    end)
  end

  @doc """
  Redacts sensitive keys recursively for telemetry-safe payloads.
  """
  @spec sanitize_sensitive(term()) :: term()
  def sanitize_sensitive(payload) when is_map(payload) do
    Map.new(payload, fn {key, value} ->
      if sensitive_key?(key) do
        {key, "[REDACTED]"}
      else
        {key, sanitize_sensitive(value)}
      end
    end)
  end

  def sanitize_sensitive(payload) when is_list(payload), do: Enum.map(payload, &sanitize_sensitive/1)
  def sanitize_sensitive(payload), do: payload

  @doc """
  Shapes arbitrary values into a telemetry-safe preview.

  This preserves simple map/list structure where practical, redacts sensitive keys,
  truncates large strings, and falls back to bounded inspect output for tuples,
  structs, and other opaque terms.
  """
  @spec telemetry_safe(term(), keyword()) :: term()
  def telemetry_safe(payload, opts \\ [])

  def telemetry_safe(%_{} = struct, opts), do: safe_inspect_preview(struct, opts)

  def telemetry_safe(payload, opts) when is_map(payload) do
    Map.new(payload, fn {key, value} ->
      if sensitive_key?(key) do
        {key, "[REDACTED]"}
      else
        {key, telemetry_safe(value, opts)}
      end
    end)
  end

  def telemetry_safe(payload, opts) when is_list(payload) do
    payload
    |> Enum.take(Keyword.get(opts, :list_limit, 20))
    |> Enum.map(&telemetry_safe(&1, opts))
  end

  def telemetry_safe(payload, opts) when is_binary(payload) do
    max_length = Keyword.get(opts, :max_length, 200)

    if String.valid?(payload) do
      truncate_binary(payload, max_length)
    else
      safe_inspect_preview(payload, opts)
    end
  end

  def telemetry_safe(payload, _opts)
      when is_atom(payload) or is_number(payload) or is_boolean(payload) or is_nil(payload),
      do: payload

  def telemetry_safe(payload, opts) when is_tuple(payload), do: safe_inspect_preview(payload, opts)
  def telemetry_safe(payload, opts), do: safe_inspect_preview(payload, opts)

  @doc """
  Returns a small, telemetry-safe summary for tool execution results.
  """
  @spec tool_result_summary(term()) :: tool_result_summary()
  def tool_result_summary(result)

  def tool_result_summary({:ok, output, effects}) do
    %{
      status: :ok,
      effect_count: effect_count(effects),
      preview: telemetry_safe(output)
    }
  end

  def tool_result_summary({:ok, output}) do
    %{
      status: :ok,
      effect_count: 0,
      preview: telemetry_safe(output)
    }
  end

  def tool_result_summary({:error, reason, effects}) do
    %{
      status: :error,
      effect_count: effect_count(effects),
      preview: telemetry_safe(reason)
    }
    |> maybe_put(:error_type, infer_error_type(reason))
  end

  def tool_result_summary({:error, reason}) do
    %{
      status: :error,
      effect_count: 0,
      preview: telemetry_safe(reason)
    }
    |> maybe_put(:error_type, infer_error_type(reason))
  end

  def tool_result_summary(other) do
    %{
      status: :unknown,
      preview: telemetry_safe(other)
    }
  end

  defp emit_enabled?(obs_cfg, opts) do
    Map.get(obs_cfg, :emit_telemetry?, true) and feature_gate_enabled?(obs_cfg, Keyword.get(opts, :feature_gate))
  end

  defp normalize_obs_cfg(obs_cfg) when is_map(obs_cfg), do: obs_cfg
  defp normalize_obs_cfg(_), do: %{}

  defp feature_gate_enabled?(_obs_cfg, nil), do: true
  defp feature_gate_enabled?(obs_cfg, :llm_deltas), do: Map.get(obs_cfg, :emit_llm_deltas?, true)
  defp feature_gate_enabled?(_obs_cfg, _unknown), do: true

  defp valid_event?([:jido, :ai, :llm, event]) when event in [:span, :start, :delta, :complete, :error], do: true

  defp valid_event?([:jido, :ai, :tool, event]) when event in [:span, :start, :retry, :complete, :error, :timeout],
    do: true

  defp valid_event?([:jido, :ai, :request, event]) when event in [:start, :complete, :failed, :rejected, :cancelled],
    do: true

  defp valid_event?([:jido, :ai, :strategy, strategy, event]) when is_atom(strategy) and is_atom(event), do: true
  defp valid_event?([:jido, :ai, :tool, :execute, event]) when event in [:start, :stop, :exception], do: true

  defp valid_event?(event) do
    Log.warning(fn -> "Jido.AI.Observe ignored invalid AI telemetry event: #{Log.safe_inspect(event)}" end)
    false
  end

  defp enrich_with_trace_metadata(metadata) do
    case Jido.Tracing.Context.get() do
      nil ->
        metadata

      ctx ->
        Map.merge(
          %{
            jido_trace_id: ctx[:trace_id],
            jido_span_id: ctx[:span_id],
            jido_parent_span_id: ctx[:parent_span_id]
          },
          metadata
        )
    end
  end

  defp sensitive_key?(key) when is_atom(key), do: key |> Atom.to_string() |> sensitive_key?()

  defp sensitive_key?(key) when is_binary(key) do
    key = String.downcase(key)

    key in @sensitive_exact_keys or
      Enum.any?(@sensitive_contains, &String.contains?(key, &1)) or
      Enum.any?(@sensitive_suffixes, &String.ends_with?(key, &1))
  end

  defp sensitive_key?(_key), do: false

  defp truncate_binary(binary, max_length) do
    if String.length(binary) > max_length do
      String.slice(binary, 0, max_length) <> "..."
    else
      binary
    end
  end

  defp safe_inspect_preview(term, opts) do
    Log.safe_inspect(term,
      max_length: Keyword.get(opts, :max_length, 200),
      limit: Keyword.get(opts, :limit, 10)
    )
  end

  defp infer_error_type(%{type: type}) when is_atom(type), do: type
  defp infer_error_type(%{code: type}) when is_atom(type), do: type
  defp infer_error_type(reason) when is_atom(reason), do: reason
  defp infer_error_type(_), do: nil

  defp effect_count(effects) when is_list(effects), do: length(effects)
  defp effect_count(nil), do: 0
  defp effect_count(_effects), do: 1

  defp maybe_put(map, _key, nil), do: map
  defp maybe_put(map, key, value), do: Map.put(map, key, value)
end
