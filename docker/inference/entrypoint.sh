#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[sock-inference] %s\n' "$*" >&2
}

cleanup() {
  local status=$?
  if [[ -n "${SOCK_PID:-}" ]] && kill -0 "${SOCK_PID}" 2>/dev/null; then
    kill "${SOCK_PID}" 2>/dev/null || true
  fi
  if [[ -n "${LINX_PID:-}" ]] && kill -0 "${LINX_PID}" 2>/dev/null; then
    kill "${LINX_PID}" 2>/dev/null || true
  fi
  wait "${SOCK_PID:-}" "${LINX_PID:-}" 2>/dev/null || true
  exit "${status}"
}
trap cleanup EXIT INT TERM

require_secret() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    log "missing required environment variable: ${name}"
    exit 64
  fi
}

bool_flag() {
  local value="${1:-}"
  [[ "${value,,}" == "1" || "${value,,}" == "true" || "${value,,}" == "yes" || "${value,,}" == "on" ]]
}

wait_http() {
  local url="$1"
  local label="$2"
  local timeout_s="${3:-900}"
  local start
  start="$(date +%s)"
  until curl -fsS "${url}" >/dev/null 2>&1; do
    if ! kill -0 "${SOCK_PID:-0}" 2>/dev/null && [[ "${label}" == "sock" ]]; then
      log "sock exited before health became ready"
      wait "${SOCK_PID}" || true
      exit 1
    fi
    if ! kill -0 "${LINX_PID:-0}" 2>/dev/null && [[ "${label}" == "linx" ]]; then
      log "linx exited before health became ready"
      wait "${LINX_PID}" || true
      exit 1
    fi
    if (( "$(date +%s)" - start > timeout_s )); then
      log "timed out waiting for ${label} at ${url}"
      exit 1
    fi
    sleep 2
  done
}

require_secret LINX_JWT_SECRET

mkdir -p "${LINX_DATA_DIR}" /data/logs

linx &
LINX_PID=$!
log "started linx pid=${LINX_PID} addr=${LINX_HTTP_ADDR}"
wait_http "http://127.0.0.1:${LINX_HTTP_ADDR##*:}/health" "linx" 120

sock_args=(
  serve "${SOCK_MODEL}"
  --host "${SOCK_HOST}"
  --port "${SOCK_PORT}"
  --served-model-name "${SOCK_SERVED_MODEL_NAME}"
  --max-model-len "${SOCK_MAX_MODEL_LEN}"
  --gpu-memory-utilization "${SOCK_GPU_MEMORY_UTILIZATION}"
  --max-num-batched-tokens "${SOCK_MAX_NUM_BATCHED_TOKENS}"
  --max-num-seqs "${SOCK_MAX_NUM_SEQS}"
  --attention-backend "${SOCK_ATTENTION_BACKEND}"
  --kv-layout "${SOCK_KV_LAYOUT}"
  --tmh-hot-budget-pct "${SOCK_TMH_HOT_BUDGET_PCT}"
  --language-model-only
  --skip-mm-profiling
)

if [[ -n "${SOCK_CHAT_TEMPLATE:-}" ]]; then
  sock_args+=(--chat-template "${SOCK_CHAT_TEMPLATE}")
fi
if bool_flag "${SOCK_TRUST_REQUEST_CHAT_TEMPLATE:-0}"; then
  sock_args+=(--trust-request-chat-template)
else
  sock_args+=(--no-trust-request-chat-template)
fi
if bool_flag "${SOCK_ENFORCE_EAGER}"; then
  sock_args+=(--enforce-eager)
fi
if [[ -n "${SOCK_API_KEYS:-}" ]]; then
  IFS=',' read -r -a api_keys <<<"${SOCK_API_KEYS}"
  sock_args+=(--api-key)
  for api_key in "${api_keys[@]}"; do
    api_key="${api_key#"${api_key%%[![:space:]]*}"}"
    api_key="${api_key%"${api_key##*[![:space:]]}"}"
    if [[ -n "${api_key}" ]]; then
      sock_args+=("${api_key}")
    fi
  done
elif [[ -n "${SOCK_API_KEY:-}" ]]; then
  sock_args+=(--api-key "${SOCK_API_KEY}")
fi
if [[ -n "${SOCK_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<<"${SOCK_EXTRA_ARGS}"
  sock_args+=("${extra_args[@]}")
fi

log "starting SOCK model=${SOCK_MODEL} layout=${SOCK_KV_LAYOUT} port=${SOCK_PORT}"
sock "${sock_args[@]}" &
SOCK_PID=$!
wait_http "http://127.0.0.1:${SOCK_PORT}/health" "sock" "${SOCK_HEALTH_TIMEOUT_S:-1800}"

ttl_json=null
if [[ "${LINX_TTL_SECS:-0}" != "0" ]]; then
  ttl_json="${LINX_TTL_SECS}"
fi

register_payload="$(jq -n \
  --arg name "${LINX_SERVICE_NAME}" \
  --arg target "http://127.0.0.1:${SOCK_PORT}" \
  --arg auth_mode "${LINX_AUTH_MODE}" \
  --argjson ttl "${ttl_json}" \
  '{name:$name,target_url:$target,enable_websockets:false,auth_mode:$auth_mode,ttl_secs:$ttl}')"

service_json="$(curl -fsS -X POST "http://127.0.0.1:${LINX_HTTP_ADDR##*:}/api/services" \
  -H 'content-type: application/json' \
  -d "${register_payload}")"
public_url="$(printf '%s' "${service_json}" | jq -r '.public_url')"

log "SOCK local OpenAI-compatible base URL: http://127.0.0.1:${SOCK_PORT}/v1"
log "linx published base URL: ${public_url}"
if [[ -n "${SOCK_API_KEY:-}${SOCK_API_KEYS:-}" ]]; then
  log "SOCK API key auth is enabled; use Authorization: Bearer <SOCK_API_KEY>"
else
  log "SOCK API key auth is disabled; set SOCK_API_KEY or SOCK_API_KEYS to require OpenAI-compatible bearer auth"
fi
log "service record: ${service_json}"

wait -n "${SOCK_PID}" "${LINX_PID}"
