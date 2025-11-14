import logging
import os
import csv
import psycopg2
from io import StringIO

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import httpx
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM

# --- Logging setup ---
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("webapp")

app = FastAPI()
templates = Jinja2Templates(directory=".")


class QueryPayload(BaseModel):
    question: str


# --- Helpers ---


def _pg_conn():
    host = os.getenv("PGHOST", "db")
    port = os.getenv("PGPORT", "5432")
    dbname = os.getenv("POSTGRES_DB", "pg_db")
    user = os.getenv("POSTGRES_USER", "user")
    log.info(f"Connecting to Postgres host={host} port={port} db={dbname} user={user}")
    return psycopg2.connect(
        dbname=dbname,
        user=user,
        password=os.getenv("POSTGRES_PASSWORD", "supersecurepassword"),
        host=host,
        port=port,
    )


def _sql_prompt(question: str) -> str:
    schema_hint = os.getenv(
        "SCHEMA_HINT",
        """
Tables:
- customers(id, name, email, created_at)
- orders(id, customer_id, product_id, quantity, total_price, created_at)
- products(id, name, category, price, created_at)
""",
    ).strip()
    return f"""You are a PostgreSQL SQL generator. Given the user request below, output ONLY a valid SQL SELECT query.
- No explanations, comments, tags, or code fences.
- Must include SELECT and FROM.
- Use the tables listed and join appropriately if needed.
- Do not use DROP/DELETE/UPDATE/INSERT.
- Do not use placeholders.

Database schema:
{schema_hint}

User request:
{question}
"""


def _ollama_base_url() -> str:
    return (
        f"http://{os.getenv('OLLAMA_HOST','ollama')}:{os.getenv('OLLAMA_PORT','11434')}"
    )


def _resolve_ollama_model_sync(preferred: str | None) -> str:
    base_url = _ollama_base_url()
    tags_url = f"{base_url}/api/tags"
    models: list[str] = []
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(tags_url)
            r.raise_for_status()
            data = r.json()
            models = [m["name"] for m in data.get("models", [])]
            log.info("Ollama /api/tags models: %s", models)
    except Exception as e:
        log.warning("Failed to fetch Ollama tags: %s", e)
        return preferred or "sqlcoder"

    if preferred and preferred in models:
        return preferred

    for m in models:
        if "sqlcoder" in m:
            return m

    if models:
        return models[0]

    return preferred or "sqlcoder"


def _ollama_llm_sync(model_name: str) -> OllamaLLM:
    base_url = _ollama_base_url()
    log.info(f"Ollama LLM init model={model_name} base_url={base_url}")
    return OllamaLLM(model=model_name, base_url=base_url)


# Inner state helpers: always use {"value": {...}} in node I/O
def _get_inner(state: dict) -> dict:
    if (
        isinstance(state, dict)
        and "value" in state
        and isinstance(state["value"], dict)
    ):
        return dict(state["value"])
    return dict(state or {})


def _wrap(inner: dict) -> dict:
    return {"value": inner}


# --- LangGraph nodes ---


def generate_sql(state: dict) -> dict:
    inner = _get_inner(state)
    log.info("Node generate_sql entered. Inner keys: %s", list(inner.keys()))
    question = (inner.get("question") or "").strip()
    if not question:
        inner["sql_query"] = ""
        inner["generate_output"] = "No question provided."
        log.warning("generate_sql: empty question; skipping model.")
        return _wrap(inner)

    preferred_model = os.getenv("OLLAMA_MODEL")  # e.g., 'sqlcoder' or 'sqlcoder:latest'
    model_name = preferred_model or "sqlcoder"
    llm = _ollama_llm_sync(model_name)
    prompt = _sql_prompt(question)
    log.debug("generate_sql: prompt=\n%s", prompt)

    try:
        response = llm.invoke(prompt) or ""
        sql_query = response.strip()
        log.info(
            "generate_sql: raw_response_len=%d model=%s", len(response), model_name
        )
        log.debug(
            "generate_sql: response_truncated=%s",
            (response[:500] + "...") if len(response) > 500 else response,
        )
        inner["sql_query"] = sql_query
        inner["generate_output"] = (
            f"Generated SQL:\n{sql_query}"
            if sql_query
            else "Model returned empty output."
        )
        log.info(
            "generate_sql: sql_query_len=%d head=%s", len(sql_query), sql_query[:20]
        )
        return _wrap(inner)
    except Exception as e:
        msg = str(e)
        log.error(
            "generate_sql: first attempt failed with model=%s error=%s", model_name, msg
        )
        resolved = _resolve_ollama_model_sync(preferred_model)
        if resolved != model_name:
            try:
                log.info("generate_sql: retrying with resolved model=%s", resolved)
                llm = _ollama_llm_sync(resolved)
                response = llm.invoke(prompt) or ""
                sql_query = response.strip()
                inner["sql_query"] = sql_query
                inner["generate_output"] = (
                    f"Generated SQL:\n{sql_query}"
                    if sql_query
                    else f"Model returned empty output (model={resolved})."
                )
                log.info(
                    "generate_sql: retry success raw_len=%d model=%s",
                    len(response),
                    resolved,
                )
                return _wrap(inner)
            except Exception as e2:
                inner["sql_query"] = ""
                inner["generate_output"] = f"Model invocation failed: {e2}"
                log.exception("generate_sql: retry model invocation error")
                return _wrap(inner)
        else:
            inner["sql_query"] = ""
            inner["generate_output"] = f"Model invocation failed: {e}"
            log.exception("generate_sql: model invocation error (no alternative)")
            return _wrap(inner)


def validate_sql(state: dict) -> dict:
    inner = _get_inner(state)
    log.info("Node validate_sql entered. Inner keys: %s", list(inner.keys()))
    sql_query = (inner.get("sql_query") or "").strip()
    lower = sql_query.lower()
    valid = (
        ("select" in lower)
        and (" from " in lower)
        and (not "drop" in lower)
        and (not "delete" in lower)
        and (not "update" in lower)
        and (not "insert" in lower)
    )

    inner["valid"] = bool(valid)
    if valid:
        inner["validate_output"] = "Validation passed ✅"
        log.info("validate_sql: passed")
    else:
        reason = "missing SELECT/FROM or empty SQL"
        inner["validate_output"] = (
            f"Validation failed ❌ ({reason})\n{sql_query or '<empty>'}"
        )
        log.warning("validate_sql: failed. sql_query_len=%d", len(sql_query))
    return _wrap(inner)


def execute_sql(state: dict) -> dict:
    inner = _get_inner(state)
    log.info("Node execute_sql entered. Inner keys: %s", list(inner.keys()))
    sql_query = (inner.get("sql_query") or "").strip()
    if not sql_query:
        inner["error"] = "Execution skipped: SQL is empty."
        inner["execute_output"] = inner["error"]
        log.warning("execute_sql: empty SQL; skipping execution.")
        return _wrap(inner)

    try:
        conn = _pg_conn()
        cur = conn.cursor()
        log.info("execute_sql: executing SQL:\n%s", sql_query)
        cur.execute(sql_query)
        rows = cur.fetchall()
        headers = [desc[0] for desc in cur.description]
        cur.close()
        conn.close()

        inner["rows"] = rows
        inner["headers"] = headers
        inner["execute_output"] = f"Executed successfully, {len(rows)} rows fetched."
        log.info("execute_sql: success rows=%d headers=%s", len(rows), headers)
    except Exception as e:
        inner["error"] = f"Execution failed: {e}"
        inner["execute_output"] = inner["error"]
        log.exception("execute_sql: DB execution error")
    return _wrap(inner)


# --- Build graph ---

graph = StateGraph(dict)
graph.add_node("generate_sql", generate_sql)
graph.add_node("validate_sql", validate_sql)
graph.add_node("execute_sql", execute_sql)

graph.set_entry_point("generate_sql")
graph.add_edge("generate_sql", "validate_sql")


def is_valid(state: dict) -> str:
    inner = _get_inner(state)
    decision = "execute_sql" if inner.get("valid") else END
    log.info("Conditional is_valid decision: %s", decision)
    return decision


graph.add_conditional_edges(
    "validate_sql", is_valid, {"execute_sql": "execute_sql", END: END}
)
workflow = graph.compile()

# --- Stream extraction ---


def _extract_final_state(initial_state: dict) -> dict:
    final_inner: dict | None = None
    log.info(
        "Starting workflow.stream with initial_state keys: %s",
        list(initial_state.keys()),
    )
    for idx, chunk in enumerate(workflow.stream(initial_state), start=1):
        log.info("Stream chunk #%d: keys=%s", idx, list(chunk.keys()))
        for node_name, payload in chunk.items():
            if (
                isinstance(payload, dict)
                and "value" in payload
                and isinstance(payload["value"], dict)
            ):
                final_inner = dict(payload["value"])
                log.info(
                    "Stream payload (wrapped). node=%s keys=%s",
                    node_name,
                    list(final_inner.keys()),
                )
            else:
                log.warning(
                    "Stream payload ignored for node=%s: type=%s keys=%s",
                    node_name,
                    type(payload),
                    list(payload.keys()) if isinstance(payload, dict) else None,
                )
    final_state = final_inner or {}
    log.info("Final inner state keys: %s", list(final_state.keys()))
    return final_state


# --- FastAPI routes ---


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/query")
async def api_query(request: Request, payload: QueryPayload):
    question = (payload.question or "").strip()
    log.info("POST /api/query question='%s' len=%d", question, len(question))
    if not question:
        log.warning("Empty question in request.")
        return JSONResponse(
            {
                "error": {
                    "question": payload.question,
                    "generate": "No question provided.",
                    "validate": "Validation failed ❌ (empty question)",
                    "execute": None,
                    "sql": "",
                }
            },
            status_code=400,
        )

    # Wrap initial state to ensure nodes receive "value" with "question"
    initial_state = {"value": {"question": question}}

    # Run workflow via stream and capture final inner state
    try:
        inner = _extract_final_state(initial_state)
    except Exception as e:
        log.exception("Workflow streaming failed.")
        return JSONResponse(
            {
                "error": {
                    "question": question,
                    "generate": f"Workflow stream failed: {e}",
                    "validate": None,
                    "execute": None,
                    "sql": None,
                }
            },
            status_code=500,
        )

    # Collect stage outputs
    sql_text = inner.get("sql_query") or ""
    stage_outputs = {
        "question": question,
        "sql": sql_text,
        "generate": inner.get("generate_output"),
        "validate": inner.get("validate_output"),
        "execute": inner.get("execute_output"),
    }
    log.info("Stage outputs: %s", stage_outputs)

    # If validation failed, return JSON with full stage outputs
    if not inner.get("valid"):
        log.warning("Validation failed; returning 400.")
        return JSONResponse({"error": stage_outputs}, status_code=400)

    # If execution failed, return JSON with full stage outputs
    if "error" in inner:
        log.error("Execution error; returning 500.")
        return JSONResponse({"error": stage_outputs}, status_code=500)

    # Success: build CSV in-memory
    rows = inner.get("rows", [])
    headers = inner.get("headers", [])

    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(headers)
    writer.writerows(rows)
    buffer.seek(0)
    csv_text = buffer.getvalue()

    # If client asked for direct download ?download=true return CSV attachment
    download_flag = request.query_params.get("download", "false").lower() in (
        "1",
        "true",
        "yes",
    )
    if download_flag:
        # Return the CSV as an attachment (no SQL in-file)
        buffer.seek(0)
        headers_out = {"Content-Disposition": 'attachment; filename="query_result.csv"'}
        return StreamingResponse(buffer, media_type="text/csv", headers=headers_out)

    # Default: return JSON with generated SQL and CSV text for client to show SQL and offer download
    return {"sql": sql_text, "csv": csv_text}
