FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[mock]"
# non-root, like a partner's platform team would insist on
RUN useradd -m agent && mkdir -p /app/traces && chown -R agent /app
USER agent
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/healthz')" || exit 1
CMD ["uvicorn", "nebius_agent.server:app", "--host", "0.0.0.0", "--port", "8080"]
