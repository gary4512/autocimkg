FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY autocimkg_mcp.py .

RUN useradd --system --uid 10001 mcp
USER mcp

ENV AUTOCIMKG_MCP_HOST=0.0.0.0 \
    AUTOCIMKG_MCP_PORT=8000 \
    PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["python", "autocimkg_mcp.py"]
