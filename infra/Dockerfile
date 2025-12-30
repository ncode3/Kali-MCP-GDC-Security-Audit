FROM python:3.11-slim

RUN groupadd -r aari && useradd -r -g aari aari
WORKDIR /app

RUN pip install --no-cache-dir numpy pandas

COPY src/ ./src/
COPY mcp-servers/ ./mcp-servers/

RUN mkdir -p /app/data /app/reports && chown -R aari:aari /app
USER aari

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["python", "-m", "mcp-servers.datacenter.server"]
