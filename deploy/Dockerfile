# Multi-stage build for topolsea-server (Phase C13).
FROM rust:1.85-bookworm AS builder
WORKDIR /app
COPY . .
RUN cargo build --release -p dv-server

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates bash \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/topolsea-server /usr/local/bin/topolsea-server
RUN mkdir -p /data
ENV TOPOLSEA_DATA_DIR=/data
EXPOSE 6333
# Health without curl: bash /dev/tcp probe to /readyz
HEALTHCHECK --interval=10s --timeout=3s --retries=5 CMD bash -c 'exec 3<>/dev/tcp/127.0.0.1/6333 && printf "GET /readyz HTTP/1.0\r\n\r\n" >&3 && head -1 <&3 | grep -q 200'
ENTRYPOINT ["topolsea-server"]
CMD ["--data-dir", "/data", "--bind", "0.0.0.0:6333"]
