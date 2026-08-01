# Multi-stage build for topolsea-server (Phase C13).
FROM rust:1.85-bookworm AS builder
WORKDIR /app
COPY . .
RUN cargo build --release -p dv-server

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/topolsea-server /usr/local/bin/topolsea-server
RUN mkdir -p /data
ENV TOPOLSEA_DATA_DIR=/data
EXPOSE 6333
ENTRYPOINT ["topolsea-server"]
CMD ["--data-dir", "/data", "--bind", "0.0.0.0:6333"]
