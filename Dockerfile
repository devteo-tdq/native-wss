# --- Stage 1: Builder ---
FROM rust:1.83-slim-bookworm AS builder

WORKDIR /app

# Cache dependencies
COPY Cargo.toml Cargo.lock build.rs ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release --bin coordinator || true

# Copy source thật
RUN rm -rf src
COPY src ./src

# Build coordinator
RUN cargo build --release --bin coordinator

# --- Stage 2: Runner ---
FROM debian:bookworm-slim

# Cài SSL certs
RUN apt-get update && apt-get install -y ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary coordinator
COPY --from=builder /app/target/release/coordinator .

# Mở port (tuỳ app bạn dùng, giữ 9000 nếu coordinator listen port này)
EXPOSE 9000

# Run coordinator
CMD ["./coordinator"]