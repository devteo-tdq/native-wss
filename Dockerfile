# --- Stage 1: Builder ---
FROM rust:1.83-slim-bookworm AS builder

WORKDIR /app

# Copy full metadata (quan trọng để cache đúng)
COPY Cargo.toml Cargo.lock ./

# Nếu bạn KHÔNG cần build.rs → bỏ dòng này
# Nếu vẫn cần thì giữ
COPY build.rs ./

# Tạo dummy src để cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs

# Build để cache deps (KHÔNG cần --bin ở bước này)
RUN cargo build --release || true

# Copy source thật
RUN rm -rf src
COPY src ./src

# Build coordinator thật
RUN cargo build --release --bin coordinator

# --- Stage 2: Runner ---
FROM debian:bookworm-slim

# Cài cert + libc (tránh crash runtime)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary
COPY --from=builder /app/target/release/coordinator .

EXPOSE 9000

CMD ["./coordinator"]
