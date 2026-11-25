FROM rust:1.96.0-slim AS rust
# Compile the optional reader against the same libav ABI as the L4 workers.
# Only headers and pkg-config metadata enter the older glibc build environment.
FROM ubuntu:24.04 AS avheaders
RUN apt-get update && apt-get install -y --no-install-recommends libavformat-dev && mkdir /headers && cd / && cp -a --parents usr/include/*/libav* usr/lib/*/pkgconfig/libav*.pc usr/lib/*/pkgconfig/libswresample.pc /headers/
FROM ubuntu:22.04 AS build
COPY --from=avheaders /headers/ /
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates pkg-config clang libclang-dev python3 && rm -rf /var/lib/apt/lists/*
COPY --from=rust /usr/local/cargo /usr/local/cargo
COPY --from=rust /usr/local/rustup /usr/local/rustup
ENV PATH=/usr/local/cargo/bin:$PATH CARGO_HOME=/usr/local/cargo RUSTUP_HOME=/usr/local/rustup CARGO_TARGET_DIR=/target
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY vendor ./vendor
COPY dist ./dist
COPY scripts/install-slang scripts/slang-cache.py scripts/shader_compiler.py scripts/build-with-shader-cache scripts/slang_cache_launcher.rs ./scripts/
RUN python3 scripts/install-slang /opt/slang
ENV VORK_SLANGC=/opt/slang/bin/slangc
COPY docs ./docs
COPY examples ./examples
COPY libraries ./libraries
COPY skills ./skills
ARG VORK_SHADER_CACHE=dist/empty-shader-cache
COPY ${VORK_SHADER_CACHE} /shader-cache
ARG VORK_BUILD_COMMIT
RUN test -n "$VORK_BUILD_COMMIT" && VORK_BUILD_COMMIT="$VORK_BUILD_COMMIT" python3 scripts/build-with-shader-cache --cache /shader-cache -- cargo build --locked --release -p vork-cli
RUN grep -qx "cargo:rustc-cfg=vork_nvdecode" /target/release/build/vork-render-*/output
RUN objcopy --only-keep-debug /target/release/vork /target/release/vork.debug && strip --strip-unneeded /target/release/vork
RUN /target/release/vork --version && objdump -T /target/release/vork | sed -n 's/.*(GLIBC_\([0-9.]*\)).*/\1/p' | sort -Vu | tail -1
FROM ubuntu:22.04 AS smoke
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 libgcc-s1 && rm -rf /var/lib/apt/lists/*
COPY --from=build /target/release/vork /usr/local/bin/vork
RUN vork --version
FROM scratch AS artifact
COPY --from=build /shader-cache /shader-cache
COPY --from=smoke /usr/local/bin/vork /vork
COPY --from=build /target/release/vork-slang-filesystem /vork-slang-filesystem
COPY --from=build /src/crates/vork-render/src/ltc/LICENSE.ltc /licenses/LICENSE.ltc
COPY --from=build /src/crates/vork-render/src/ltc/LICENSE.three /licenses/LICENSE.three
COPY --from=build /src/crates/vork-render/src/ltc/README.md /licenses/LTC.md
