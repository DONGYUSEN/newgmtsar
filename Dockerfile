ARG BASE_IMAGE=debian:bookworm
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# Runtime + build dependencies for GMTSAR, plus requested tools.
RUN sed -i 's|http://deb.debian.org/debian|http://mirrors.tuna.tsinghua.edu.cn/debian|g; s|http://security.debian.org/debian-security|http://mirrors.tuna.tsinghua.edu.cn/debian-security|g' /etc/apt/sources.list.d/debian.sources \
    && printf '[global]\nindex-url = %s\ntrusted-host = %s\n' "$PIP_INDEX_URL" "$PIP_TRUSTED_HOST" > /etc/pip.conf \
    && apt-get update && apt-get install -y --no-install-recommends \
    aria2 \
    autoconf \
    build-essential \
    ca-certificates \
    csh \
    gfortran \
    gawk \
    gdal-bin \
    gmt \
    ghostscript \
    libblas-dev \
    libcurl4-openssl-dev \
    libfftw3-dev \
    libgdal-dev \
    libglib2.0-dev \
    libgomp1 \
    libgmt-dev \
    libhdf5-dev \
    liblapack-dev \
    libnetcdf-dev \
    libtiff-dev \
    libx11-dev \
    mawk \
    ncview \
    netcdf-bin \
    nano \
    perl \
    pkg-config \
    tzdata \
    wget \
    bc \
    && for f in /usr/include/hdf5/serial/*.h; do ln -sf "$f" "/usr/include/$(basename "$f")"; done \
    && for f in /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5*.so; do ln -sf "$f" "/usr/lib/x86_64-linux-gnu/$(basename "$f")"; done \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/src/GMTSAR

# Copy source tree (DInSAR is excluded by .dockerignore).
COPY . /opt/src/GMTSAR

# Some distros install gmtvector outside PATH; ensure configure can find it.
RUN if ! command -v gmtvector >/dev/null 2>&1; then \
      GV="$(find /usr -type f -name gmtvector 2>/dev/null | head -n1)"; \
      if [ -n "$GV" ]; then ln -sf "$GV" /usr/local/bin/gmtvector; fi; \
    fi

# Build + install GMTSAR into /opt/gmtsar.
RUN autoconf && \
    ./configure \
      --prefix=/opt/gmtsar \
      --with-orbits-dir=/opt/orbits && \
    make -j"$(nproc)" all && \
    make install && \
    ldconfig

ENV PATH="/opt/gmtsar/bin:${PATH}"
ENV GMTSAR_HOME="/opt/gmtsar"
ENV ORBITS_DIR="/opt/orbits"

ENV OMP_NUM_THREADS=6

WORKDIR /work
CMD ["/bin/bash"]
