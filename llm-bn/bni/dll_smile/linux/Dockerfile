FROM gcc:9.4

WORKDIR /usr/src/app

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

COPY . .
COPY bni_smile.py bni_smile.py
COPY test_bni_smile.py test_bni_smile.py

RUN chmod +x make_bismile.sh

CMD ./make_bismile.sh