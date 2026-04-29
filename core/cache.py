import os
import redis
import certifi

def get_redis_client():
    redis_url = os.getenv("REDIS_URL")
    return redis.from_url(
        redis_url,
        ssl_ca_certs=certifi.where(),
        ssl_cert_reqs="required",  
        socket_keepalive=True,          # keeps TCP connection alive
        socket_timeout=10,              # fail fast if Redis is unresponsive
        socket_connect_timeout=10,
        retry_on_timeout=True,          # auto-retry on timeout
        health_check_interval=30,       # ping every 30s to prevent idle drop
    )