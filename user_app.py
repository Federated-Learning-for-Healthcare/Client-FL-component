"""
Legacy CLI entry point.
Used for quick local testing.
Production deployment uses src/service/app.py
"""


# user_app.py
import flwr as fl

from src.config.loader import load_config
from src.config.builder import build_from_config


def main():
    cfg = load_config("conf/config.yaml")
    print("CONFIG LOADED:", cfg)
    built = build_from_config(cfg)

    print("🏥 Hospital Client Connecting to Server...")
    fl.client.start_client(
        server_address=built.server_address,
        client=built.client.to_client()
    )


if __name__ == "__main__":
    main()
