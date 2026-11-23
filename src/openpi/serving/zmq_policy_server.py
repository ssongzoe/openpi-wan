from dataclasses import dataclass
import io
from typing import Any

import msgpack
import numpy as np
import tyro
import zmq

from openpi.training import config as _config
from openpi.policies import policy_config


DEFAULT_MODEL_SERVER_PORT = 5555


class MsgSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer._encode, use_bin_type=True)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer._decode, raw=False, strict_map_key=False)

    @staticmethod
    def _encode(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        raise TypeError(f"Non-serializable type: {type(obj)}")

    @staticmethod
    def _decode(obj: Any) -> Any:
        if isinstance(obj, dict) and "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj


@dataclass
class ServerConfig:
    """Configuration for running a pi0.5 ZMQ inference server."""

    config_name: str = "pi05_rby1"
    """OpenPI train config name."""

    checkpoint_dir: str = "checkpoints/pi05_rby1/rby1_abs_v1/5000"
    """Path to trained checkpoint directory."""

    host: str = "0.0.0.0"
    """Host address."""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """ZMQ port."""

    timeout_ms: int = 30000
    """Receive timeout in milliseconds."""


class Pi05ZMQPolicyServer:
    def __init__(self, policy: Any, host: str, port: int, timeout_ms: int):
        self.policy = policy
        self.address = f"tcp://{host}:{port}"

        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(self.address)

    def run(self) -> None:
        print(f"pi0.5 ZMQ policy server listening on {self.address}")

        while True:
            try:
                raw = self.sock.recv()
                request = MsgSerializer.from_bytes(raw)

                endpoint = request.get("endpoint")
                data = request.get("data", {})

                if endpoint == "ping":
                    response = True

                elif endpoint == "reset":
                    response = {"ok": True}

                elif endpoint == "get_action":
                    observation = data["observation"]
                    result = self.policy.infer(observation)
                    response = result

                else:
                    response = {"error": f"Unknown endpoint: {endpoint}"}

                self.sock.send(MsgSerializer.to_bytes(response))

            except zmq.Again:
                continue
            except KeyboardInterrupt:
                print("\nShutting down server...")
                break
            except Exception as exc:
                self.sock.send(MsgSerializer.to_bytes({"error": repr(exc)}))

        self.close()

    def close(self) -> None:
        self.sock.close()
        self.ctx.term()


def main(config: ServerConfig) -> None:
    print("Starting pi0.5 ZMQ inference server...")
    print(f"  Config name: {config.config_name}")
    print(f"  Checkpoint dir: {config.checkpoint_dir}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")

    train_config = _config.get_config(config.config_name)
    policy = policy_config.create_trained_policy(train_config, config.checkpoint_dir)

    server = Pi05ZMQPolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
        timeout_ms=config.timeout_ms,
    )
    server.run()


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))