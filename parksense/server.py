from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from parksense.multimodal.decision import assess_multimodal_risk
from parksense.spiral.inference import predict_spiral_image
from parksense.spiral.model import ARTIFACT_PATH, predict_records
from parksense.handwriting.model import SPIRAL_IMAGE_ARTIFACT_PATH


def json_response(handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status.value)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class ParkSenseHandler(BaseHTTPRequestHandler):
    model_path = ARTIFACT_PATH
    spiral_image_model_path = SPIRAL_IMAGE_ARTIFACT_PATH

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            json_response(self, HTTPStatus.NOT_FOUND, {"error": "Route not found"})
            return
        json_response(
            self,
            HTTPStatus.OK,
            {
                "status": "ok",
                "spiral_feature_model_ready": Path(self.model_path).expanduser().exists(),
                "spiral_image_model_ready": Path(self.spiral_image_model_path).expanduser().exists(),
                "spiral_feature_model_path": str(Path(self.model_path).expanduser().resolve()),
                "spiral_image_model_path": str(Path(self.spiral_image_model_path).expanduser().resolve()),
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(content_length) or b"{}")
        except json.JSONDecodeError:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "Request body must be valid JSON"})
            return

        if self.path == "/predict/spiral":
            self._predict_spiral_features(payload)
            return

        if self.path == "/predict/spiral/image":
            self._predict_spiral_image(payload)
            return

        if self.path == "/predict/multimodal":
            self._predict_multimodal(payload)
            return

        json_response(self, HTTPStatus.NOT_FOUND, {"error": "Route not found"})

    def _predict_spiral_features(self, payload: dict[str, Any]) -> None:
        if not Path(self.model_path).expanduser().exists():
            json_response(
                self,
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "Spiral feature model artifact missing. Run finalize_spiral_models.py first."},
            )
            return

        records = payload.get("records") if isinstance(payload, dict) else None
        if records is None:
            records = payload.get("features", payload) if isinstance(payload, dict) else payload

        try:
            predictions = predict_records(records, model_path=self.model_path)
        except Exception as exc:  # pragma: no cover - defensive API boundary
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        json_response(self, HTTPStatus.OK, {"predictions": predictions})

    def _predict_spiral_image(self, payload: dict[str, Any]) -> None:
        if not Path(self.spiral_image_model_path).expanduser().exists():
            json_response(
                self,
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "Spiral image model artifact missing. Run finalize_spiral_models.py first."},
            )
            return

        image_path = payload.get("image_path") if isinstance(payload, dict) else None
        if not image_path:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "image_path is required"})
            return

        try:
            prediction = predict_spiral_image(image_path, model_path=self.spiral_image_model_path)
        except Exception as exc:  # pragma: no cover - defensive API boundary
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        json_response(self, HTTPStatus.OK, {"prediction": prediction})

    def _predict_multimodal(self, payload: dict[str, Any]) -> None:
        spiral_score = payload.get("spiral_score")
        if spiral_score is None and payload.get("spiral_image_path"):
            if not Path(self.spiral_image_model_path).expanduser().exists():
                json_response(
                    self,
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"error": "Spiral image model artifact missing. Run finalize_spiral_models.py first."},
                )
                return
            try:
                spiral_prediction = predict_spiral_image(
                    payload["spiral_image_path"],
                    model_path=self.spiral_image_model_path,
                )
                spiral_score = spiral_prediction["recommended_multimodal_score"]
            except Exception as exc:  # pragma: no cover
                json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
        else:
            spiral_prediction = None

        if spiral_score is None and payload.get("voice_score") is None and payload.get("tapping_score") is None:
            response = {
                "decision": {
                    "decision_label": "unscorable_input",
                    "summary": "No valid modality score is available yet. Repeat the spiral image or provide voice/tapping scores.",
                    "disclaimer": "Research screening prototype only. Not a medical diagnosis.",
                }
            }
            if spiral_prediction is not None:
                response["spiral_prediction"] = spiral_prediction
            json_response(self, HTTPStatus.OK, response)
            return

        try:
            decision = assess_multimodal_risk(
                spiral_score=spiral_score,
                voice_score=payload.get("voice_score"),
                tapping_score=payload.get("tapping_score"),
            )
        except Exception as exc:  # pragma: no cover - defensive API boundary
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        response = {"decision": decision}
        if spiral_prediction is not None:
            response["spiral_prediction"] = spiral_prediction
        json_response(self, HTTPStatus.OK, response)

    def log_message(self, format: str, *args: Any) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the ParkSense spiral prediction API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-path", default=str(ARTIFACT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ParkSenseHandler.model_path = Path(args.model_path).expanduser().resolve()
    server = ThreadingHTTPServer((args.host, args.port), ParkSenseHandler)
    print(f"Serving ParkSense API on http://{args.host}:{args.port}")
    print("Routes: GET /health, POST /predict/spiral, POST /predict/spiral/image, POST /predict/multimodal")
    server.serve_forever()


if __name__ == "__main__":
    main()
