"""摄像头 WebRTC 推流到检测引擎，并使用 cv2 实时显示检测结果。"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fractions
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import grpc
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import VideoFrame


def _ensure_local_src_importable() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_ensure_local_src_importable()

from artemis_cam.protos.detector import common_pb2  # noqa: E402
from artemis_cam.protos.detector import webrtc_detector_pb2 as pb2  # noqa: E402
from artemis_cam.protos.detector import webrtc_detector_pb2_grpc as pb2_grpc  # noqa: E402


DEFAULT_SERVER = "192.168.1.24:40001"
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30
FRAME_QUEUE_SIZE = 8
MAX_PENDING_DETECTIONS = 120
LATEST_DETECTION_HOLD_SECONDS = 0.25


@dataclass(slots=True)
class CameraFrame:
    img: object
    frame_id: int
    captured_at: float


class CameraCapture:
    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.display_queue: queue.Queue[CameraFrame] = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.send_queue: queue.Queue[CameraFrame] = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None
        self._start_monotonic = time.monotonic()

    def start(self) -> None:
        self._capture = cv2.VideoCapture(self.camera_index)
        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open camera index {self.camera_index}")

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.fps)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _run(self) -> None:
        assert self._capture is not None
        while not self._stop_event.is_set():
            ok, frame = self._capture.read()
            if not ok:
                time.sleep(0.02)
                continue

            frame_id = int((time.monotonic() - self._start_monotonic) * 1000.0)
            camera_frame = CameraFrame(
                img=frame,
                frame_id=frame_id,
                captured_at=time.monotonic(),
            )
            self._push_latest(self.display_queue, camera_frame)
            self._push_latest(self.send_queue, camera_frame)

    @staticmethod
    def _push_latest(target: queue.Queue[CameraFrame], item: CameraFrame) -> None:
        try:
            target.put_nowait(item)
            return
        except queue.Full:
            pass

        try:
            target.get_nowait()
        except queue.Empty:
            pass

        try:
            target.put_nowait(item)
        except queue.Full:
            pass


class CameraVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, capture: CameraCapture) -> None:
        super().__init__()
        self.capture = capture

    async def recv(self) -> VideoFrame:
        frame = await asyncio.to_thread(self.capture.send_queue.get)
        video_frame = VideoFrame.from_ndarray(frame.img, format="bgr24")
        video_frame.pts = frame.frame_id
        video_frame.time_base = fractions.Fraction(1, 1000)
        return video_frame


async def run_demo(
    server: str,
    camera_index: int,
    width: int,
    height: int,
    fps: int,
    score_threshold: float,
) -> None:
    capture = CameraCapture(
        camera_index=camera_index,
        width=width,
        height=height,
        fps=fps,
    )
    capture.start()

    channel = grpc.aio.insecure_channel(server)
    stub = pb2_grpc.WebRtcDetectorEngineStub(channel)
    pc = RTCPeerConnection()
    pc.addTrack(CameraVideoTrack(capture))

    latest_detection: pb2.StreamDetectionsReply | None = None
    latest_detection_at = 0.0
    detections_by_frame_id: OrderedDict[int, pb2.StreamDetectionsReply] = OrderedDict()
    detection_lock = threading.Lock()

    async def receive_detections(stream_id: str) -> None:
        nonlocal latest_detection, latest_detection_at
        async for reply in stub.StreamDetections(
            pb2.StreamDetectionsRequest(stream_id=stream_id)
        ):
            with detection_lock:
                detections_by_frame_id[reply.frame_id] = reply
                latest_detection = reply
                latest_detection_at = time.monotonic()
                while len(detections_by_frame_id) > MAX_PENDING_DETECTIONS:
                    detections_by_frame_id.popitem(last=False)

    detection_task: asyncio.Task[None] | None = None

    try:
        create_reply = await stub.CreateStream(
            pb2.CreateStreamRequest(
                config=common_pb2.StreamConfig(
                    video_codec="vp8",
                    width=width,
                    height=height,
                    score_threshold=score_threshold,
                )
            )
        )

        await pc.setRemoteDescription(
            RTCSessionDescription(
                sdp=create_reply.offer.sdp,
                type=create_reply.offer.type,
            )
        )
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        while pc.iceGatheringState != "complete":
            await asyncio.sleep(0.05)

        await stub.UpdateStream(
            pb2.StreamSignal(
                stream_id=create_reply.stream_id,
                answer=pb2.SessionDescription(
                    type=pc.localDescription.type,
                    sdp=pc.localDescription.sdp,
                ),
            )
        )

        detection_task = asyncio.create_task(receive_detections(create_reply.stream_id))
        print(f"Connected to {server}, stream_id={create_reply.stream_id}")
        print("Press q to quit.")

        while True:
            frame = await asyncio.to_thread(capture.display_queue.get)
            overlay = frame.img.copy()
            source = "none"

            with detection_lock:
                matched = detections_by_frame_id.pop(frame.frame_id, None)
                fallback = latest_detection
                fallback_age = time.monotonic() - latest_detection_at

            display_detection = matched
            if display_detection is not None:
                source = "matched"
            elif (
                fallback is not None
                and fallback_age <= LATEST_DETECTION_HOLD_SECONDS
            ):
                display_detection = fallback
                source = "latest"

            if display_detection is not None:
                for det in display_detection.detections:
                    if det.geometry.HasField("point"):
                        px = int(det.geometry.point.x)
                        py = int(det.geometry.point.y)
                        cv2.circle(overlay, (px, py), 10, (0, 0, 255), -1)
                        cv2.circle(overlay, (px, py), 14, (255, 255, 255), 2)
                        cv2.putText(
                            overlay,
                            f"{det.class_name} {det.score:.3f}",
                            (px + 12, py - 12),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )

            cv2.putText(
                overlay,
                f"frame_id={frame.frame_id} detection={source}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("artemis-cam webrtc demo", overlay)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        if detection_task is not None:
            detection_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await detection_task
        await pc.close()
        await channel.close()
        capture.stop()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use local camera via WebRTC and display realtime detections with cv2."
    )
    parser.add_argument("--server", default=DEFAULT_SERVER, help="gRPC server address")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Capture width")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Capture height")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Capture FPS")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Detection score threshold passed to the remote engine",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(
        run_demo(
            server=args.server,
            camera_index=args.camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
            score_threshold=args.score_threshold,
        )
    )


if __name__ == "__main__":
    main()
