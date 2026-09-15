# src/embed.py

"""
ArcFace ONNX embedding demo.

Pipeline:
    Camera
      -> Haar face detection
      -> MediaPipe Tasks FaceLandmarker (5-point landmarks)
      -> 5-point face alignment
      -> ArcFace ONNX embedding
      -> L2-normalized embedding
      -> Embedding visualization

Run:
    python -m src.embed

Keys:
    q - quit
    p - print embedding statistics to terminal
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import time 

import cv2
import numpy as np
import onnxruntime as ort

from .haar_5pt import Haar5ptDetector, align_face_5pt


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------


@dataclass
class EmbeddingResult:
    """
    Result returned by the ArcFace embedder.

    embedding:
        L2-normalized embedding vector, float32.

    norm_before:
        L2 norm of the raw model output before normalization.

    dim:
        Embedding dimensionality.
    """

    embedding: np.ndarray
    norm_before: float
    dim: int


# ---------------------------------------------------------------------
# ArcFace ONNX Embedder
# ---------------------------------------------------------------------


class ArcFaceEmbedderONNX:
    """
    ArcFace / InsightFace-style ONNX embedder.

    Input:
        Aligned 112x112 BGR image.

    Output:
        L2-normalized embedding vector.
    """

    def __init__(
        self,
        model_path: str = "models/embedder_arcface.onnx",
        input_size: Tuple[int, int] = (112, 112),
        debug: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.in_w, self.in_h = map(int, input_size)
        self.debug = debug

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"ArcFace ONNX model not found: {self.model_path}"
            )

        self.sess = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        inputs = self.sess.get_inputs()
        outputs = self.sess.get_outputs()

        if not inputs:
            raise RuntimeError("ArcFace ONNX model has no inputs.")

        if not outputs:
            raise RuntimeError("ArcFace ONNX model has no outputs.")

        self.in_name = inputs[0].name
        self.out_name = outputs[0].name

        if self.debug:
            print("[embed] model loaded:", self.model_path)
            print("[embed] input name:", self.in_name)
            print("[embed] input shape:", inputs[0].shape)
            print("[embed] output name:", self.out_name)
            print("[embed] output shape:", outputs[0].shape)

    def _preprocess(
        self,
        aligned_bgr: np.ndarray,
    ) -> np.ndarray:
        """
        Convert aligned BGR image into ArcFace model input.
        """

        if aligned_bgr is None or aligned_bgr.size == 0:
            raise ValueError("Received an empty aligned face image.")

        if aligned_bgr.ndim != 3 or aligned_bgr.shape[2] != 3:
            raise ValueError(
                "Expected a BGR image with shape (H, W, 3)."
            )

        if aligned_bgr.shape[:2] != (self.in_h, self.in_w):
            aligned_bgr = cv2.resize(
                aligned_bgr,
                (self.in_w, self.in_h),
                interpolation=cv2.INTER_LINEAR,
            )

        # ArcFace preprocessing:
        # BGR -> RGB
        # uint8 -> float32
        # normalize roughly to [-1, 1]
        rgb = cv2.cvtColor(
            aligned_bgr,
            cv2.COLOR_BGR2RGB,
        ).astype(np.float32)

        rgb = (rgb - 127.5) / 128.0

        # HWC -> CHW -> NCHW
        x = np.transpose(
            rgb,
            (2, 0, 1),
        )[None, ...]

        return np.ascontiguousarray(
            x,
            dtype=np.float32,
        )

    @staticmethod
    def _l2_normalize(
        vector: np.ndarray,
        eps: float = 1e-12,
    ) -> Tuple[np.ndarray, float]:
        """
        L2-normalize an embedding vector.
        """

        norm = float(np.linalg.norm(vector))

        if norm < eps:
            raise ValueError(
                "ArcFace model returned a near-zero embedding."
            )

        normalized = vector / norm

        return (
            normalized.astype(np.float32),
            norm,
        )

    def embed(
        self,
        aligned_bgr: np.ndarray,
    ) -> EmbeddingResult:
        """
        Generate an L2-normalized ArcFace embedding.
        """

        x = self._preprocess(aligned_bgr)

        output = self.sess.run(
            [self.out_name],
            {
                self.in_name: x,
            },
        )[0]

        raw_embedding = output.reshape(-1).astype(
            np.float32,
            copy=False,
        )

        embedding, norm_before = self._l2_normalize(
            raw_embedding
        )

        return EmbeddingResult(
            embedding=embedding,
            norm_before=norm_before,
            dim=embedding.size,
        )


# ---------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------


def draw_text_block(
    img: np.ndarray,
    lines: list[str],
    origin: Tuple[int, int] = (10, 30),
    scale: float = 0.7,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> None:
    """
    Draw multiple lines of text vertically.
    """

    x, y = origin

    for line in lines:
        cv2.putText(
            img,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2,
            cv2.LINE_AA,
        )

        y += int(28 * scale)


def draw_embedding_matrix(
    img: np.ndarray,
    embedding: np.ndarray,
    top_left: Tuple[int, int] = (10, 220),
    cell_scale: int = 6,
    title: str = "embedding",
) -> Tuple[int, int]:
    """
    Visualize an embedding vector as a heatmap matrix.
    """

    dimension = embedding.size

    if dimension == 0:
        return 0, 0

    cols = int(np.ceil(np.sqrt(dimension)))
    rows = int(np.ceil(dimension / cols))

    matrix = np.zeros(
        (rows, cols),
        dtype=np.float32,
    )

    matrix.flat[:dimension] = embedding

    minimum = matrix.min()
    maximum = matrix.max()

    normalized = (
        matrix - minimum
    ) / (
        maximum - minimum + 1e-6
    )

    gray = (
        normalized * 255
    ).astype(np.uint8)

    heatmap = cv2.applyColorMap(
        gray,
        cv2.COLORMAP_JET,
    )

    heatmap = cv2.resize(
        heatmap,
        (
            cols * cell_scale,
            rows * cell_scale,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    x, y = top_left

    heat_h, heat_w = heatmap.shape[:2]
    image_h, image_w = img.shape[:2]

    if (
        x < 0
        or y < 0
        or x + heat_w > image_w
        or y + heat_h > image_h
    ):
        return 0, 0

    img[
        y : y + heat_h,
        x : x + heat_w,
    ] = heatmap

    cv2.putText(
        img,
        title,
        (x, max(15, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )

    return heat_w, heat_h


def emb_preview_str(
    embedding: np.ndarray,
    n: int = 8,
) -> str:
    """
    Create a short textual preview of the embedding.
    """

    n = min(n, embedding.size)

    values = " ".join(
        f"{value:+.3f}"
        for value in embedding[:n]
    )

    return f"vec[0:{n}]: {values} ..."


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Calculate cosine similarity.

    Embeddings produced by ArcFaceEmbedderONNX are already
    L2-normalized, so this is simply their dot product.
    """

    if a.size != b.size:
        raise ValueError(
            "Cannot compare embeddings with different dimensions."
        )

    return float(np.dot(a, b))


# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------


def main(
    camera_index: int = 1,
) -> None:
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {camera_index}."
        )

    detector = Haar5ptDetector(
        min_size=(70, 70),
        smooth_alpha=0.80,
        debug=False,
    )

    embedder = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx",
        input_size=(112, 112),
        debug=False,
    )

    previous_embedding: Optional[np.ndarray] = None

    print(
        "Embedding Demo running.\n"
        "Press 'q' to quit.\n"
        "Press 'p' to print embedding statistics."
    )

    fps_start = time.time()
    fps_frames = 0
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print("[embed] failed to read frame.")
                break

            vis = frame.copy()

            faces = detector.detect(
                frame,
                max_faces=1,
            )

            info: list[str] = []

            if faces:
                face = faces[0]

                # -----------------------------------------------------
                # Draw detection
                # -----------------------------------------------------

                cv2.rectangle(
                    vis,
                    (face.x1, face.y1),
                    (face.x2, face.y2),
                    (0, 255, 0),
                    2,
                )

                for x, y in face.kps.astype(int):
                    cv2.circle(
                        vis,
                        (int(x), int(y)),
                        3,
                        (0, 255, 0),
                        -1,
                    )

                # -----------------------------------------------------
                # Align face
                # -----------------------------------------------------

                aligned, _matrix = align_face_5pt(
                    frame,
                    face.kps,
                    out_size=(112, 112),
                )

                # -----------------------------------------------------
                # Generate embedding
                # -----------------------------------------------------

                result = embedder.embed(aligned)

                info.append(
                    f"embedding dim: {result.dim}"
                )

                info.append(
                    f"norm(before L2): "
                    f"{result.norm_before:.2f}"
                )

                # -----------------------------------------------------
                # Compare against previous frame
                # -----------------------------------------------------

                if previous_embedding is not None:
                    similarity = cosine_similarity(
                        previous_embedding,
                        result.embedding,
                    )

                    info.append(
                        f"cos(prev,this): {similarity:.3f}"
                    )

                previous_embedding = result.embedding

                # -----------------------------------------------------
                # Aligned face preview
                # -----------------------------------------------------

                aligned_small = cv2.resize(
                    aligned,
                    (160, 160),
                    interpolation=cv2.INTER_LINEAR,
                )

                image_h, image_w = vis.shape[:2]

                preview_x1 = image_w - 170
                preview_y1 = 10
                preview_x2 = image_w - 10
                preview_y2 = 170

                if (
                    preview_x1 >= 0
                    and preview_y2 <= image_h
                ):
                    vis[
                        preview_y1:preview_y2,
                        preview_x1:preview_x2,
                    ] = aligned_small

            else:
                info.append("no face")

            # ---------------------------------------------------------
            # Text information
            # ---------------------------------------------------------

            draw_text_block(
                vis,
                info,
                origin=(10, 30),
            )

            # ---------------------------------------------------------
            # Embedding heatmap
            # ---------------------------------------------------------

            if faces and previous_embedding is not None:
                heatmap_x = 10
                heatmap_y = 220
                cell_scale = 6

                heat_w, heat_h = draw_embedding_matrix(
                    vis,
                    previous_embedding,
                    top_left=(
                        heatmap_x,
                        heatmap_y,
                    ),
                    cell_scale=cell_scale,
                    title="embedding heatmap",
                )

                if heat_w > 0:
                    cv2.putText(
                        vis,
                        emb_preview_str(
                            previous_embedding
                        ),
                        (
                            heatmap_x,
                            heatmap_y + heat_h + 28,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (200, 200, 200),
                        2,
                        cv2.LINE_AA,
                    )

            # ---------------------------------------------------------
            # FPS
            # ---------------------------------------------------------

            fps_frames += 1

            elapsed = time.time() - fps_start

            if elapsed >= 1.0:
                fps = fps_frames / elapsed
                fps_frames = 0
                fps_start = time.time()

            cv2.putText(
                vis,
                f"FPS: {fps:.1f}",
                (
                    10,
                    vis.shape[0] - 15,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # ---------------------------------------------------------
            # Display
            # ---------------------------------------------------------

            cv2.imshow(
                "Face Embedding",
                vis,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if (
                key == ord("p")
                and previous_embedding is not None
            ):
                print("\n[embedding]")
                print(" dim:", previous_embedding.size)
                print(
                    " min/max:",
                    previous_embedding.min(),
                    previous_embedding.max(),
                )
                print(
                    " first10:",
                    previous_embedding[:10],
                )

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------


if __name__ == "__main__":
    main()