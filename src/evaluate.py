# src/evaluate.py

"""
Threshold tuning / evaluation using enrollment crops.

Assumptions:
    - Enrollment crops exist under:
          data/enroll/<name>/*.jpg

    - Crops are already aligned to 112x112.

    - ArcFaceEmbedderONNX from embed.py is used to generate
      L2-normalized embeddings.

Evaluation:
    - Genuine pairs:
          samples from the same person

    - Impostor pairs:
          samples from different people

Outputs:
    - Summary statistics for genuine/impostor cosine distances.
    - Threshold sweep.
    - Suggested threshold based on target FAR.

Run:
    python -m src.evaluate
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .embed import ArcFaceEmbedderONNX


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    # Enrollment crop directory
    enroll_dir: Path = Path("data/enroll")

    # Dataset requirements
    min_imgs_per_person: int = 5
    max_imgs_per_person: int = 80

    # Target false-accept rate
    target_far: float = 0.01

    # Threshold sweep:
    #     start, end, step
    thresholds: Tuple[
        float,
        float,
        float,
    ] = (0.10, 1.20, 0.01)

    # Required aligned crop size
    require_size: Tuple[
        int,
        int,
    ] = (112, 112)


# ---------------------------------------------------------------------
# Similarity / distance
# ---------------------------------------------------------------------


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Calculate cosine similarity.

    ArcFaceEmbedderONNX already returns L2-normalized
    embeddings, so cosine similarity is simply their
    dot product.
    """

    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)

    if a.size != b.size:
        raise ValueError(
            "Cannot compare embeddings with different dimensions."
        )

    return float(np.dot(a, b))


def cosine_distance(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Calculate cosine distance.

    distance = 1 - cosine similarity
    """

    return 1.0 - cosine_similarity(a, b)


# ---------------------------------------------------------------------
# Dataset / IO
# ---------------------------------------------------------------------


def list_people(
    cfg: EvalConfig,
) -> List[Path]:
    """
    Find all identity directories under the enrollment directory.
    """

    if not cfg.enroll_dir.exists():
        raise FileNotFoundError(
            f"Enroll directory not found: "
            f"{cfg.enroll_dir}. "
            "Run enroll.py first."
        )

    return sorted(
        [
            path
            for path in cfg.enroll_dir.iterdir()
            if path.is_dir()
        ]
    )


def _is_aligned_crop(
    image: np.ndarray,
    required_size: Tuple[int, int],
) -> bool:
    """
    Check whether an image has the expected dimensions.
    """

    height, width = image.shape[:2]

    required_width = int(required_size[0])
    required_height = int(required_size[1])

    return (
        width == required_width
        and height == required_height
    )


def load_embeddings_for_person(
    embedder: ArcFaceEmbedderONNX,
    person_dir: Path,
    cfg: EvalConfig,
) -> List[np.ndarray]:
    """
    Load aligned crops for one person and generate embeddings.
    """

    image_paths = sorted(
        person_dir.glob("*.jpg")
    )[: cfg.max_imgs_per_person]

    embeddings: List[np.ndarray] = []

    for image_path in image_paths:
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            print(
                f"[evaluate] Could not read: "
                f"{image_path}"
            )
            continue

        # Keep evaluation restricted to correctly aligned crops.
        if (
            cfg.require_size is not None
            and not _is_aligned_crop(
                image,
                cfg.require_size,
            )
        ):
            print(
                f"[evaluate] Skipping wrong-sized crop: "
                f"{image_path}"
            )
            continue

        try:
            result = embedder.embed(image)
            embeddings.append(result.embedding)

        except Exception as exc:
            print(
                f"[evaluate] Failed to embed "
                f"{image_path}: {exc}"
            )

    return embeddings


# ---------------------------------------------------------------------
# Pairwise evaluation
# ---------------------------------------------------------------------


def pairwise_distances(
    embeddings_a: List[np.ndarray],
    embeddings_b: List[np.ndarray],
    same: bool,
) -> List[float]:
    """
    Generate cosine distances.

    same=True:
        Compare unique pairs within embeddings_a.

    same=False:
        Compare every embedding in A against every
        embedding in B.
    """

    distances: List[float] = []

    if same:
        for i in range(len(embeddings_a)):
            for j in range(
                i + 1,
                len(embeddings_a),
            ):
                distances.append(
                    cosine_distance(
                        embeddings_a[i],
                        embeddings_a[j],
                    )
                )

    else:
        for embedding_a in embeddings_a:
            for embedding_b in embeddings_b:
                distances.append(
                    cosine_distance(
                        embedding_a,
                        embedding_b,
                    )
                )

    return distances


# ---------------------------------------------------------------------
# Threshold evaluation
# ---------------------------------------------------------------------


def sweep_thresholds(
    genuine: np.ndarray,
    impostor: np.ndarray,
    cfg: EvalConfig,
) -> List[Tuple[float, float, float]]:
    """
    Evaluate FAR and FRR across a range of thresholds.

    FAR:
        Impostor accepted.

        distance <= threshold

    FRR:
        Genuine sample rejected.

        distance > threshold
    """

    start, end, step = cfg.thresholds

    thresholds = np.arange(
        start,
        end + 1e-9,
        step,
        dtype=np.float32,
    )

    results: List[
        Tuple[float, float, float]
    ] = []

    for threshold in thresholds:
        far = (
            float(
                np.mean(
                    impostor <= threshold
                )
            )
            if impostor.size
            else 0.0
        )

        frr = (
            float(
                np.mean(
                    genuine > threshold
                )
            )
            if genuine.size
            else 0.0
        )

        results.append(
            (
                float(threshold),
                far,
                frr,
            )
        )

    return results


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------


def describe(
    values: np.ndarray,
) -> str:
    """
    Return summary statistics for a distance distribution.
    """

    if values.size == 0:
        return "n=0"

    return (
        f"n={values.size} "
        f"mean={values.mean():.3f} "
        f"std={values.std():.3f} "
        f"p05={np.percentile(values, 5):.3f} "
        f"p50={np.percentile(values, 50):.3f} "
        f"p95={np.percentile(values, 95):.3f}"
    )


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------


def main() -> None:
    cfg = EvalConfig()

    # -------------------------------------------------------------
    # Initialize ArcFace embedder
    # -------------------------------------------------------------

    embedder = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx",
        input_size=(112, 112),
        debug=False,
    )

    # -------------------------------------------------------------
    # Find enrolled people
    # -------------------------------------------------------------

    people_dirs = list_people(cfg)

    if not people_dirs:
        print(
            "No enrolled people found."
        )
        return

    # -------------------------------------------------------------
    # Load embeddings
    # -------------------------------------------------------------

    per_person: Dict[
        str,
        List[np.ndarray],
    ] = {}

    for person_dir in people_dirs:
        name = person_dir.name

        embeddings = load_embeddings_for_person(
            embedder,
            person_dir,
            cfg,
        )

        if len(embeddings) >= cfg.min_imgs_per_person:
            per_person[name] = embeddings

        else:
            print(
                f"Skipping {name}: only "
                f"{len(embeddings)} valid aligned "
                f"crops "
                f"(need >= "
                f"{cfg.min_imgs_per_person})."
            )

    names = sorted(
        per_person.keys()
    )

    if not names:
        print(
            "Not enough valid data to evaluate. "
            "Enroll more samples."
        )
        return

    # -------------------------------------------------------------
    # Genuine distances
    # -------------------------------------------------------------

    genuine_all: List[float] = []

    for name in names:
        genuine_all.extend(
            pairwise_distances(
                per_person[name],
                per_person[name],
                same=True,
            )
        )

    # -------------------------------------------------------------
    # Impostor distances
    # -------------------------------------------------------------

    impostor_all: List[float] = []

    for i in range(len(names)):
        for j in range(
            i + 1,
            len(names),
        ):
            impostor_all.extend(
                pairwise_distances(
                    per_person[names[i]],
                    per_person[names[j]],
                    same=False,
                )
            )

    genuine = np.asarray(
        genuine_all,
        dtype=np.float32,
    )

    impostor = np.asarray(
        impostor_all,
        dtype=np.float32,
    )

    # -------------------------------------------------------------
    # Distribution summary
    # -------------------------------------------------------------

    print(
        "\n=== Distance Distributions "
        "(cosine distance = "
        "1 - cosine similarity) ==="
    )

    print(
        "Genuine (same person):"
    )
    print(
        f"  {describe(genuine)}"
    )

    print(
        "Impostor (different persons):"
    )
    print(
        f"  {describe(impostor)}"
    )

    # -------------------------------------------------------------
    # Threshold sweep
    # -------------------------------------------------------------

    results = sweep_thresholds(
        genuine,
        impostor,
        cfg,
    )

    # -------------------------------------------------------------
    # Find best threshold
    #
    # Constraint:
    #     FAR <= target FAR
    #
    # Objective:
    #     Minimize FRR
    # -------------------------------------------------------------

    best: Optional[
        Tuple[float, float, float]
    ] = None

    for threshold, far, frr in results:
        if far <= cfg.target_far:
            if (
                best is None
                or frr < best[2]
            ):
                best = (
                    threshold,
                    far,
                    frr,
                )

    # -------------------------------------------------------------
    # Print threshold sweep
    # -------------------------------------------------------------

    print(
        "\n=== Threshold Sweep ==="
    )

    if results:
        stride = max(
            1,
            len(results) // 10,
        )

        for threshold, far, frr in results[::stride]:
            print(
                f"thr={threshold:.2f} "
                f"FAR={far * 100:5.2f}% "
                f"FRR={frr * 100:5.2f}%"
            )

    # -------------------------------------------------------------
    # Suggested threshold
    # -------------------------------------------------------------

    if best is not None:
        threshold, far, frr = best

        print(
            "\nSuggested threshold "
            f"(target FAR "
            f"{cfg.target_far * 100:.1f}%):"
        )

        print(
            f"  cosine distance threshold: "
            f"{threshold:.2f}"
        )

        print(
            f"  FAR: {far * 100:.2f}%"
        )

        print(
            f"  FRR: {frr * 100:.2f}%"
        )

        # Equivalent cosine similarity threshold.
        similarity_threshold = 1.0 - threshold

        print(
            "\nEquivalent cosine similarity "
            f"threshold: "
            f"{similarity_threshold:.3f}"
        )

    else:
        print(
            "\nNo threshold in the configured "
            "range met the target FAR "
            f"<= {cfg.target_far * 100:.1f}%."
        )

        print(
            "Try widening the threshold sweep "
            "range or collecting more varied "
            "enrollment samples."
        )

    print()


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------


if __name__ == "__main__":
    main()