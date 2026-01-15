import os
import logging
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


def download_and_save_model_snapshot():
    QUALITY_MODEL_REPO_ID = os.getenv("QUALITY_MODEL_REPO_ID")
    DURATION_MODEL_REPO_ID = os.getenv("DURATION_MODEL_REPO_ID")
    QUALITY_MODEL_PATH = os.getenv("QUALITY_MODEL_PATH")
    DURATION_MODEL_PATH = os.getenv("DURATION_MODEL_PATH")

    if not all(
        [
            QUALITY_MODEL_REPO_ID,
            QUALITY_MODEL_PATH,
            DURATION_MODEL_REPO_ID,
            DURATION_MODEL_PATH,
        ]
    ):
        error_msg = "All model environment variables (REPO_ID and PATH) must be set."
        logger.error(error_msg)
        raise ValueError(error_msg)

    ignore_patterns = ["*.h5", "*.ot", "*.msgpack"]

    for model_id, model_path in [
        (QUALITY_MODEL_REPO_ID, QUALITY_MODEL_PATH),
        (DURATION_MODEL_REPO_ID, DURATION_MODEL_PATH),
    ]:
        logger.info(f"Starting download for model: {model_id}")
        logger.info(f"Model will be saved to: {model_path}")

        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=model_path,
                ignore_patterns=ignore_patterns,
            )
            logger.info(f"Successfully downloaded {model_id}")

        except Exception as e:
            logger.error(f"Failed to download {model_id}. Error: {e}")
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_and_save_model_snapshot()
