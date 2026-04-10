import logging
from fastapi import FastAPI
import gradio as gr
from app.constants.phonemes import ALL_PHONEMES
from app.model.utils import (
    process_audio_bytes,
    run_model_inference,
    parse_delta_value,
)
from app.constants.environmental_variables import (
    QUALITY_PROB_GAP_DELTA,
    DURATION_PROB_GAP_DELTA,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

css = """
.phoneme-scores { display: flex; flex-wrap: wrap; justify-content: center; gap: 15px; }
.phoneme-container { text-align: center; padding: 10px; border: 1px solid #ddd; border-radius: 8px; }
.phoneme { font-size: 1.5em; font-weight: bold; margin-bottom: 5px; }
.score { padding: 8px 12px; border-radius: 5px; color: white; font-weight: bold; }
.good { background-color: #28a745; } /* Green */
.medium { background-color: #ffc107; } /* Yellow */
.bad { background-color: #dc3545; } /* Red */
"""


def get_score_class(score, score_type):
    if score_type == "quality":
        if score == 1:
            return "good"
        if score == 2:
            return "medium"
        return "bad"
    else:
        return "good" if score == 1 else "bad"


def generate_html_output(phonemes, scores, score_type):
    html_output = (
        "<div class='phoneme-section'>"
        f"<h3 class='scores-title'>{'Quality Scores' if score_type == 'quality' else 'Duration Scores'}</h3>"
        "</div><div class='phoneme-scores'>"
    )

    for token, score in zip(phonemes, scores):
        score_class = get_score_class(score, score_type)
        html_output += (
            "<div class='phoneme-container'>"
            f"<div class='phoneme'>{token}</div>"
            f"<div class='score {score_class}'>{score}</div>"
            "</div>"
        )

    html_output += "</div>"
    return html_output


def validate_phonemes_text(phoneme_text):
    phoneme_list = phoneme_text.strip().split()
    if not phoneme_list:
        return "Please enter at least one phoneme."
    for p in phoneme_list:
        if p not in ALL_PHONEMES:
            return f"Invalid phoneme: {p}"
    return None


def create_gradio_app(app: FastAPI) -> gr.Blocks:
    def score_phonemes(phoneme_text, audio_file):
        if audio_file is None:
            return (
                "<p style='text-align:center; color:red;'>Please upload a .wav audio file.</p>",
                "",
            )

        phonemes_validation_error = validate_phonemes_text(phoneme_text)
        if phonemes_validation_error:
            return (
                f"<p style='text-align:center; color:red;'>{phonemes_validation_error}</p>",
                "",
            )

        if not hasattr(app.state, "model_artifacts"):
            return (
                "<p style='text-align:center; color:red;'>Model is not loaded.</p>",
                "",
            )

        try:
            with open(audio_file, "rb") as f:
                audio_bytes = f.read()
            waveform = process_audio_bytes(audio_bytes)
        except ValueError as e:
            return f"<p style='text-align:center; color:red;'>{e}</p>", ""
        except Exception:
            logger.exception("Audio processing failed.")
            return (
                "<p style='text-align:center; color:red;'>Invalid audio file.</p>",
                "",
            )

        phoneme_list = phoneme_text.strip().split()
        model, processor = app.state.model_artifacts

        scores_by_head = run_model_inference(
            waveform,
            phoneme_list,
            model,
            processor,
            {
                "quality": parse_delta_value(QUALITY_PROB_GAP_DELTA),
                "duration": parse_delta_value(DURATION_PROB_GAP_DELTA),
            },
            0,
        )
        q_scores = scores_by_head.get("quality")
        d_scores = scores_by_head.get("duration")
        if q_scores is None or d_scores is None:
            logger.error("Model output is missing required heads: %s", scores_by_head.keys())
            return (
                "<p style='text-align:center; color:red;'>Internal model error.</p>",
                "",
            )

        quality_html = generate_html_output(phoneme_list, q_scores, "quality")
        duration_html = generate_html_output(phoneme_list, d_scores, "duration")

        return quality_html, duration_html

    with gr.Blocks() as demo:
        gr.HTML(f"<style>{css}</style>")
        gr.Markdown(
            """
            # Phoneme Pronunciation and Duration Scorer
            Enter the phonemes directly into the text box (space-separated).
            Then, upload a `.wav` file or record the audio of the pronounced word.
            The application will provide a pronunciation (quality) and duration score for each phoneme.

            Scores legend:
            - Quality: 1 (good), 2 (medium), 3 (bad)
            - Duration: 1 (good), 2 (bad)
            """
        )

        with gr.Row():
            phoneme_text_input = gr.Textbox(label="Phonemes (space-separated)")
            audio_input = gr.Audio(type="filepath", label="Audio File (.wav)")

        btn = gr.Button("Generate Scores", variant="primary")

        gr.Markdown("---")
        gr.Markdown("## Results")

        with gr.Row():
            phoneme_output_html = gr.HTML()
            duration_output_html = gr.HTML()

        btn.click(
            fn=score_phonemes,
            inputs=[phoneme_text_input, audio_input],
            outputs=[phoneme_output_html, duration_output_html],
        )

    return demo
