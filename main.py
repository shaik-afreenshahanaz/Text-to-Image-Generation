!pip install -q --upgrade diffusers transformers accelerate sentencepiece gradio

import torch
from diffusers import StableDiffusionXLPipeline
import gradio as gr
import time
from PIL import Image

# --- Setup ---
MODEL_ID = "stabilityai/sdxl-turbo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"🚀 Loading SDXL-Turbo on {DEVICE}...")

try:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        variant="fp16" if DEVICE == "cuda" else None
    ).to(DEVICE)
    pipe.enable_attention_slicing()
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Load Failed: {e}")
    pipe = None

# --- Generation Logic ---  ← UNTOUCHED
def generate_image(prompt):
    if pipe is None:
        return None, "Model failed to load."

    start_time = time.time()
    image = pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=0.0
    ).images[0]

    duration = f"Generated in {time.time() - start_time:.2f}s"
    return image, duration

# ─────────────────────────────────────────────
#  Custom CSS  (only visual — zero logic change)
# ─────────────────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg:       #0b0d14;
    --surface:  #12151f;
    --card:     #181c2a;
    --border:   #252a3d;
    --accent:   #7c6aff;
    --accent2:  #ff6af0;
    --glow:     rgba(124, 106, 255, 0.35);
    --text:     #e8eaf6;
    --muted:    #7b80a0;
    --success:  #4ade80;
}

/* ── Base ── */
body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
    min-height: 100vh;
}

/* Animated mesh background */
.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(124,106,255,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 80%, rgba(255,106,240,0.08) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

/* ── Header ── */
.header-wrap {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
    position: relative;
}

.header-wrap h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(1.8rem, 4vw, 3rem) !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #a78bff 0%, #ff6af0 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0 !important;
    letter-spacing: -0.02em !important;
}

.header-wrap p {
    color: var(--muted) !important;
    font-size: 0.95rem !important;
    margin-top: 0.4rem !important;
}

/* ── Panel cards ── */
.panel-card {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    padding: 1.5rem !important;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

.panel-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    opacity: 0.7;
}

.panel-card:hover {
    border-color: rgba(124,106,255,0.4) !important;
    box-shadow: 0 0 30px var(--glow) !important;
}

/* ── Labels ── */
label span, .gr-label, .label-wrap span {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--accent) !important;
}

/* ── Textbox ── */
textarea, input[type="text"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.9rem 1.1rem !important;
    resize: none !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
    line-height: 1.6 !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(124,106,255,0.18) !important;
    outline: none !important;
}

textarea::placeholder {
    color: #3d4260 !important;
    font-style: italic !important;
}

/* ── Generate button ── */
button.primary {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    padding: 0.85rem 2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    position: relative;
    overflow: hidden;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s !important;
    box-shadow: 0 4px 20px rgba(124,106,255,0.4) !important;
}

button.primary::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, transparent 60%);
    pointer-events: none;
}

button.primary:hover {
    opacity: 0.92 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 28px rgba(124,106,255,0.55) !important;
}

button.primary:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 12px rgba(124,106,255,0.3) !important;
}

/* ── Image output ── */
.image-container, .output-image, [data-testid="image"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    min-height: 300px !important;
}

.image-container img {
    border-radius: 12px !important;
    width: 100% !important;
    object-fit: contain !important;
}

/* ── Status box ── */
.status-box textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--success) !important;
    font-family: 'DM Mono', 'Courier New', monospace !important;
    font-size: 0.82rem !important;
    padding: 0.7rem 1rem !important;
}

/* ── Row spacing ── */
.gr-row {
    gap: 1.5rem !important;
    align-items: flex-start !important;
}

.gr-column {
    gap: 1rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 6px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Footer hint ── */
.footer-hint {
    text-align: center;
    color: var(--muted) !important;
    font-size: 0.78rem !important;
    padding: 1.2rem 0 2rem !important;
    opacity: 0.6;
}

/* ── Badge ── */
.badge {
    display: inline-block;
    background: rgba(124,106,255,0.15);
    border: 1px solid rgba(124,106,255,0.3);
    color: #a78bff;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.72rem;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
"""

# ─────────────────────────────────────────────
#  Gradio UI  (layout + style only — logic unchanged)
# ─────────────────────────────────────────────
with gr.Blocks(title="TEXT-TO-IMAGE Generator") as demo:

    # ── Header ──
    gr.HTML("""
    <div class="header-wrap">
        <div class="badge">⚡ Fast Image Generator</div>
        <h1>Test-to-image Generator</h1>
        <p>Ultra-fast image synthesis powered by Stability AI</p>
    </div>
    """)

    # ── Main layout ──
    with gr.Row(equal_height=False):

        # Left column — prompt + button
        with gr.Column(scale=1, elem_classes=["panel-card"]):
            prompt_input = gr.Textbox(
                label="Enter Prompt",
                placeholder="A high-speed racing car on a neon track, cinematic lighting...",
                lines=5,
            )
            generate_btn = gr.Button(
                "⚡  Generate Image",
                variant="primary",
                size="lg",
            )
            gr.HTML("""
            <div style="margin-top:1rem; padding:0.85rem 1rem;
                        background:rgba(124,106,255,0.07);
                        border:1px solid rgba(124,106,255,0.18);
                        border-radius:10px; font-size:0.8rem; color:#7b80a0; line-height:1.7;">
                💡 <strong style="color:#a78bff">Tips for best results:</strong><br>
                • Add style words: <em>cinematic, 8k, dramatic lighting</em><br>
                • Be specific about composition and mood<br>
                • SDXL-Turbo excels at vivid, detailed scenes
            </div>
            """)

        # Right column — result + status
        with gr.Column(scale=1, elem_classes=["panel-card"]):
            output_image = gr.Image(
                label="Result Image",
                type="pil",
                height=380,
            )
            output_status = gr.Textbox(
                label="Status",
                interactive=False,
                max_lines=1,
                elem_classes=["status-box"],
                placeholder="Waiting for generation...",
            )

    # ── Footer ──
    gr.HTML('<div class="footer-hint">Powered by stabilityai/sdxl-turbo · 4 steps · guidance 0.0</div>')

    # ── Event binding — UNTOUCHED ──
    generate_btn.click(
        fn=generate_image,
        inputs=prompt_input,
        outputs=[output_image, output_status]
    )

# Launching with share=True creates the public URL
demo.launch(
    share=True,
    theme=gr.themes.Base(
        primary_hue="violet",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"],
    ),
    css=custom_css,
)
