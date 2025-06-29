document.addEventListener('DOMContentLoaded', function() {
  const app = document.getElementById('app');
  
  app.innerHTML = `
    <div class="container">
      <div class="header">
        <h1>Pyramid Flow</h1>
        <p>Open-source video generation model with autoregressive video generation and training-free video enhancement</p>
      </div>

      <div class="alert">
        <h3>Environment Limitation</h3>
        <p>This Python project cannot run in the current WebContainer environment due to missing dependencies (PyTorch, CUDA, etc.). This interface provides an overview of the project structure and capabilities.</p>
      </div>

      <div class="demo-section">
        <h2>🎬 Video Generation Capabilities</h2>
        <p>Pyramid Flow supports high-quality video generation with various resolutions and frame rates</p>
      </div>

      <div class="grid">
        <div class="card">
          <h2>🚀 Key Features</h2>
          <ul class="feature-list">
            <li>Autoregressive video generation</li>
            <li>Training-free video enhancement</li>
            <li>Multiple resolution support</li>
            <li>Efficient DiT architecture</li>
            <li>Causal Video VAE</li>
            <li>Flow matching scheduler</li>
          </ul>
        </div>

        <div class="card">
          <h2>📊 Model Specifications</h2>
          <h3>Supported Resolutions:</h3>
          <ul class="feature-list">
            <li>768x768 (Image generation)</li>
            <li>384x640 (Video generation)</li>
            <li>576x1024 (Video generation)</li>
          </ul>
          <h3>Video Capabilities:</h3>
          <ul class="feature-list">
            <li>5-second video generation</li>
            <li>24 FPS output</li>
            <li>Text-to-video synthesis</li>
          </ul>
        </div>
      </div>

      <div class="card">
        <h2>📁 Project Structure</h2>
        <div class="file-structure">
          <div class="folder">pyramid_dit/</div>
          <div class="file">├── flux_modules/ - Flux-based DiT modules</div>
          <div class="file">├── mmdit_modules/ - Multi-modal DiT modules</div>
          <div class="file">└── pyramid_dit_for_video_gen_pipeline.py</div>
          <div class="folder">video_vae/</div>
          <div class="file">├── modeling_causal_vae.py - Causal Video VAE</div>
          <div class="file">├── modeling_enc_dec.py - Encoder/Decoder</div>
          <div class="file">└── causal_video_vae_wrapper.py</div>
          <div class="folder">diffusion_schedulers/</div>
          <div class="file">├── scheduling_flow_matching.py</div>
          <div class="file">└── scheduling_cosine_ddpm.py</div>
          <div class="folder">train/</div>
          <div class="file">├── train_pyramid_flow.py</div>
          <div class="file">└── train_video_vae.py</div>
          <div class="folder">improvements/</div>
          <div class="file">├── enhanced_inference.py</div>
          <div class="file">├── quality_enhancer.py</div>
          <div class="file">└── memory_optimization.py</div>
        </div>
      </div>

      <div class="grid">
        <div class="card">
          <h2>🛠️ Available Scripts</h2>
          <h3>Training:</h3>
          <ul class="feature-list">
            <li>train_pyramid_flow.sh</li>
            <li>train_causal_video_vae.sh</li>
          </ul>
          <h3>Inference:</h3>
          <ul class="feature-list">
            <li>inference_multigpu.py</li>
            <li>app_multigpu.py</li>
          </ul>
          <h3>Demos:</h3>
          <ul class="feature-list">
            <li>video_generation_demo.ipynb</li>
            <li>image_generation_demo.ipynb</li>
            <li>causal_video_vae_demo.ipynb</li>
          </ul>
        </div>

        <div class="card">
          <h2>⚡ Enhancements</h2>
          <p>The improvements/ directory contains several enhancements:</p>
          <ul class="feature-list">
            <li>Enhanced inference pipeline</li>
            <li>Quality enhancement algorithms</li>
            <li>Memory optimization techniques</li>
            <li>Batch processing capabilities</li>
          </ul>
        </div>
      </div>

      <div class="card">
        <h2>🔧 To Run This Project Locally</h2>
        <p>Since this project requires PyTorch and CUDA dependencies, you'll need to run it in a proper Python environment:</p>
        <ol style="margin-left: 2rem; margin-top: 1rem;">
          <li>Install Python 3.8+ with CUDA support</li>
          <li>Run: <code style="background: #f7fafc; padding: 0.2rem 0.4rem; border-radius: 4px;">pip install -r requirements.txt</code></li>
          <li>Download the model weights</li>
          <li>Run the demo notebooks or use the web interface via <code style="background: #f7fafc; padding: 0.2rem 0.4rem; border-radius: 4px;">python app.py</code></li>
        </ol>
      </div>
    </div>
  `;
});