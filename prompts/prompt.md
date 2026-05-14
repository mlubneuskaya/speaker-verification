**Role:** Senior AI Engineer and Audio Signal Processing Specialist.
**Goal:** Build a text-independent Voice Identification System using **SpeechBrain** and implement a rigorous testing suite to evaluate its robustness against various signal degradations.

### 1. System Architecture

* **Model Backbone:** Use a pre-trained model from SpeechBrain (trained on VoxCeleb1 and VoxCeleb2 train sets):
~~~
from speechbrain.inference.speaker import SpeakerRecognition
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
score, prediction = verification.verify_files("/content/example1.wav", "/content/example2.flac")
~~~
* **Enrollment Logic** 
  * To register a user, the system must capture/process three separate audio recordings (each 3–8 seconds long).
  * Processing: Extract an embedding for each sample, calculate the Arithmetic Mean of the three vectors,
  and then apply L2-normalization to the result to create a single "Master Speaker Template."
* **Database:** Implement a simple JSON-based local store. It must store:
  * User ID / Name.
  * Master Speaker Template (stored as a flat tensor/array).
  * Metadata (enrollment date, original sample reference).


* **Interface:** Create a minimal **Streamlit** interface to:
1. Add User: Upload/Record 3 samples -> Generate Master Template -> Save.
2. Verify User: Upload 1 sample -> Compare against DB -> Return Match/No Match.


### 2. Testing Suite Requirements

Implement a benchmarking module to run the following 7 experiments based on the provided specifications. For all tests, use **Equal Error Rate (EER)** and **Accuracy** as metrics.

* **Task 1: Baseline Effectiveness.** Test on 500+ samples. Create a balanced test set (50% "genuine" matches, 50% "impostor" attempts).
* **Task 2: Amplitude Scaling.** For 500 samples, randomly scale amplitude by factors of $\{25, 1, 0.04\}$. Compare results to the baseline.
* **Task 3: Resampling Impact.**
* **Naive:** Implement subsampling by keeping every 2nd, 5th, and 10th sample.
* **Interpolated:** Use standard resampling (2x, 5x, 10x) with proper interpolation.
* Evaluate how this affects the required clip length for a stable match.


* **Task 4: Gaussian Noise.** For 100 samples, add additive Gaussian noise at SNR levels of $40\text{dB}$, $20\text{dB}$, and $10\text{dB}$.
* **Task 5: Environmental Noise.** For 100 samples, mix in background noise (e.g., from UrbanSound8K) at SNR levels of $20\text{dB}$, $10\text{dB}$, and $0\text{dB}$.
* **Task 6: Lossy Compression.** Test 3 codecs (MP3, AAC, Opus) each at 3 different bitrate settings (low, medium, high quality).
* **Task 7: Reverberation.** Simulate room acoustics by convolving samples with Room Impulse Responses (RIR) from the OpenSLR SLR28 dataset.

### 3. Technical Constraints

* **Preprocessing:** All input audio must be resampled to $16\text{kHz}$ mono and passed through a VAD (Voice Activity Detector).
* **Embedding Logic:** All embeddings must be **L2-normalized** before comparison.
* **Comparison:** Use Cosine Similarity with a configurable threshold for "Match/No Match."
* **Language:** Python 3.10+. Libraries: `speechbrain`, `torchaudio`, `numpy`, `pandas`, `sqlite3`.
