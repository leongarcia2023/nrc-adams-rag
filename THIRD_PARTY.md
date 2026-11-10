# Third-party resources

Resource records distinguish original NRC material from community extraction. The application retains publisher identifiers and revisions rather than treating an imported dataset as a complete or current NRC archive.

| Resource | Use | Publisher license / access |
| --- | --- | --- |
| [NRC ADAMS API](https://adams-search.nrc.gov/assets/APS-API-Guide.pdf) | Official discovery and individual document text, using the existing subscription | Public records; subscription required for API calls |
| [Davenport NRC regulatory dataset](https://huggingface.co/datasets/davenporten/nrc-regulatory-embeddings) | Imported source text with original links and dataset provenance | Dataset card lists MIT |
| [BGE small](https://huggingface.co/BAAI/bge-small-en-v1.5) and [Qdrant ONNX conversion](https://huggingface.co/Qdrant/bge-small-en-v1.5-onnx-Q) | Local 384-dimensional retrieval embeddings | MIT |
| [Qwen3 4B Instruct MLX conversion](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit) | Local query planning and cited generation | Apache-2.0 |
| [FermiBench](https://huggingface.co/datasets/atomic-canyon/FermiBench) | Import/evaluation adapter only; no benchmark files included | Card lists Apache-2.0; file access requires acceptance of Hugging Face conditions |

Pinned resources for this installation:

- Regulatory dataset: `50d49e8a51cf16c64d49db9074621182c1e224b8`. The original 37,734 rows produced 35,779 accepted source chunks across 3,183 documents/sections. The importer excluded 149 short chunks and 1,806 repeated chunks. Its normalized text produces a separate set of retrieval passages.
- Embedding weights: Qdrant revision `aa8f8b060edb00e03bfdd08813a2949946c8ba55`; ONNX SHA-256 `51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431`.
- Answer model: MLX community revision `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`; weights SHA-256 `2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f`.

The dataset's supplied embeddings require a different query model, so the importer discards them. The runtime uses local embeddings and does not send source text to a hosted model. The installed FastEmbed, USearch, MLX, ONNX Runtime and extraction libraries retain their own distribution metadata; exact installed versions are in the dependency lock file.

Optional scanned-page extraction uses the installed macOS [Vision text-recognition framework](https://developer.apple.com/documentation/vision/recognizing-text-in-images), [Quartz PDF rendering](https://developer.apple.com/library/archive/documentation/GraphicsImaging/Conceptual/drawingwithquartz2d/dq_pdf/dq_pdf.html), and the [PyObjC bridge](https://pyobjc.readthedocs.io/en/latest/apinotes/Vision.html). It sends no source pages to a hosted OCR service. Bridge packages are pinned to 13.0; OS version and recognition revision are recorded with subsequent extractions. The first recovery pilot predates the OS-version field and retains the recognition revision only.

Atomic Canyon's FERMI retrieval weights and FermiBench remain optional candidates for a measured comparison after authorized access. No commercial product, hosted retrieval service, or private corpus has been integrated. No gated files or account information have been shared to obtain access.

## Interface typeface

CMU Serif (Computer Modern Unicode), by Andrey V. Panov and the original Metafont authors, is bundled under the SIL Open Font License 1.1. The unmodified regular, bold, and italic WOFF2 files come from the owner’s reference page at https://leonmgarcia.com/uprate/assets/fonts/. The full upstream copyright and license are included at `adams_rag/web/fonts/OFL.txt`, obtained from https://mirrors.ctan.org/fonts/cm-unicode/doc/OFL.txt. Fonts are served locally; the interface does not request third-party assets.
