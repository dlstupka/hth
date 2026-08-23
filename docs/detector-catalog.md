# HTH Document Detector Catalog

This is the human-readable catalog of the **authoritative geometry detector registry** in `hth/geometry/registry.py`. The registry, not this document, is the execution source of truth.

The current registry contains **47 detectors**:

- **30** classical / heuristic HTH detectors
- **3** calibrated fusion detectors
- **9** learned / model-backed page-mask detectors
- **4** direct OpenCV primitives
- **1** external-concept hybrid

The current production recommendation for Golden Set `HTH-0001` is **Fusion Gen3 — AMSRE + Doc-UFCN** (`amsre_doc_ufcn_fusion`), resolved from persisted authoritative calibration evidence. The catalog includes research and historical candidates as well as the production winner; presence in the registry does not imply production approval.

## Registry inventory

| # | Detector | Method ID | Category | Origin | Foundation |
|---:|---|---|---|---|---|
| 1 | [Contour](detector-contour.md) | `contour` | Classical / heuristic | HTH | OpenCV |
| 2 | [Contour Quadrilateral](detector-contour-quad.md) | `contour_quad` | Classical / heuristic | HTH | Contour geometry, OpenCV |
| 3 | [Contour + Components](detector-contour-components.md) | `contour_components` | Classical / heuristic | HTH | Contour geometry, Connected components, OpenCV |
| 4 | [Contour + GrabCut](detector-contour-grabcut.md) | `contour_grabcut` | Classical / heuristic | HTH | Contour geometry, GrabCut, OpenCV |
| 5 | [GrabCut + Contour](detector-grabcut-contour.md) | `grabcut_contour` | Classical / heuristic | HTH | GrabCut, Contour geometry, OpenCV |
| 6 | [Contour + Projection](detector-contour-projection.md) | `contour_projection` | Classical / heuristic | HTH | Contour geometry, Projection profiles, OpenCV |
| 7 | [Consensus Quad](detector-consensus-quad.md) | `consensus_quad` | Classical / heuristic | HTH | Contour Quadrilateral, Edge-Contour Hybrid, OpenCV |
| 8 | [Cross-Edge Contour](detector-cross-edge-contour.md) | `cross_edge_contour` | Classical / heuristic | HTH | Contour geometry, Cross-boundary intensity sampling, OpenCV |
| 9 | [Gradient Boundary Voting](detector-gradient-vote.md) | `gradient_vote` | Classical / heuristic | HTH | Sobel gradients, Projection voting, OpenCV |
| 10 | [Radial Edge Search](detector-radial-edge.md) | `radial_edge` | Classical / heuristic | HTH | Radial gradient search, OpenCV |
| 11 | [Adaptive Multi-Scale Radial Edge Search](detector-adaptive-multi-scale-radial-edge.md) | `adaptive_multi_scale_radial_edge` | Classical / heuristic | HTH | Multi-scale gradients, Adaptive angular refinement, Radial gradient search, OpenCV |
| 12 | [Fusion Gen2 — AMSRE + BFQ + SPBV + Page Background](detector-amsre-bfq-spbv-pbg.md) | `amsre_bfq_spbv_pbg` | Fusion | HTH | Adaptive Multi-Scale Radial Edge, Border Fusion Quad, Signed Polar Boundary Voting, Page Background, Side-level consensus, OpenCV |
| 13 | Fusion Gen3 — AMSRE + Doc-UFCN | `amsre_doc_ufcn_fusion` | Fusion | HTH | Adaptive Multi-Scale Radial Edge, Doc-UFCN, Confidence-gated rescue, Geometric disagreement, OpenCV |
| 14 | [Adaptive Radial Edge Search](detector-adaptive-radial-edge.md) | `adaptive_radial_edge` | Classical / heuristic | HTH | Two-pass radial gradient search, Adaptive angular refinement, OpenCV |
| 15 | [Multi-Scale Radial Edge Search](detector-multi-scale-radial-edge.md) | `multi_scale_radial_edge` | Classical / heuristic | HTH | Scale-space gradients, Radial gradient search, OpenCV |
| 16 | [Fusion Gen1 — MSRE + BFQ + SPBV + Page Background](detector-msre-bfq-spbv-pbg.md) | `msre_bfq_spbv_pbg` | Fusion | HTH | Multi-Scale Radial Edge, Border Fusion Quad, Signed Polar Boundary Voting, Page Background, Side-level consensus, OpenCV |
| 17 | [Projective Gradient Vote](detector-projective-gradient-vote.md) | `projective_gradient_vote` | Classical / heuristic | HTH | Sobel gradients, Line Segment Detector, Projective line intersections, OpenCV |
| 18 | [Border Fusion Quad](detector-border-fusion-quad.md) | `border_fusion_quad` | Classical / heuristic | HTH | Radial Edge Search, Polar Boundary Voting, Gradient Boundary Voting, Side-level fusion, OpenCV |
| 19 | [Border Energy Validator](detector-border-energy.md) | `border_energy` | Classical / heuristic | HTH | Contour geometry, Sobel border energy, OpenCV |
| 20 | [Edge-Contour Hybrid](detector-edge-contour.md) | `edge_contour` | Classical / heuristic | HTH | Contour geometry, Line Segment Detector, OpenCV |
| 21 | Convex Hull Detector | `convex_hull` | Classical / heuristic | HTH | Convex hull geometry, OpenCV |
| 22 | Distance Transform Detector | `distance_transform` | Classical / heuristic | HTH | Distance transform, Connected components, OpenCV |
| 23 | Polar Boundary Voting | `polar_boundary_vote` | Classical / heuristic | HTH | Polar gradient voting, OpenCV |
| 24 | Signed Polar Boundary Voting | `signed_polar_boundary_vote` | Classical / heuristic | HTH | Signed radial gradients, Polar boundary voting, OpenCV |
| 25 | Segment-Supported Polar Voting | `segment_supported_polar_vote` | Classical / heuristic | HTH | Polar boundary voting, Line Segment Detector, OpenCV |
| 26 | Star-Convex Boundary Optimization | `star_convex` | Classical / heuristic | HTH | Star-convex geometry, Radial mask support, OpenCV |
| 27 | Distance-Transform Rectangle Proposal | `distance_transform_rect` | Classical / heuristic | HTH | Distance transform, Rectangle proposal, OpenCV |
| 28 | Radon Boundary Projection | `radon_boundary` | Classical / heuristic | HTH | Projection-angle integration, Sobel gradients, OpenCV |
| 29 | Text Flow Envelope | `text_flow` | Classical / heuristic | HTH | Connected components, Text-line geometry, OpenCV |
| 30 | [Page Background](detector-page-background.md) | `page_background` | Classical / heuristic | HTH | Robust border background model, CIE Lab color distance, Negative-space segmentation, OpenCV |
| 31 | Whitespace Frame | `whitespace_frame` | Classical / heuristic | HTH | Negative-space segmentation, Morphology, OpenCV |
| 32 | Joint Rectangle Voting | `joint_rectangle_vote` | Classical / heuristic | HTH | Hough lines, Joint rectangle scoring, OpenCV |
| 33 | [ScanTailor Page Frame](detector-scantailor-page-frame.md) | `scantailor_page_frame` | External-concept hybrid | ScanTailor concepts / HTH | ScanTailor-style scan processing, Content-guided page framing, Projection profiles, OpenCV |
| 34 | [Eynollah Page-Mask Detector](detector-eynollah-page-mask.md) | `eynollah_page_mask` | Learned / model-backed | SBB Eynollah / HTH | Eynollah, Historical-document page extraction, TensorFlow SavedModel |
| 35 | [PageNet Page-Mask Detector](detector-pagenet-page-mask.md) | `pagenet_page_mask` | Learned / model-backed | PageNet / HTH | PageNet, Ohio Death Records, OpenCV DNN, Caffe |
| 36 | [docExtractor Page-Mask Detector](detector-docextractor-page-mask.md) | `docextractor_page_mask` | Learned / model-backed | docExtractor / HTH | docExtractor, ResUNet, Historical-document segmentation, PyTorch |
| 37 | [Mask R-CNN Page-Mask Detector](detector-mask-rcnn-page-mask.md) | `mask_rcnn_page_mask` | Learned / model-backed | HJDataset / LayoutParser / HTH | Mask R-CNN, ResNet-50 FPN, HJDataset historical-document layout segmentation, Detectron2 |
| 38 | [Doc-UFCN Page-Mask Detector](detector-doc-ufcn-page-mask.md) | `doc_ufcn_page_mask` | Learned / model-backed | Teklia Doc-UFCN / HTH | Doc-UFCN, Historical-document page segmentation, PyTorch |
| 39 | [dhSegment Page-Mask Detector](detector-dhsegment-page-mask.md) | `dhsegment_page_mask` | Learned / model-backed | dhSegment / HTH | dhSegment, ResNet-50, Learned page segmentation, TensorFlow SavedModel |
| 40 | Kraken Page-Mask Detector | `kraken_page_mask` | Learned / model-backed | Kraken BLLA / HTH | Kraken, BLLA, Historical-document layout segmentation, PyTorch |
| 41 | [Orli Page-Mask Detector](detector-orli-page-mask.md) | `orli_page_mask` | Learned / model-backed | Orli / HTH | Orli, Kraken 7 plugin, Historical-document baseline segmentation, PyTorch |
| 42 | [Learned Page-Mask Detector](detector-learned-page-mask.md) | `learned_page_mask` | Learned / model-backed | PageNet / HTH | PageNet, Learned page segmentation, OpenCV DNN, Caffe |
| 43 | [Connected Components](detector-components.md) | `components` | OpenCV primitive | OpenCV | OpenCV |
| 44 | [RANSAC](detector-ransac.md) | `ransac` | Classical / heuristic | HTH | RANSAC, OpenCV |
| 45 | [Hough Lines](detector-hough.md) | `hough` | OpenCV primitive | OpenCV | Hough transform, OpenCV |
| 46 | [Line Segment Detector](detector-lsd.md) | `lsd` | OpenCV primitive | OpenCV | LSD, OpenCV |
| 47 | [GrabCut](detector-grabcut.md) | `grabcut` | OpenCV primitive | OpenCV | GrabCut, OpenCV |

## Detector families

### Fusion

The fusion line records the progression from classical multi-evidence consensus to the current hybrid production detector:

- **[Fusion Gen2 — AMSRE + BFQ + SPBV + Page Background](detector-amsre-bfq-spbv-pbg.md)** — `amsre_bfq_spbv_pbg`; Adaptive Multi-Scale Radial Edge, Border Fusion Quad, Signed Polar Boundary Voting, Page Background, Side-level consensus, OpenCV.
- **Fusion Gen3 — AMSRE + Doc-UFCN** — `amsre_doc_ufcn_fusion`; Adaptive Multi-Scale Radial Edge, Doc-UFCN, Confidence-gated rescue, Geometric disagreement, OpenCV.
- **[Fusion Gen1 — MSRE + BFQ + SPBV + Page Background](detector-msre-bfq-spbv-pbg.md)** — `msre_bfq_spbv_pbg`; Multi-Scale Radial Edge, Border Fusion Quad, Signed Polar Boundary Voting, Page Background, Side-level consensus, OpenCV.

### Learned / model-backed

HTH deliberately evaluated multiple historical-document segmentation families rather than assuming one neural architecture would generalize to the collection:

- **[Eynollah Page-Mask Detector](detector-eynollah-page-mask.md)** — `eynollah_page_mask`; origin: SBB Eynollah / HTH.
- **[PageNet Page-Mask Detector](detector-pagenet-page-mask.md)** — `pagenet_page_mask`; origin: PageNet / HTH.
- **[docExtractor Page-Mask Detector](detector-docextractor-page-mask.md)** — `docextractor_page_mask`; origin: docExtractor / HTH.
- **[Mask R-CNN Page-Mask Detector](detector-mask-rcnn-page-mask.md)** — `mask_rcnn_page_mask`; origin: HJDataset / LayoutParser / HTH.
- **[Doc-UFCN Page-Mask Detector](detector-doc-ufcn-page-mask.md)** — `doc_ufcn_page_mask`; origin: Teklia Doc-UFCN / HTH.
- **[dhSegment Page-Mask Detector](detector-dhsegment-page-mask.md)** — `dhsegment_page_mask`; origin: dhSegment / HTH.
- **Kraken Page-Mask Detector** — `kraken_page_mask`; origin: Kraken BLLA / HTH.
- **[Orli Page-Mask Detector](detector-orli-page-mask.md)** — `orli_page_mask`; origin: Orli / HTH.
- **[Learned Page-Mask Detector](detector-learned-page-mask.md)** — `learned_page_mask`; origin: PageNet / HTH.

### Classical, heuristic, and geometric evidence

The remaining HTH detectors span contour geometry, radial and multi-scale edge search, gradient/polar voting, projection and transform methods, background/negative-space reasoning, line fitting, rectangle proposals, and hybrid evidence validation. These detectors remain useful as independent evidence, fusion children, calibration baselines, and regression controls even when they are not the production winner.

## Calibration coverage

The repository currently contains **47** detector calibration configuration files under `config/detectors/`, matching the registry one-for-one. Exhaustive, contracted, smoke, and optimizer searches operate over those declared discrete domains while preserving stable detector method IDs.

Detector quality and execution recommendations are persisted independently. See [Detector regression](regression.md), [Calibration selection](calibration-selection.md), [Golden Set](golden-set.md), and [Full-collection detector review](document-detector-review.md).

## Keeping this catalog current

A repository contract test verifies that every detector in the authoritative registry is present here and that every registered detector has a calibration configuration. Additions or removals therefore require an intentional catalog update rather than silently leaving public documentation behind.
