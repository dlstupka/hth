# Contour + Components Detector

The Contour + Components detector generates quadrilateral hypotheses from external contours and ranks them with independent connected-component evidence. Its stable method identifier is `contour_components`.

For each contour-derived quadrilateral, the detector compares the candidate with the envelope and selected foreground regions produced by the Connected Components detector. Component evidence combines:

- containment of selected component pixels inside the quadrilateral;
- overlap between the component envelope and candidate polygon;
- spatial spread of the component envelope across the candidate; and
- selected-component density inside the candidate.

The final score combines component evidence with contour area, rectangularity, and right-angle consistency. Component disagreement can reject a contour hypothesis through `minimum_component_score`.

`config/detectors/contour_components.json` contains reusable baseline parameters and calibration ranges. It contains no collection identity, Golden Set page ordinals, or document-specific exceptions.

Winner debug artifacts include contour hypotheses, component labels, selected components, the component envelope, a combined evidence overlay, the selected quadrilateral, the final overlay, and diagnostics JSON.
