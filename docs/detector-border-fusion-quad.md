# Border Fusion Quad

`border_fusion_quad` is a side-level fusion generator. It runs the reviewed baseline forms of Radial Edge Search, Polar Boundary Voting, and Gradient Boundary Voting, converts each successful child quadrilateral into top/right/bottom/left line hypotheses, and searches mixed-source side combinations for the strongest geometrically valid page quadrilateral.

Configuration: `config/detectors/border_fusion_quad.json`.

This differs from detector-level consensus: the returned quadrilateral does not have to come from any one child detector. One child may supply the top edge while another supplies a side that it sees more clearly. A fused proposal must use at least two distinct child sources and is independently validated against Sobel gradient support on all four selected sides. Child detectors use their baseline parameters so calibration of the fusion layer remains isolated from calibration of each child algorithm.

Winner debug artifacts show the independent child quadrilaterals, the fusion gradient field, and the selected fused quadrilateral. Verbose debug colors the four selected sides by their source detector.
