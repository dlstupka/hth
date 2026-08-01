# Gradient Boundary Voting

The stable method identifier is `gradient_vote`. The detector accumulates Sobel gradient evidence into horizontal and vertical boundary votes, selects opposing page boundaries, and returns their quadrilateral. It is a generator based on distributed photometric evidence rather than connected contours or segmentation.

Configuration: `config/detectors/gradient_vote.json`.
