# Border Energy Validator detector

The stable method identifier is `border_energy`. The detector uses Contour Quadrilateral as its geometry generator, then measures Sobel gradient energy in a narrow band along each proposed page border. It rejects geometrically plausible candidates whose borders lack sufficient or sufficiently consistent image energy.

Configuration: `config/detectors/border_energy.json`.

The calibration space controls contour geometry, gradient smoothing, border-band width, minimum energy, side consistency, and fusion weights. Debug artifacts preserve the normalized border-energy image and the validated contour overlay.
