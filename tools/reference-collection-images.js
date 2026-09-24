// Read the immutable GS0002 ZIP locally. No repository tree or network fetch is needed.
window.HTH_REFERENCE_IMAGES = (() => {
  'use strict';
  const defaults = window.HTH_REFERENCE_DEFAULTS;
  const bundle = defaults.imageBundle;
  const expected = new Map(defaults.detector.pages.map(page => [Number(page.global_ordinal), page.image_sha256]));
  const ordinal = name => {
    const match = name.match(/^raw\/fs_(\d{4})\.png$/);
    return match ? Number(match[1]) : null;
  };
  const sha256 = async bytes => [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
    .map(value => value.toString(16).padStart(2, '0')).join('');

  async function readGs0002Bundle(file, progress = () => {}) {
    if (file.name !== bundle.asset || file.size !== bundle.size)
      throw Error(`Choose the immutable ${bundle.asset} release asset (${bundle.size} bytes).`);
    progress('Checking bundle SHA-256…');
    const bytes = await file.arrayBuffer();
    if (await sha256(bytes) !== bundle.sha256) throw Error('GS0002 image bundle SHA-256 mismatch.');
    const view = new DataView(bytes), images = [], seen = new Set();
    let position = 0;
    while (position + 30 <= bytes.byteLength && view.getUint32(position, true) === 0x04034b50) {
      const flags = view.getUint16(position + 6, true);
      const method = view.getUint16(position + 8, true);
      const compressed = view.getUint32(position + 18, true);
      const size = view.getUint32(position + 22, true);
      const nameLength = view.getUint16(position + 26, true);
      const extraLength = view.getUint16(position + 28, true);
      const start = position + 30 + nameLength + extraLength, end = start + compressed;
      if (flags !== 0 || method !== 0 || compressed !== size || end > bytes.byteLength)
        throw Error('GS0002 bundle must contain ordinary stored ZIP entries.');
      const name = new TextDecoder().decode(new Uint8Array(bytes, position + 30, nameLength));
      const number = ordinal(name);
      if (number === null || !expected.has(number) || seen.has(number))
        throw Error(`Unexpected or duplicate GS0002 image: ${name}`);
      progress(`Checking image ${seen.size + 1}/${expected.size}…`);
      const imageBytes = bytes.slice(start, end);
      if (await sha256(imageBytes) !== expected.get(number))
        throw Error(`GS0002 image SHA-256 mismatch: ${name}`);
      images.push(new File([imageBytes], name.split('/').at(-1), { type: 'image/png' }));
      seen.add(number);
      position = end;
    }
    if (seen.size !== expected.size || view.getUint32(position, true) !== 0x02014b50)
      throw Error(`GS0002 bundle is incomplete: found ${seen.size}/${expected.size} images.`);
    return images;
  }
  return { readGs0002Bundle };
})();
