// Verify immutable image bundles in the browser before either editor sees pixels.
window.HTH_REFERENCE_IMAGES = (() => {
  'use strict';
  const defaults = window.HTH_REFERENCE_DEFAULTS;
  const sha256 = async bytes => [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
    .map(value => value.toString(16).padStart(2, '0')).join('');

  async function readBundle(file, bundle, goldenSet, progress = () => {}) {
    if (file.name !== bundle.asset || file.size !== bundle.size)
      throw Error(`Expected the immutable ${bundle.asset} release asset (${bundle.size} bytes).`);
    if (bundle.format !== 'zip/store' || !Array.isArray(bundle.images))
      throw Error('Expected a frozen ZIP/store Golden Set image manifest.');
    const pages = new Map(goldenSet.pages.map(page => [Number(page.global_ordinal), page.image_sha256]));
    const expected = new Map();
    for (const record of bundle.images) {
      const match = String(record.path).match(/^raw\/fs_(\d+)\.(png|jpe?g|tiff?|webp)$/i);
      const number = match && Number(match[1]);
      if (!match || !pages.has(number) || pages.get(number) !== record.sha256 || expected.has(record.path))
        throw Error(`Image manifest does not match Golden Set page ${record.path}.`);
      expected.set(record.path, record);
    }
    if (expected.size !== pages.size) throw Error('Golden Set image manifest has incomplete page membership.');
    progress('Checking bundle SHA-256…');
    const bytes = await file.arrayBuffer();
    if (await sha256(bytes) !== bundle.sha256) throw Error('Golden Set image bundle SHA-256 mismatch.');
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
        throw Error('Golden Set bundle must contain ordinary stored ZIP entries.');
      const name = new TextDecoder().decode(new Uint8Array(bytes, position + 30, nameLength));
      const record = expected.get(name);
      if (!record || seen.has(name) || (record.size !== undefined && size !== record.size))
        throw Error(`Unexpected or duplicate Golden Set image: ${name}`);
      progress(`Checking image ${seen.size + 1}/${expected.size}…`);
      const imageBytes = bytes.slice(start, end);
      if (await sha256(imageBytes) !== record.sha256)
        throw Error(`Golden Set image SHA-256 mismatch: ${name}`);
      const extension = name.split('.').at(-1).toLowerCase();
      const mime = extension === 'png' ? 'image/png' : extension === 'jpg' || extension === 'jpeg' ? 'image/jpeg' : extension === 'webp' ? 'image/webp' : 'image/tiff';
      images.push(new File([imageBytes], name.split('/').at(-1), { type: mime }));
      seen.add(name);
      position = end;
    }
    if (seen.size !== expected.size || position + 4 > bytes.byteLength || view.getUint32(position, true) !== 0x02014b50)
      throw Error(`Golden Set bundle is incomplete: found ${seen.size}/${expected.size} images.`);
    return images;
  }

  function readGs0002Bundle(file, progress = () => {}) {
    const images = defaults.detector.pages.map(page => ({
      global_ordinal: Number(page.global_ordinal),
      path: `raw/fs_${String(page.global_ordinal).padStart(4, '0')}.png`,
      sha256: page.image_sha256,
    }));
    const bundle = { ...defaults.imageBundle, format: 'zip/store', images };
    return readBundle(file, bundle, defaults.detector, progress);
  }
  return { readBundle, readGs0002Bundle };
})();
